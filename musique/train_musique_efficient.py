import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, logging, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import trange, tqdm
import gc
import sys
import re
import string
from collections import Counter
import os

# Set environment variable to reduce memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.set_verbosity_error()

###########################
# Configuration
###########################
if len(sys.argv) < 2:
    print("Usage: python train_musique_efficient.py <qwen|mistral>")
    sys.exit(1)

if sys.argv[1] not in ['qwen', 'mistral']:
    print("First argument must be 'qwen' or 'mistral'")
    sys.exit(1)

if sys.argv[1] == 'qwen':
    MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
else:
    MODEL_A = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

NUM_LAYERS = 28 if 'Qwen' in MODEL_A else 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = f"kv_translators_musique.pth"
CACHE_DIR = "kv_cache_chunks"

# Memory-optimized configuration
NUM_PROMPTS = 1000
MAX_CTX_TOKENS = 1024
TRAIN_STEPS = 5000
BATCH_SIZE = 2
LR = 1e-3
SEED = 42
TRAINING_BATCH_SIZE = 8
COMPILE_MODEL = False
MIXED_PRECISION = True
USE_8BIT = True
CHUNK_SIZE = 50
GRADIENT_ACCUMULATION_STEPS = 2

# Evaluation settings
EVAL_EVERY = 1000
NUM_EVAL_SAMPLES = 50
MAX_NEW_TOKENS = 64
###########################

torch.manual_seed(SEED)
random.seed(SEED)
os.makedirs(CACHE_DIR, exist_ok=True)

def normalize_text(s):
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s

def calculate_f1_score(prediction, ground_truth):
    pred_tokens, truth_tokens = normalize_text(prediction).split(), normalize_text(ground_truth).split()
    if not truth_tokens and not pred_tokens: return 1.0
    if not truth_tokens or not pred_tokens: return 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def format_musique_prompt(example):
    """Format a MuSiQue example into a contextual QA prompt."""
    supporting_idx = example['question_decomposition'][0]['paragraph_support_idx']
    supporting_paragraphs = [
        p['paragraph_text'] 
        for p in example['paragraphs'] 
        if p['idx'] == supporting_idx
    ]
    context_paragraphs = "\n\n".join(supporting_paragraphs)
    question_text = example['question_decomposition'][0]['question']
    prompt = f"Context:\n{context_paragraphs}\n\nQuestion: {question_text} Answer:"
    return prompt, question_text, example['question_decomposition'][0].get('answer', '')

def generate_kv_cache_pair_batch(prompts, tokenizer_a, tokenizer_b, model_a, model_b, device, max_length):
    """Generate KV cache pairs sequentially to save memory."""
    # Process Model A
    inputs_a = tokenizer_a(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            out_a = model_a(**inputs_a, use_cache=True)
        kv_a = out_a.past_key_values
    
    source_keys = [kv_a[i][0].cpu() for i in range(NUM_LAYERS)]
    source_vals = [kv_a[i][1].cpu() for i in range(NUM_LAYERS)]
    del out_a, kv_a, inputs_a
    torch.cuda.empty_cache()
    
    # Process Model B
    inputs_b = tokenizer_b(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            out_b = model_b(**inputs_b, use_cache=True)
        kv_b = out_b.past_key_values
    
    target_keys = [kv_b[i][0].cpu() for i in range(NUM_LAYERS)]
    target_vals = [kv_b[i][1].cpu() for i in range(NUM_LAYERS)]
    del out_b, kv_b, inputs_b
    torch.cuda.empty_cache()
    
    return source_keys, source_vals, target_keys, target_vals

def generate_and_save_chunk(chunk_id, train_dataset, tokenizer_a, tokenizer_b, model_a, model_b, device, max_length, num_batches):
    """Generate a chunk of data and save to disk."""
    chunk_data = [[[] for _ in range(NUM_LAYERS)] for _ in range(4)]
    
    for _ in range(num_batches):
        prompts = []
        for _ in range(BATCH_SIZE):
            example = train_dataset[random.randint(0, len(train_dataset) - 1)]
            prompt, _, _ = format_musique_prompt(example)
            prompts.append(prompt)

        sk, sv, tk, tv = generate_kv_cache_pair_batch(prompts, tokenizer_a, tokenizer_b, model_a, model_b, device, max_length)
        
        for i in range(NUM_LAYERS):
            chunk_data[0][i].append(sk[i])
            chunk_data[1][i].append(sv[i])
            chunk_data[2][i].append(tk[i])
            chunk_data[3][i].append(tv[i])
    
    # Concatenate and save this chunk
    chunk_tensors = [[torch.cat(d) for d in layer_data] for layer_data in chunk_data]
    
    chunk_path = os.path.join(CACHE_DIR, f"chunk_{chunk_id}.pt")
    torch.save(chunk_tensors, chunk_path)
    
    # Free memory
    del chunk_data, chunk_tensors
    gc.collect()
    
    return chunk_path

def load_chunk(chunk_path):
    """Load a chunk from disk."""
    return torch.load(chunk_path, map_location='cpu')

class SimpleDeepTranslator(nn.Module):
    def __init__(self, input_size, output_size, target_heads, target_head_dim):
        super().__init__()
        self.target_heads, self.target_head_dim = target_heads, target_head_dim
        hidden_size = max(input_size, output_size)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, output_size, bias=False),
        )

    def forward(self, cache_tensor_a):
        batch, _, seq_len, _ = cache_tensor_a.shape
        x = cache_tensor_a.permute(0, 2, 1, 3).contiguous().view(batch * seq_len, -1)
        y = self.net(x)
        y = y.view(batch, seq_len, self.target_heads, self.target_head_dim)
        return y.permute(0, 2, 1, 3).contiguous()

def load_models_for_eval(model_name_a, model_name_b):
    """Load models fresh for evaluation."""
    print("Loading models for evaluation...")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16
    )
    model_a = AutoModelForCausalLM.from_pretrained(
        model_name_a, 
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    model_b = AutoModelForCausalLM.from_pretrained(
        model_name_b, 
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    model_a.eval()
    model_b.eval()
    return model_a, model_b

def evaluate_translators(model_a, model_b, tokenizer_a, tokenizer_b, translators_k, translators_v, device, eval_dataset):
    """Evaluate translators on held-out MuSiQue examples."""
    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80)
    
    translators_k.eval()
    translators_v.eval()
    model_a.eval()
    model_b.eval()
    
    total_f1 = 0
    num_evaluated = 0
    
    for i in range(min(NUM_EVAL_SAMPLES, len(eval_dataset))):
        try:
            example = eval_dataset[i]
            context_prompt, question, ground_truth_answer = format_musique_prompt(example)
            
            if not ground_truth_answer:
                continue
            
            question_prompt = " Answer:"
            
            # Generate KV cache on correct device
            inputs_a = tokenizer_a([context_prompt], return_tensors="pt", padding="max_length", 
                                   truncation=True, max_length=MAX_CTX_TOKENS).to(device)
            with torch.no_grad():
                with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
                    out_a = model_a(**inputs_a, use_cache=True)
                kv_a = out_a.past_key_values
            
            sk = [kv_a[j][0] for j in range(NUM_LAYERS)]
            sv = [kv_a[j][1] for j in range(NUM_LAYERS)]
            del out_a, kv_a, inputs_a
            
            # Translate on GPU
            with torch.no_grad():
                with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
                    translated_k = [translators_k[j](sk[j]) for j in range(NUM_LAYERS)]
                    translated_v = [translators_v[j](sv[j]) for j in range(NUM_LAYERS)]
            
            del sk, sv
            translated_cache = tuple(zip(translated_k, translated_v))
            
            # Prepare question input
            q_inputs = tokenizer_b(question_prompt, return_tensors="pt").to(device)
            context_len = translated_cache[0][0].shape[2]
            question_len = q_inputs.input_ids.shape[1]
            
            # Create attention mask and cache_position on the correct device with correct dtype
            attention_mask = torch.ones((1, context_len + question_len), device=device, dtype=torch.long)
            cache_position = torch.arange(context_len, context_len + question_len, device=device, dtype=torch.long)
            
            with torch.no_grad():
                generated = model_b.generate(
                    input_ids=q_inputs.input_ids,
                    attention_mask=attention_mask,
                    past_key_values=translated_cache,
                    cache_position=cache_position,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer_b.eos_token_id
                )
            
            response = tokenizer_b.decode(generated[0][question_len:], skip_special_tokens=True).strip()
            cleaned_response = response.split('\n')[0].strip()
            for ending in ['.', '?', '!']:
                if ending in cleaned_response:
                    cleaned_response = cleaned_response.split(ending)[0].strip()
                    break
            
            f1 = calculate_f1_score(cleaned_response, ground_truth_answer)
            total_f1 += f1
            num_evaluated += 1
            
            if i < 3:
                print(f"\nEval {num_evaluated}/{NUM_EVAL_SAMPLES}")
                print(f"Q: {question[:100]}...")
                print(f"GT: {ground_truth_answer}")
                print(f"Pred: {cleaned_response}")
                print(f"F1: {f1:.4f}")
            
            # Clean up after each example
            del translated_k, translated_v, translated_cache, q_inputs, attention_mask, cache_position, generated
            torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Eval example {i+1} failed: {e}")
            continue
    
    avg_f1 = total_f1 / num_evaluated if num_evaluated > 0 else 0
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE: Average F1 = {avg_f1:.4f} ({num_evaluated} samples)")
    print(f"{'='*80}\n")
    
    translators_k.train()
    translators_v.train()
    
    return avg_f1

def main():
    print(f"Device: {DEVICE}")
    print(f"Training with MuSiQue dataset (Ultra Memory-optimized)")
    print(f"Configuration:")
    print(f"  - Chunk size: {CHUNK_SIZE} examples")
    print(f"  - Total examples: {NUM_PROMPTS}")
    print(f"  - Chunks: {NUM_PROMPTS // CHUNK_SIZE}")
    print(f"  - 8-bit quantization: {USE_8BIT}")
    print(f"  - Training batch size: {TRAINING_BATCH_SIZE}")
    print(f"  - Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    
    print("\n--- Loading Models and Tokenizers ---")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    
    # Use 8-bit quantization to save memory
    if USE_8BIT:
        print("Loading models with 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        model_a = AutoModelForCausalLM.from_pretrained(
            MODEL_A, 
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        model_b = AutoModelForCausalLM.from_pretrained(
            MODEL_B, 
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
        model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)

    if tokenizer_a.pad_token is None: tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token
    model_a.eval()
    model_b.eval()
    print("✅ Models loaded")

    print("\n--- Loading MuSiQue Dataset ---")
    try:
        train_dataset = load_dataset("dgslibisey/MuSiQue", split="train").select(range(NUM_PROMPTS))
        eval_dataset = load_dataset("dgslibisey/MuSiQue", split="validation").select(range(NUM_EVAL_SAMPLES))
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        train_dataset = load_dataset("dgslibisey/MuSiQue", split="train").select(range(min(NUM_PROMPTS, 100)))
        eval_dataset = load_dataset("dgslibisey/MuSiQue", split="validation").select(range(min(NUM_EVAL_SAMPLES, 20)))

    print(f"Loaded {len(train_dataset)} training examples and {len(eval_dataset)} eval examples.")

    # Check if chunks already exist
    num_chunks = NUM_PROMPTS // CHUNK_SIZE
    chunk_paths = [os.path.join(CACHE_DIR, f"chunk_{i}.pt") for i in range(num_chunks)]
    
    if all(os.path.exists(p) for p in chunk_paths):
        print(f"\n✅ Found existing {num_chunks} chunks in {CACHE_DIR}/")
        print("Skipping data generation...")
    else:
        # Generate data in chunks
        print(f"\n--- Generating Training Data in Chunks ---")
        batches_per_chunk = CHUNK_SIZE // BATCH_SIZE
        chunk_paths = []
        
        for chunk_id in tqdm(range(num_chunks), desc="Generating chunks"):
            chunk_path = generate_and_save_chunk(
                chunk_id, train_dataset, tokenizer_a, tokenizer_b, 
                model_a, model_b, DEVICE, MAX_CTX_TOKENS, batches_per_chunk
            )
            chunk_paths.append(chunk_path)
        
        print(f"✅ Generated {num_chunks} chunks, saved to {CACHE_DIR}/")
    
    # Delete models to free memory (8-bit models can't be moved)
    print("\n--- Deleting models to free GPU memory ---")
    del model_a, model_b
    torch.cuda.empty_cache()
    gc.collect()
    print("✅ Models deleted, GPU memory freed")
    
    # Load first chunk to initialize translators
    print("\n--- Creating Translators ---")
    first_chunk = load_chunk(chunk_paths[0])
    
    translators_k, translators_v = nn.ModuleList(), nn.ModuleList()
    for i in range(NUM_LAYERS):
        sk, sv, tk, tv = first_chunk[0][i], first_chunk[1][i], first_chunk[2][i], first_chunk[3][i]
        k_in_size = sk.shape[1] * sk.shape[3]
        k_out_size = tk.shape[1] * tk.shape[3]
        v_in_size = sv.shape[1] * sv.shape[3]
        v_out_size = tv.shape[1] * tv.shape[3]
        translators_k.append(SimpleDeepTranslator(k_in_size, k_out_size, tk.shape[1], tk.shape[3]).to(device=DEVICE))
        translators_v.append(SimpleDeepTranslator(v_in_size, v_out_size, tv.shape[1], tv.shape[3]).to(device=DEVICE))
    
    del first_chunk
    gc.collect()
    
    total_params = sum(p.numel() for p in translators_k.parameters()) + sum(p.numel() for p in translators_v.parameters())
    print(f"Total translator parameters: {total_params:,}")

    params = list(translators_k.parameters()) + list(translators_v.parameters())
    optimizer = optim.AdamW(params, lr=LR)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=(MIXED_PRECISION and DEVICE=='cuda'))

    print("\n--- Starting Training with Layer-by-Layer Processing ---")
    print("Note: Evaluation will only run at the end to save memory")
    
    current_chunk = None
    current_chunk_idx = -1
    
    for step in trange(TRAIN_STEPS, desc="Training"):
        translators_k.train()
        translators_v.train()
        
        # Determine which chunk to use
        chunk_idx = (step // (CHUNK_SIZE // TRAINING_BATCH_SIZE)) % num_chunks
        
        # Load current chunk if needed
        if chunk_idx != current_chunk_idx:
            if current_chunk is not None:
                del current_chunk
                gc.collect()
            
            current_chunk = load_chunk(chunk_paths[chunk_idx])
            chunk_size = current_chunk[0][0].shape[0]
            chunk_indices = torch.randperm(chunk_size)
            current_chunk_idx = chunk_idx
        
        # Sample batch from current chunk
        batch_start = (step % (CHUNK_SIZE // TRAINING_BATCH_SIZE)) * TRAINING_BATCH_SIZE
        batch_end = min(batch_start + TRAINING_BATCH_SIZE, chunk_size)
        batch_indices = chunk_indices[batch_start:batch_end]

        # Process ONE LAYER AT A TIME to save memory
        total_loss_value = 0
        for i in range(NUM_LAYERS):
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
                source_k_batch = current_chunk[0][i][batch_indices].to(DEVICE)
                source_v_batch = current_chunk[1][i][batch_indices].to(DEVICE)
                target_k_batch = current_chunk[2][i][batch_indices].to(DEVICE)
                target_v_batch = current_chunk[3][i][batch_indices].to(DEVICE)
                
                pred_k = translators_k[i](source_k_batch)
                pred_v = translators_v[i](source_v_batch)
                
                loss = (loss_fn(pred_k, target_k_batch) + loss_fn(pred_v, target_v_batch)) / GRADIENT_ACCUMULATION_STEPS
                total_loss_value += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # Backward for this layer only
            scaler.scale(loss).backward()
            
            # Free memory immediately
            del source_k_batch, source_v_batch, target_k_batch, target_v_batch, pred_k, pred_v, loss
            torch.cuda.empty_cache()
        
        # Update weights after processing all layers
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        if (step + 1) % EVAL_EVERY == 0:
            avg_loss = total_loss_value / (2 * NUM_LAYERS)
            print(f"\n[Step {step+1}] Avg Layer Loss: {avg_loss:.6f}")
            
            checkpoint_path = f"kv_translators_musique_step{step+1}.pth"
            torch.save({
                'translators_k_state_dict': translators_k.state_dict(),
                'translators_v_state_dict': translators_v.state_dict(),
                'step': step + 1,
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("\nTraining complete.")
    
    # Final evaluation
    print("\n--- Running Final Evaluation ---")
    model_a, model_b = load_models_for_eval(MODEL_A, MODEL_B)
    
    avg_f1 = evaluate_translators(
        model_a, model_b, tokenizer_a, tokenizer_b,
        translators_k, translators_v, DEVICE, eval_dataset
    )
    
    print(f"\n--- Final Results ---")
    print(f"Final F1 Score: {avg_f1:.4f}")
    
    print(f"\n--- Saving Final Translators to {SAVE_PATH} ---")
    torch.save({
        'translators_k_state_dict': translators_k.state_dict(),
        'translators_v_state_dict': translators_v.state_dict(),
        'final_f1': avg_f1,
    }, SAVE_PATH)
    print(f"✅ Final translators saved successfully to {SAVE_PATH}")
    
    # Cleanup
    print(f"\nCleaning up cache chunks from {CACHE_DIR}/")
    for chunk_path in chunk_paths:
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
    print("✅ Cache chunks removed")

if __name__ == "__main__":
    main()
