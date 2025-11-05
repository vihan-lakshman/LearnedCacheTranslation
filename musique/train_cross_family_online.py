import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset
from tqdm import trange
import sys
import re
import string
from collections import Counter
import argparse

logging.set_verbosity_error()

###########################
# Configuration
###########################
parser = argparse.ArgumentParser(description='Train cross-family KV translators with on-the-fly data generation')
parser.add_argument('--strategy', type=str, default='interpolate', 
                    choices=['interpolate', 'skip_layers', 'repeat_last'],
                    help='Layer mapping strategy')
parser.add_argument('--train_steps', type=int, default=10000,
                    help='Number of training steps')
parser.add_argument('--num_prompts', type=int, default=1000,
                    help='Size of dataset to sample from')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')
args = parser.parse_args()

print(f"Cross-Family KV Cache Translation: Qwen 1.5B -> Mistral 7B")
print(f"Strategy: {args.strategy}")
print(f"Training steps: {args.train_steps}")
print(f"Dataset size: {args.num_prompts}")
print(f"On-the-fly data generation: ENABLED")

MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

NUM_LAYERS_A = 28  # Qwen layers
NUM_LAYERS_B = 32  # Mistral layers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = f"kv_translators_cross_family_{args.strategy}_online.pth"

# Configuration
NUM_PROMPTS = args.num_prompts
MAX_CTX_TOKENS = 1024
TRAIN_STEPS = args.train_steps
BATCH_SIZE = 2  # Batch size for KV cache generation
LR = args.lr
SEED = 42
TRAINING_BATCH_SIZE = 4  # Batch size for training
COMPILE_MODEL = True
MIXED_PRECISION = True

# Evaluation settings
EVAL_EVERY = 1000
NUM_EVAL_SAMPLES = 100
MAX_NEW_TOKENS = 64

LAYER_MAPPING_STRATEGY = args.strategy
###########################

torch.manual_seed(SEED)
random.seed(SEED)

if not (hasattr(torch, 'compile') and torch.cuda.is_available()):
    COMPILE_MODEL = False

def get_layer_mapping(num_layers_a, num_layers_b, strategy="interpolate"):
    """Create a mapping from Model A layers to Model B layers."""
    if strategy == "interpolate":
        mapping = {}
        for i in range(num_layers_b):
            a_layer = int(i * num_layers_a / num_layers_b)
            mapping[i] = a_layer
        return mapping
    
    elif strategy == "skip_layers":
        mapping = {}
        for i in range(min(num_layers_a, num_layers_b)):
            mapping[i] = i
        return mapping
    
    elif strategy == "repeat_last":
        mapping = {}
        for i in range(num_layers_b):
            if i < num_layers_a:
                mapping[i] = i
            else:
                mapping[i] = num_layers_a - 1
        return mapping
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

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

def generate_kv_cache_pair_batch_online(prompts, tokenizer_a, tokenizer_b, model_a, model_b, device, max_length):
    """Generate KV cache pairs on-the-fly - returns tensors on GPU."""
    inputs_a = tokenizer_a(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    inputs_b = tokenizer_b(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            out_a = model_a(**inputs_a, use_cache=True)
            out_b = model_b(**inputs_b, use_cache=True)
        kv_a, kv_b = out_a.past_key_values, out_b.past_key_values

    # Keep on GPU, don't move to CPU
    source_keys = [kv_a[i][0] for i in range(NUM_LAYERS_A)]
    source_vals = [kv_a[i][1] for i in range(NUM_LAYERS_A)]
    target_keys = [kv_b[i][0] for i in range(NUM_LAYERS_B)]
    target_vals = [kv_b[i][1] for i in range(NUM_LAYERS_B)]
    
    del out_a, out_b, kv_a, kv_b, inputs_a, inputs_b
    
    return source_keys, source_vals, target_keys, target_vals

def generate_kv_cache(prompts, tokenizer_a, model_a, device, max_length, num_layers):
    """Generate KV cache from Model A."""
    inputs_a = tokenizer_a(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            out_a = model_a(**inputs_a, use_cache=True)
        kv_a = out_a.past_key_values
    source_keys = [kv_a[i][0] for i in range(num_layers)]
    source_vals = [kv_a[i][1] for i in range(num_layers)]
    del out_a, kv_a, inputs_a
    return source_keys, source_vals

class SimpleDeepTranslator(nn.Module):
    def __init__(self, input_size, output_size, target_heads, target_head_dim):
        super().__init__()
        self.target_heads, self.target_head_dim = target_heads, target_head_dim
        
        hidden_size = max(input_size, output_size)
        
        # 4-layer deep network
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

def evaluate_translators(model_a, model_b, tokenizer_a, tokenizer_b, translators_k, translators_v, 
                         layer_mapping, device, eval_dataset):
    """Evaluate cross-family translators with proper handling of skip_layers strategy."""
    print("\n" + "="*80)
    print("RUNNING EVALUATION (Cross-Family)")
    print("="*80)
    
    for t in translators_k:
        if t is not None:
            t.eval()
    for t in translators_v:
        if t is not None:
            t.eval()
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
            
            # Generate KV cache from Model A (Qwen)
            sk, sv = generate_kv_cache([context_prompt], tokenizer_a, model_a, device, MAX_CTX_TOKENS, NUM_LAYERS_A)
            
            # For skip_layers, we need native Mistral cache for unmapped layers
            if LAYER_MAPPING_STRATEGY == "skip_layers":
                inputs_b = tokenizer_b([context_prompt], return_tensors="pt", truncation=True,
                                      max_length=MAX_CTX_TOKENS).to(device)
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
                        outputs_b = model_b(**inputs_b, use_cache=True)
                        native_cache_b = outputs_b.past_key_values
                del inputs_b, outputs_b
            
            # Translate the cache to Model B (Mistral) format
            with torch.no_grad():
                with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
                    translated_k = []
                    translated_v = []
                    
                    for b_layer in range(NUM_LAYERS_B):
                        if b_layer in layer_mapping:
                            # Translate from Qwen
                            a_layer = layer_mapping[b_layer]
                            trans_k = translators_k[b_layer](sk[a_layer])
                            trans_v = translators_v[b_layer](sv[a_layer])
                            translated_k.append(trans_k)
                            translated_v.append(trans_v)
                        else:
                            # Use native Mistral (for skip_layers, layers 28-31)
                            translated_k.append(native_cache_b[b_layer][0])
                            translated_v.append(native_cache_b[b_layer][1])
            
            translated_cache = tuple(zip(translated_k, translated_v))
            
            # Generate answer using translated cache
            q_inputs = tokenizer_b(question_prompt, return_tensors="pt").to(device)
            context_len = translated_cache[0][0].shape[2]
            question_len = q_inputs.input_ids.shape[1]
            attention_mask = torch.ones(1, context_len + question_len, device=device)
            cache_position = torch.arange(context_len, context_len + question_len, device=device)
            
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
                print(f"Q: {question}")
                print(f"GT: {ground_truth_answer}")
                print(f"Pred: {cleaned_response}")
                print(f"F1: {f1:.4f}")
        
        except Exception as e:
            print(f"Eval example {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    avg_f1 = total_f1 / num_evaluated if num_evaluated > 0 else 0
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE: Average F1 = {avg_f1:.4f} ({num_evaluated} samples)")
    print(f"{'='*80}\n")
    
    for t in translators_k:
        if t is not None:
            t.train()
    for t in translators_v:
        if t is not None:
            t.train()
    
    return avg_f1

def main():
    print(f"Device: {DEVICE}")
    print(f"Cross-Family Translation: Qwen 1.5B ({NUM_LAYERS_A} layers) -> Mistral 7B ({NUM_LAYERS_B} layers)")
    print(f"Layer mapping strategy: {LAYER_MAPPING_STRATEGY}")
    print(f"Training with MuSiQue dataset")
    print(f"Max context tokens: {MAX_CTX_TOKENS}")
    print(f"Memory optimization: On-the-fly data generation")
    
    # Create layer mapping
    layer_mapping = get_layer_mapping(NUM_LAYERS_A, NUM_LAYERS_B, LAYER_MAPPING_STRATEGY)
    
    num_translators = len(layer_mapping)
    print(f"\nNumber of translators to train: {num_translators}")
    print(f"Layer Mapping (Model B layer -> Model A layer):")
    
    if LAYER_MAPPING_STRATEGY == "skip_layers":
        print("Strategy: Skip Layers - 1-to-1 mapping for first 28, native for 29-32")
    
    for b_layer in sorted(layer_mapping.keys())[:10]:
        a_layer = layer_mapping[b_layer]
        print(f"  Mistral Layer {b_layer:2d} <- Qwen Layer {a_layer:2d}")
    
    if len(layer_mapping) > 20:
        print(f"  ...")
        for b_layer in sorted(layer_mapping.keys())[-5:]:
            a_layer = layer_mapping[b_layer]
            print(f"  Mistral Layer {b_layer:2d} <- Qwen Layer {a_layer:2d}")
    
    if LAYER_MAPPING_STRATEGY == "skip_layers":
        unmapped = [i for i in range(NUM_LAYERS_B) if i not in layer_mapping]
        print(f"\n  Unmapped layers (will use native Mistral): {unmapped}")
    
    print("\n--- Loading Models and Tokenizers ---")

    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)

    if tokenizer_a.pad_token is None: tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token
    model_a.eval()
    model_b.eval()
    print("✅ Models loaded and ready for on-the-fly generation")

    print("\n--- Loading MuSiQue Dataset ---")
    try:
        train_dataset = load_dataset("dgslibisey/MuSiQue", split="train").select(range(NUM_PROMPTS))
        eval_dataset = load_dataset("dgslibisey/MuSiQue", split="validation").select(range(NUM_EVAL_SAMPLES))
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        print(f"Loading smaller subset...")
        train_dataset = load_dataset("dgslibisey/MuSiQue", split="train").select(range(min(NUM_PROMPTS, 100)))
        eval_dataset = load_dataset("dgslibisey/MuSiQue", split="validation").select(range(min(NUM_EVAL_SAMPLES, 20)))

    print(f"Loaded {len(train_dataset)} training examples and {len(eval_dataset)} eval examples.")
    print(f"💡 No pre-generation needed - data generated on-the-fly during training!")

    print("\n--- Creating Translators (generating sample batch for dimensions) ---")
    
    # Generate ONE sample batch just to get dimensions
    sample_prompts = []
    for _ in range(BATCH_SIZE):
        example = train_dataset[random.randint(0, len(train_dataset) - 1)]
        prompt, _, _ = format_musique_prompt(example)
        sample_prompts.append(prompt)
    
    sample_sk, sample_sv, sample_tk, sample_tv = generate_kv_cache_pair_batch_online(
        sample_prompts, tokenizer_a, tokenizer_b, model_a, model_b, DEVICE, MAX_CTX_TOKENS
    )
    
    print(f"Sample dimensions obtained from batch")
    
    # Create translator architecture
    translators_k, translators_v = nn.ModuleList(), nn.ModuleList()
    
    for b_layer in range(NUM_LAYERS_B):
        if b_layer in layer_mapping:
            a_layer = layer_mapping[b_layer]
            
            # Get dimensions from sample
            sk = sample_sk[a_layer]
            sv = sample_sv[a_layer]
            tk = sample_tk[b_layer]
            tv = sample_tv[b_layer]
            
            k_in_size = sk.shape[1] * sk.shape[3]
            k_out_size = tk.shape[1] * tk.shape[3]
            v_in_size = sv.shape[1] * sv.shape[3]
            v_out_size = tv.shape[1] * tv.shape[3]
            
            if b_layer < 5 or b_layer >= NUM_LAYERS_B - 5:
                print(f"Layer {b_layer}: Qwen[{a_layer}] ({k_in_size}) -> Mistral[{b_layer}] ({k_out_size})")
            
            translators_k.append(SimpleDeepTranslator(k_in_size, k_out_size, tk.shape[1], tk.shape[3]).to(device=DEVICE))
            translators_v.append(SimpleDeepTranslator(v_in_size, v_out_size, tv.shape[1], tv.shape[3]).to(device=DEVICE))
        else:
            # For unmapped layers (skip_layers strategy)
            translators_k.append(None)
            translators_v.append(None)
    
    # Clean up sample batch
    del sample_sk, sample_sv, sample_tk, sample_tv
    torch.cuda.empty_cache()
    
    # Count trainable parameters
    trainable_k = [t for t in translators_k if t is not None]
    trainable_v = [t for t in translators_v if t is not None]
    total_params = sum(p.numel() for t in trainable_k for p in t.parameters()) + \
                   sum(p.numel() for t in trainable_v for p in t.parameters())
    print(f"\nTotal translator parameters: {total_params:,}")
    print(f"Training {len(trainable_k)} layer translators")

    if COMPILE_MODEL:
        print("\nCompiling translators for faster training...")
        for i in range(NUM_LAYERS_B):
            if translators_k[i] is not None:
                translators_k[i] = torch.compile(translators_k[i])
                translators_v[i] = torch.compile(translators_v[i])

    params = [p for t in trainable_k for p in t.parameters()] + \
             [p for t in trainable_v for p in t.parameters()]
    optimizer = optim.AdamW(params, lr=LR)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=(MIXED_PRECISION and DEVICE=='cuda'))

    print(f"\n--- Starting Training for {TRAIN_STEPS} steps ---")
    print(f"Batch size for KV generation: {BATCH_SIZE}")
    print(f"Training batch size: {TRAINING_BATCH_SIZE}")
    print(f"Data will be generated fresh for each training step")
    
    if LAYER_MAPPING_STRATEGY == "skip_layers":
        print("💡 Skip layers may need 15k-20k steps for best results")
    
    eval_history = []
    
    for step in trange(TRAIN_STEPS, desc="Training"):
        # Set to training mode
        for t in trainable_k + trainable_v:
            t.train()
        
        # Generate batch on-the-fly
        prompts = []
        for _ in range(TRAINING_BATCH_SIZE):
            example = train_dataset[random.randint(0, len(train_dataset) - 1)]
            prompt, _, _ = format_musique_prompt(example)
            prompts.append(prompt)
        
        # Generate KV caches directly on GPU (no CPU storage!)
        source_keys, source_vals, target_keys, target_vals = generate_kv_cache_pair_batch_online(
            prompts, tokenizer_a, tokenizer_b, model_a, model_b, DEVICE, MAX_CTX_TOKENS
        )
        
        # Training step
        total_loss = 0
        layers_trained = 0
        
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
            for b_layer in layer_mapping.keys():  # Only train mapped layers
                a_layer = layer_mapping[b_layer]
                
                # Data is already on GPU!
                source_k_batch = source_keys[a_layer]
                source_v_batch = source_vals[a_layer]
                target_k_batch = target_keys[b_layer]
                target_v_batch = target_vals[b_layer]
                
                pred_k = translators_k[b_layer](source_k_batch)
                pred_v = translators_v[b_layer](source_v_batch)
                
                loss = loss_fn(pred_k, target_k_batch) + loss_fn(pred_v, target_v_batch)
                total_loss += loss
                layers_trained += 1

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Clean up batch
        del source_keys, source_vals, target_keys, target_vals
        
        # Periodic evaluation
        if (step + 1) % EVAL_EVERY == 0:
            avg_loss = total_loss.item() / (2 * layers_trained)
            print(f"\n[Step {step+1}/{TRAIN_STEPS}] Avg Layer Loss: {avg_loss:.6f}")
            
            # Clear cache before eval
            torch.cuda.empty_cache()
            
            avg_f1 = evaluate_translators(
                model_a, model_b, tokenizer_a, tokenizer_b,
                translators_k, translators_v, layer_mapping, DEVICE, eval_dataset
            )
            eval_history.append((step + 1, avg_loss, avg_f1))
            
            checkpoint_path = f"kv_translators_cross_family_{LAYER_MAPPING_STRATEGY}_online_step{step+1}.pth"
            
            # Save checkpoint
            k_state_dict = {}
            v_state_dict = {}
            for i, (tk, tv) in enumerate(zip(translators_k, translators_v)):
                if tk is not None:
                    if COMPILE_MODEL and hasattr(tk, '_orig_mod'):
                        for name, param in tk._orig_mod.named_parameters():
                            k_state_dict[f'{i}.{name}'] = param.cpu()
                    else:
                        for name, param in tk.named_parameters():
                            k_state_dict[f'{i}.{name}'] = param.cpu()
                if tv is not None:
                    if COMPILE_MODEL and hasattr(tv, '_orig_mod'):
                        for name, param in tv._orig_mod.named_parameters():
                            v_state_dict[f'{i}.{name}'] = param.cpu()
                    else:
                        for name, param in tv.named_parameters():
                            v_state_dict[f'{i}.{name}'] = param.cpu()
            
            torch.save({
                'translators_k_state_dict': k_state_dict,
                'translators_v_state_dict': v_state_dict,
                'layer_mapping': layer_mapping,
                'num_layers_a': NUM_LAYERS_A,
                'num_layers_b': NUM_LAYERS_B,
                'mapping_strategy': LAYER_MAPPING_STRATEGY,
                'step': step + 1,
                'loss': avg_loss,
                'f1': avg_f1,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("\n✅ Training complete!")
    
    print("\n" + "="*80)
    print(f"EVALUATION HISTORY (Cross-Family: Qwen->Mistral, {LAYER_MAPPING_STRATEGY})")
    print("="*80)
    print(f"{'Step':<10} {'Loss':<12} {'F1 Score':<12}")
    print("-" * 80)
    for step, loss, f1 in eval_history:
        print(f"{step:<10} {loss:<12.6f} {f1:<12.4f}")
    print("="*80)
    
    # Analyze convergence
    if len(eval_history) >= 2:
        early_f1 = eval_history[0][2]
        late_f1 = eval_history[-1][2]
        improvement = late_f1 - early_f1
        print(f"\nConvergence analysis:")
        print(f"  F1 at step {eval_history[0][0]}: {early_f1:.4f}")
        print(f"  F1 at step {eval_history[-1][0]}: {late_f1:.4f}")
        print(f"  Total improvement: {improvement:+.4f}")
        
        if LAYER_MAPPING_STRATEGY == "skip_layers" and TRAIN_STEPS < 15000:
            print(f"\n💡 Tip: skip_layers might improve further with 15k-20k steps")

    print(f"\n--- Saving Final Translators to {SAVE_PATH} ---")
    
    # Final save
    k_state_dict = {}
    v_state_dict = {}
    for i, (tk, tv) in enumerate(zip(translators_k, translators_v)):
        if tk is not None:
            if COMPILE_MODEL and hasattr(tk, '_orig_mod'):
                for name, param in tk._orig_mod.named_parameters():
                    k_state_dict[f'{i}.{name}'] = param.cpu()
            else:
                for name, param in tk.named_parameters():
                    k_state_dict[f'{i}.{name}'] = param.cpu()
        if tv is not None:
            if COMPILE_MODEL and hasattr(tv, '_orig_mod'):
                for name, param in tv._orig_mod.named_parameters():
                    v_state_dict[f'{i}.{name}'] = param.cpu()
            else:
                for name, param in tv.named_parameters():
                    v_state_dict[f'{i}.{name}'] = param.cpu()
    
    torch.save({
        'translators_k_state_dict': k_state_dict,
        'translators_v_state_dict': v_state_dict,
        'layer_mapping': layer_mapping,
        'num_layers_a': NUM_LAYERS_A,
        'num_layers_b': NUM_LAYERS_B,
        'mapping_strategy': LAYER_MAPPING_STRATEGY,
        'eval_history': eval_history,
        'num_prompts': NUM_PROMPTS,
    }, SAVE_PATH)
    print(f"✅ Final translators saved successfully to {SAVE_PATH}")
    print(f"\n🎉 Training complete with on-the-fly data generation!")
    print(f"   Memory efficient: No CPU storage of training data")
    print(f"   Dataset size: {NUM_PROMPTS} examples")
    print(f"   Total steps: {TRAIN_STEPS}")

if __name__ == "__main__":
    main()
