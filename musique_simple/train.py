import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset
from tqdm import trange
import gc
import sys

# Suppress informational warnings
logging.set_verbosity_error()

###########################
# Configuration
# Usage: python train_musique.py <qwen|mistral>
###########################
if len(sys.argv) < 2 or sys.argv[1] not in ['qwen', 'mistral']:
    print("Usage: python train_musique.py <qwen|mistral>")
    sys.exit(1)

if sys.argv[1] == 'qwen':
    MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
else:
    MODEL_A = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

NUM_LAYERS = 28 if 'Qwen' in MODEL_A else 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "kv_translators_musique.pth" # Updated save path

# Configuration for MuSiQue task
NUM_PROMPTS = 1000
MAX_CTX_TOKENS = 1024 # Increased for MuSiQue's longer contexts
TRAIN_STEPS = 5000
BATCH_SIZE = 4
LR = 1e-4
SEED = 42
TRAINING_BATCH_SIZE = 32
COMPILE_MODEL = True
MIXED_PRECISION = True
###########################

torch.manual_seed(SEED)
random.seed(SEED)

if not (hasattr(torch, 'compile') and torch.cuda.is_available()):
    COMPILE_MODEL = False

def generate_kv_cache_pair_batch(prompts, tokenizer_a, tokenizer_b, model_a, model_b, device, max_length):
    inputs_a = tokenizer_a(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    inputs_b = tokenizer_b(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            out_a = model_a(**inputs_a, use_cache=True)
            out_b = model_b(**inputs_b, use_cache=True)
        kv_a, kv_b = out_a.past_key_values, out_b.past_key_values

    source_keys = [kv_a[i][0].cpu() for i in range(NUM_LAYERS)]
    source_vals = [kv_a[i][1].cpu() for i in range(NUM_LAYERS)]
    target_keys = [kv_b[i][0].cpu() for i in range(NUM_LAYERS)]
    target_vals = [kv_b[i][1].cpu() for i in range(NUM_LAYERS)]
    del out_a, out_b, kv_a, kv_b, inputs_a, inputs_b
    return source_keys, source_vals, target_keys, target_vals

class FastKVTranslator(nn.Module):
    def __init__(self, input_size, output_size, target_heads, target_head_dim):
        super().__init__()
        self.target_heads, self.target_head_dim = target_heads, target_head_dim
        hidden_size = (input_size + output_size) // 2
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size, bias=False), nn.GELU(), nn.Linear(hidden_size, output_size, bias=False))

    def forward(self, cache_tensor_a):
        batch, _, seq_len, _ = cache_tensor_a.shape
        x = cache_tensor_a.permute(0, 2, 1, 3).contiguous().view(batch * seq_len, -1)
        y = self.net(x).view(batch, seq_len, self.target_heads, self.target_head_dim)
        return y.permute(0, 2, 1, 3).contiguous()

def main():
    print(f"Device: {DEVICE}")
    print("--- Loading Models and Tokenizers ---")

    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)

    if tokenizer_a.pad_token is None: tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token
    model_a.eval(); model_b.eval()

    # --- FIX for UnboundLocalError and MuSiQue Data Loading ---
    print("--- Loading MuSiQue Dataset ---")
    
    # Get the global NUM_PROMPTS value locally
    num_prompts_to_load = globals()['NUM_PROMPTS']
    
    try:
        # Load a portion of the train split for data generation
        musique_dataset = load_dataset("dgslibisey/MuSiQue", split="train").select(range(num_prompts_to_load))
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        # Use the local variable here
        print(f"Could not load {num_prompts_to_load} examples. Loading a smaller subset.")
        # Re-set the local variable for the smaller subset
        num_prompts_to_load = min(num_prompts_to_load, 100)
        musique_dataset = load_dataset("dgslibisey/MuSiQue", split="train").select(range(num_prompts_to_load))

    print(f"Loaded {len(musique_dataset)} examples for training data generation.")

    print("--- Generating Training Data ---")
    data_by_layer = [[[] for _ in range(NUM_LAYERS)] for _ in range(4)]
    
    # Iterate based on the actual number of loaded examples
    for _ in trange(len(musique_dataset) // BATCH_SIZE, desc="Generating data"):
        
        # Create contextual prompts from MuSiQue paragraphs and questions
        prompts = []
        for _ in range(BATCH_SIZE):
            example = musique_dataset[random.randint(0, len(musique_dataset) - 1)]
            
            supporting_idx = example['question_decomposition'][0]['paragraph_support_idx']

            # 1. Filter for only supporting paragraphs
            supporting_paragraphs = [
                p['paragraph_text'] 
                for p in example['paragraphs'] 
                if p['idx'] == supporting_idx
            ]
            
            context_paragraphs = "\n\n".join(supporting_paragraphs)
            
            # 2. Format the combined context and question into a single prompt
            question_text = example['question_decomposition'][0]['question']
            
            # Use a simple contextual QA format for the KV cache prompt
            prompt = f"Context:\n{context_paragraphs}\n\nQuestion: {question_text}"
            prompts.append(prompt)

        # Generate KV cache pairs for the contextual prompts
        sk, sv, tk, tv = generate_kv_cache_pair_batch(prompts, tokenizer_a, tokenizer_b, model_a, model_b, DEVICE, MAX_CTX_TOKENS)
        for i in range(NUM_LAYERS):
            data_by_layer[0][i].append(sk[i]); data_by_layer[1][i].append(sv[i])
            data_by_layer[2][i].append(tk[i]); data_by_layer[3][i].append(tv[i])
    data_tensors = [[torch.cat(d) for d in layer_data] for layer_data in data_by_layer]

    # Recalculate NUM_PROMPTS_ACTUAL based on actual generated samples
    NUM_PROMPTS_ACTUAL = data_tensors[0][0].shape[0]

    print("Unloading base models from VRAM...")
    del model_a, model_b, musique_dataset
    gc.collect(); torch.cuda.empty_cache()

    print("--- Creating Translators ---")
    translators_k, translators_v = nn.ModuleList(), nn.ModuleList()
    for i in range(NUM_LAYERS):
        sk, sv, tk, tv = data_tensors[0][i], data_tensors[1][i], data_tensors[2][i], data_tensors[3][i]
        k_in_size = sk.shape[1] * sk.shape[3]; k_out_size = tk.shape[1] * tk.shape[3]
        v_in_size = sv.shape[1] * sv.shape[3]; v_out_size = tv.shape[1] * tv.shape[3]
        translators_k.append(FastKVTranslator(k_in_size, k_out_size, tk.shape[1], tk.shape[3]).to(device=DEVICE))
        translators_v.append(FastKVTranslator(v_in_size, v_out_size, tv.shape[1], tv.shape[3]).to(device=DEVICE))

    if COMPILE_MODEL:
        for i in range(NUM_LAYERS):
            translators_k[i] = torch.compile(translators_k[i]); translators_v[i] = torch.compile(translators_v[i])

    params = list(translators_k.parameters()) + list(translators_v.parameters())
    optimizer = optim.AdamW(params, lr=LR)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=(MIXED_PRECISION and DEVICE=='cuda'))

    print("--- Starting Training ---")
    indices = torch.randperm(NUM_PROMPTS_ACTUAL)
    for step in trange(TRAIN_STEPS, desc="Training"):
        start_idx = (step * TRAINING_BATCH_SIZE) % NUM_PROMPTS_ACTUAL
        end_idx = start_idx + TRAINING_BATCH_SIZE
        # Handle wrap-around indexing for batching
        batch_indices = indices[start_idx:end_idx] if end_idx <= NUM_PROMPTS_ACTUAL else torch.cat((indices[start_idx:], indices[:end_idx-NUM_PROMPTS_ACTUAL]))

        total_loss = 0
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
            for i in range(NUM_LAYERS):
                source_k_batch = data_tensors[0][i][batch_indices].to(DEVICE)
                source_v_batch = data_tensors[1][i][batch_indices].to(DEVICE)
                target_k_batch = data_tensors[2][i][batch_indices].to(DEVICE)
                target_v_batch = data_tensors[3][i][batch_indices].to(DEVICE)
                pred_k = translators_k[i](source_k_batch)
                pred_v = translators_v[i](source_v_batch)
                loss = loss_fn(pred_k, target_k_batch) + loss_fn(pred_v, target_v_batch)
                total_loss += loss

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if (step + 1) % 500 == 0:
            print(f"\n[Step {step+1}] Avg Layer Loss: {total_loss.item() / (2 * NUM_LAYERS):.6f}")

    print("\nTraining complete.")
    print(f"--- Saving Translators to {SAVE_PATH} ---")
    if COMPILE_MODEL and hasattr(translators_k[0], '_orig_mod'):
        original_translators_k = nn.ModuleList([m._orig_mod for m in translators_k])
        original_translators_v = nn.ModuleList([m._orig_mod for m in translators_v])
        k_state_dict = original_translators_k.state_dict()
        v_state_dict = original_translators_v.state_dict()
    else:
        k_state_dict = translators_k.state_dict()
        v_state_dict = translators_v.state_dict()
    torch.save({
        'translators_k_state_dict': k_state_dict,
        'translators_v_state_dict': v_state_dict,
    }, SAVE_PATH)
    print(f"✅ Translators saved successfully to {SAVE_PATH}")

if __name__ == "__main__":
    main()
