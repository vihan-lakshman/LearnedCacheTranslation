import random
import string
import math
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from tqdm import trange
import gc
import time
import traceback
import sys
from datasets import load_dataset

# Suppress informational warnings
logging.set_verbosity_error()

###########################
# Configuration
###########################
# Using command line arguments
if sys.argv[1] == 'qwen':
    MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"  # Source Model
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"    # Target Model
elif sys.argv[1] == 'mistral':
    MODEL_A = "mistralai/Mistral-7B-Instruct-v0.2"  # Source Model
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"     # Target Model
else:
    MODEL_A = "google/gemma-2-9b"
    MODEL_B = "google/gemma-2-9b-it"

if "Qwen" in MODEL_A:
    NUM_LAYERS = 28
elif "Mistral" in MODEL_A:
    NUM_LAYERS = 32
else:
    NUM_LAYERS = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "kv_translators_multihop.pth"
NUM_PROMPTS = 5000  # Number of multi-hop examples to use for training data
MAX_CTX_TOKENS = 512 # We need a larger context window for multi-hop
TRAIN_STEPS = 5000
DATA_GEN_BATCH_SIZE = 4
TRAINING_BATCH_SIZE = 32
LR = 1e-4
SEED = 42

# GPU optimization settings
COMPILE_MODEL = True
MIXED_PRECISION = True

torch.manual_seed(SEED)
random.seed(SEED)

if not (hasattr(torch, 'compile') and torch.cuda.is_available()):
    COMPILE_MODEL = False

def print_gpu_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def generate_kv_cache_pair_batch(prompts, tokenizer_a, tokenizer_b, model_a, model_b, device, max_length):
    """
    Generate KV caches for a batch of prompts from both models.
    """
    inputs_a = tokenizer_a(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    inputs_b = tokenizer_b(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
            out_a = model_a(**inputs_a, use_cache=True)
            out_b = model_b(**inputs_b, use_cache=True)
        kv_a, kv_b = out_a.past_key_values, out_b.past_key_values

    source_keys, source_vals = [], []
    target_keys, target_vals = [], []
    for i in range(NUM_LAYERS):
        source_keys.append(kv_a[i][0].cpu()); source_vals.append(kv_a[i][1].cpu())
        target_keys.append(kv_b[i][0].cpu()); target_vals.append(kv_b[i][1].cpu())
    
    del out_a, out_b, kv_a, kv_b, inputs_a, inputs_b
    return source_keys, source_vals, target_keys, target_vals

class FastKVTranslator(nn.Module):
    def __init__(self, input_size, output_size, target_heads, target_head_dim):
        super().__init__()
        self.target_heads, self.target_head_dim = target_heads, target_head_dim
        
        hidden_size = (input_size + output_size) // 2
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, output_size, bias=False),
        )

    def forward(self, cache_tensor_a):
        batch, _, seq_len, _ = cache_tensor_a.shape
        x = cache_tensor_a.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch * seq_len, -1)
        y = self.net(x)
        y = y.view(batch, seq_len, self.target_heads, self.target_head_dim)
        return y.permute(0, 2, 1, 3).contiguous()

def main():
    print(f"Device: {DEVICE}")
    print("--- Loading Models (This may take a while) ---")
    
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A)
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16).to(DEVICE)

    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16).to(DEVICE)

    if tokenizer_a.pad_token is None: tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token

    model_a.eval(); model_b.eval()
    print_gpu_usage()

    # Data Generation for Multi-hop Reasoning with Musique dataset
    print("--- Loading and Generating Training Data from Musique Dataset ---")
    
    # Load the Musique dataset
    dataset = load_dataset("dgslibisey/MuSiQue", split="train")
    
    # The dataset is an iterable, we can select a number of samples to use.
    train_data = [example for i, example in enumerate(dataset) if i < NUM_PROMPTS]

    data_by_layer = [[[] for _ in range(NUM_LAYERS)] for _ in range(4)]
    
    for i in trange(0, NUM_PROMPTS, DATA_GEN_BATCH_SIZE, desc="Generating data"):
        batch_data = train_data[i:i + DATA_GEN_BATCH_SIZE]
        first_hop_prompts = []
        for d in batch_data:
            # Join the context paragraphs and append the first decomposed question
            context_text = " ".join([p["paragraph_text"] for p in d["paragraphs"]])
            # CORRECTION: Use 'question_decomposition' instead of 'decomposed_question'
            first_hop_prompt = f"{context_text} {d['question_decomposition'][0]}"
            first_hop_prompts.append(first_hop_prompt)

        # Generate KV caches for the first part of the multi-hop query
        sk, sv, tk, tv = generate_kv_cache_pair_batch(first_hop_prompts, tokenizer_a, tokenizer_b, model_a, model_b, DEVICE, MAX_CTX_TOKENS)
        
        for j in range(NUM_LAYERS):
            data_by_layer[0][j].append(sk[j]); data_by_layer[1][j].append(sv[j])
            data_by_layer[2][j].append(tk[j]); data_by_layer[3][j].append(tv[j])
        
        gc.collect()
        torch.cuda.empty_cache()

    data_tensors = [[torch.cat(d) for d in layer_data] for layer_data in data_by_layer]
    
    del model_a, model_b
    gc.collect()
    torch.cuda.empty_cache()
    print("Models unloaded from VRAM.")
    print_gpu_usage()

    print("--- Creating Translators ---")
    translators_k, translators_v = nn.ModuleList(), nn.ModuleList()

    for i in range(NUM_LAYERS):
        sk, sv = data_tensors[0][i], data_tensors[1][i]
        tk, tv = data_tensors[2][i], data_tensors[3][i]

        k_in_size = sk.shape[1] * sk.shape[3]
        k_out_size = tk.shape[1] * tk.shape[3]
        v_in_size = sv.shape[1] * sv.shape[3]
        v_out_size = tv.shape[1] * tv.shape[3]
        
        translators_k.append(FastKVTranslator(k_in_size, k_out_size, tk.shape[1], tk.shape[3]).to(device=DEVICE))
        translators_v.append(FastKVTranslator(v_in_size, v_out_size, tv.shape[1], tv.shape[3]).to(device=DEVICE))

    if COMPILE_MODEL:
        for i in range(NUM_LAYERS):
            translators_k[i] = torch.compile(translators_k[i]); translators_v[i] = torch.compile(translators_v[i])

    params = list(translators_k.parameters()) + list(translators_v.parameters())
    optimizer = optim.AdamW(params, lr=LR)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=(MIXED_PRECISION and DEVICE=='cuda'))

    # Training loop using the new multi-hop data
    print("--- Starting Training ---")
    indices = torch.randperm(NUM_PROMPTS)

    for step in trange(TRAIN_STEPS, desc="Training"):
        start_idx = (step * TRAINING_BATCH_SIZE) % NUM_PROMPTS
        end_idx = start_idx + TRAINING_BATCH_SIZE
        if end_idx > NUM_PROMPTS:
            batch_indices = torch.cat((indices[start_idx:], indices[:end_idx-NUM_PROMPTS]))
        else:
            batch_indices = indices[start_idx:end_idx]

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

    print("\n--- Saving Translators ---")
    k_state_dict = translators_k._orig_mod.state_dict() if COMPILE_MODEL and hasattr(translators_k, '_orig_mod') else translators_k.state_dict()
    v_state_dict = translators_v._orig_mod.state_dict() if COMPILE_MODEL and hasattr(translators_v, '_orig_mod') else translators_v.state_dict()
 
    torch.save({
        'translators_k_state_dict': translators_k.state_dict(),
        'translators_v_state_dict': translators_v.state_dict(),
    }, SAVE_PATH)
    
    print(f"✅ Translators saved successfully to {SAVE_PATH}")


if __name__ == "__main__":
    main()
