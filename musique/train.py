import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset
from tqdm import trange, tqdm
import gc
import sys
import os

logging.set_verbosity_error()

###########################
# Configuration
###########################
MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_B = "Qwen/Qwen2.5-7B-Instruct"

NUM_LAYERS = 28

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "kv_translators_musique_improved.pth" # New save path for improved model
TEMP_DATA_DIR = "kv_cache_chunks_improved" # Directory to save intermediate files

# Configuration for Musique Multi-hop task
NUM_PROMPTS = 5000 # Number of training samples
MAX_CTX_TOKENS = 2048 # UPDATED: Larger context window to prevent truncation
TRAIN_STEPS = 1000
DATA_GEN_BATCH_SIZE = 8 # Batch size for sequential generation
DATA_CHUNK_SIZE = 250 # Number of prompts per chunk to save to disk
TRAINING_BATCH_SIZE = 16 # Mini-batch size for GPU training
LR = 1e-4
SEED = 42
COMPILE_MODEL = True
MIXED_PRECISION = True
###########################

torch.manual_seed(SEED)
random.seed(SEED)

if not (hasattr(torch, 'compile') and torch.cuda.is_available()):
    COMPILE_MODEL = False

def generate_kv_cache_pair_batch(prompts, tokenizer_a, tokenizer_b, model_a, model_b, device, max_length):
    """Generates aligned KV caches for a batch of fact extraction prompts."""
    # Note: Use padding="longest" to handle variable-length prompts if needed, but "max_length" is fine if you've already controlled prompt creation.
    inputs_a = tokenizer_a(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    inputs_b = tokenizer_b(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            out_a = model_a(**inputs_a, use_cache=True)
            out_b = model_b(**inputs_b, use_cache=True)
        kv_a, kv_b = out_a.past_key_values, out_b.past_key_values

    # Move to CPU immediately after generation to manage VRAM
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
        # Using a slightly larger hidden size for more capacity
        hidden_size = int((input_size + output_size) * 0.75) 
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size), # Stability
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False), # Second hidden layer
            nn.LayerNorm(hidden_size), # Stability
            nn.GELU(),
            nn.Linear(hidden_size, output_size, bias=False) # Output layer
        )

    def forward(self, cache_tensor_a):
        batch, _, seq_len, _ = cache_tensor_a.shape
        # Reshape to (batch * seq_len, head_dim * heads)
        x = cache_tensor_a.permute(0, 2, 1, 3).contiguous().view(batch * seq_len, -1)
        y = self.net(x).view(batch, seq_len, self.target_heads, self.target_head_dim)
        # Reshape back to (batch, heads, seq_len, head_dim)
        return y.permute(0, 2, 1, 3).contiguous()

def main():
    # 1. Setup Data Directory and Config
    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
    
    print("--- Loading Models and Tokenizers ---")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B)
    
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16).to(DEVICE)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16).to(DEVICE)

    if tokenizer_a.pad_token is None: tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token
    
    model_a.eval(); model_b.eval()

    # 2. Load and prepare Musique dataset
    print("--- Loading and preparing Musique Dataset ---")
    dataset = load_dataset("dgslibisey/MuSiQue", split="train").select(range(NUM_PROMPTS))
    
    # 3. Generate and Save Data in Chunks
    
    data_tensors_for_init = None 
    num_chunks = (NUM_PROMPTS + DATA_CHUNK_SIZE - 1) // DATA_CHUNK_SIZE
    
    for chunk_idx in trange(num_chunks, desc="Generating & Saving Chunks"):
        start_idx = chunk_idx * DATA_CHUNK_SIZE
        end_idx = min((chunk_idx + 1) * DATA_CHUNK_SIZE, NUM_PROMPTS)
        
        chunk_data = dataset.select(range(start_idx, end_idx))
        
        chunk_kv_data = [[[] for _ in range(NUM_LAYERS)] for _ in range(4)] # sk, sv, tk, tv
        
        for i in range(0, len(chunk_data), DATA_GEN_BATCH_SIZE):
            batch_data = chunk_data.select(range(i, min(i + DATA_GEN_BATCH_SIZE, len(chunk_data))))
            
            prompts = []
            for d in batch_data:
                supporting_paras = "\n".join([p["paragraph_text"] for p in d["paragraphs"] if p['is_supporting'] == True])
                distractor_paras = "\n".join([p["paragraph_text"] for p in d["paragraphs"] if p['is_supporting'] == False][:3])
                context_text_a = f"{supporting_paras}\n{distractor_paras}"
                
                # Use the "Fact Extraction" instruction prompt to force Model A to encode the intermediate fact
                first_hop_prompt = (
                    f"Extract the key fact required to answer the following question from the context. Be concise and state only the fact.\n\n"
                    f"Context: {context_text_a}\n\n"
                    f"Question: {d['question_decomposition'][0]['question']}\n\n"
                    f"Fact:"
                )
                prompts.append(first_hop_prompt)
            
            # Generate KV caches
            sk, sv, tk, tv = generate_kv_cache_pair_batch(prompts, tokenizer_a, tokenizer_b, model_a, model_b, DEVICE, MAX_CTX_TOKENS)
            
            # Accumulate
            for l in range(NUM_LAYERS):
                chunk_kv_data[0][l].append(sk[l]); chunk_kv_data[1][l].append(sv[l])
                chunk_kv_data[2][l].append(tk[l]); chunk_kv_data[3][l].append(tv[l])

        # Concatenate into full chunk tensors (on CPU)
        chunk_tensors = [[torch.cat(d) for d in layer_data] for layer_data in chunk_kv_data]

        if data_tensors_for_init is None:
            data_tensors_for_init = chunk_tensors

        # Save the chunk to disk
        chunk_save_path = os.path.join(TEMP_DATA_DIR, f"chunk_{chunk_idx}.pth")
        torch.save(chunk_tensors, chunk_save_path)
        
        del chunk_kv_data, chunk_tensors
        gc.collect(); torch.cuda.empty_cache()

    print("Unloading base models from VRAM...")
    del model_a, model_b, dataset, tokenizer_a, tokenizer_b
    gc.collect(); torch.cuda.empty_cache()

    # 4. Create Translators (using shapes from the first saved chunk)
    print("--- Creating Translators ---")
    translators_k, translators_v = nn.ModuleList(), nn.ModuleList()
    
    for i in range(NUM_LAYERS):
        sk, sv, tk, tv = data_tensors_for_init[0][i], data_tensors_for_init[1][i], data_tensors_for_init[2][i], data_tensors_for_init[3][i]
        k_in_size = sk.shape[1] * sk.shape[3]; k_out_size = tk.shape[1] * tk.shape[3]
        v_in_size = sv.shape[1] * sv.shape[3]; v_out_size = tv.shape[1] * tv.shape[3]
        # Uses the UPDATED FastKVTranslator with the more complex architecture
        translators_k.append(FastKVTranslator(k_in_size, k_out_size, tk.shape[1], tk.shape[3]).to(device=DEVICE))
        translators_v.append(FastKVTranslator(v_in_size, v_out_size, tv.shape[1], tv.shape[3]).to(device=DEVICE))
    
    del data_tensors_for_init
    gc.collect()

    if COMPILE_MODEL:
        print("Compiling models...")
        for i in range(NUM_LAYERS):
            translators_k[i] = torch.compile(translators_k[i]); translators_v[i] = torch.compile(translators_v[i])

    params = list(translators_k.parameters()) + list(translators_v.parameters())
    optimizer = optim.AdamW(params, lr=LR)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=(MIXED_PRECISION and DEVICE=='cuda'))

    # 5. Training Loop using Chunks
    print("--- Starting Training (Batching on GPU) ---")
    
    num_chunks = (NUM_PROMPTS + DATA_CHUNK_SIZE - 1) // DATA_CHUNK_SIZE
    
    current_chunk_on_gpu = None
    chunk_indices = None
    current_chunk_size = 0
    
    for step in trange(TRAIN_STEPS, desc="Training"):
        chunk_idx = (step % num_chunks)
        
        if step == 0 or (step % num_chunks) == 0:
            if current_chunk_on_gpu is not None:
                del current_chunk_on_gpu
                gc.collect(); torch.cuda.empty_cache()
            
            chunk_load_path = os.path.join(TEMP_DATA_DIR, f"chunk_{chunk_idx}.pth")
            current_chunk_cpu = torch.load(chunk_load_path)
            current_chunk_size = current_chunk_cpu[0][0].shape[0]
            
            current_chunk_on_gpu = [[current_chunk_cpu[c][l].to(DEVICE) for l in range(NUM_LAYERS)] for c in range(4)]
            del current_chunk_cpu
            
            chunk_indices = torch.randperm(current_chunk_size)
            
        # 2. Determine batch indices within the currently loaded chunk
        chunk_steps_per_full_pass = (current_chunk_size + TRAINING_BATCH_SIZE - 1) // TRAINING_BATCH_SIZE
        chunk_step = (step // num_chunks) % chunk_steps_per_full_pass 
        
        start_idx_in_chunk = (chunk_step * TRAINING_BATCH_SIZE)
        end_idx_in_chunk = start_idx_in_chunk + TRAINING_BATCH_SIZE
        
        if end_idx_in_chunk > current_chunk_size:
            batch_indices = torch.cat((chunk_indices[start_idx_in_chunk:], chunk_indices[:end_idx_in_chunk-current_chunk_size]))
        else:
            batch_indices = chunk_indices[start_idx_in_chunk:end_idx_in_chunk]

        total_loss = 0
        optimizer.zero_grad()
        
        # 3. GPU-only Mini-batching and Forward Pass
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
            for i in range(NUM_LAYERS):
                # Slicing the full chunk tensor (already on GPU) is fast
                source_k_batch = current_chunk_on_gpu[0][i][batch_indices]
                source_v_batch = current_chunk_on_gpu[1][i][batch_indices]
                target_k_batch = current_chunk_on_gpu[2][i][batch_indices]
                target_v_batch = current_chunk_on_gpu[3][i][batch_indices]
                
                pred_k = translators_k[i](source_k_batch)
                pred_v = translators_v[i](source_v_batch)
                
                loss = loss_fn(pred_k, target_k_batch) + loss_fn(pred_v, target_v_batch)
                total_loss += loss
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        gc.collect(); torch.cuda.empty_cache()

        if (step + 1) % 500 == 0:
            print(f"\n[Step {step+1}] Avg Layer Loss: {total_loss.item() / (2 * NUM_LAYERS):.6f}")

    print("\nTraining complete.")
    
    # 6. Save and Cleanup
    print(f"--- Saving Translators to {SAVE_PATH} ---")
    k_state_dict = translators_k.state_dict()
    v_state_dict = translators_v.state_dict()
    torch.save({'translators_k_state_dict': k_state_dict, 'translators_v_state_dict': v_state_dict}, SAVE_PATH)
    print(f"✅ Translators saved successfully to {SAVE_PATH}")
    
    try:
        import shutil
        shutil.rmtree(TEMP_DATA_DIR)
        print(f"Cleaned up temporary directory: {TEMP_DATA_DIR}")
    except OSError as e:
        print(f"Error cleaning up directory {TEMP_DATA_DIR}: {e}")

if __name__ == "__main__":
    main()
