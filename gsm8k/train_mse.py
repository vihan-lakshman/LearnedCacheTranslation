import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from tqdm import trange
import sys
import re

logging.set_verbosity_error()

###########################
# Configuration
###########################
MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_B = "Qwen/Qwen2.5-7B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_LAYERS = 28  # Qwen 1.5B layers
SAVE_PATH = "kv_translators_gsm8k_mse_only"

# Training Settings - Single Stage MSE Only
TRAIN_STEPS = 7000      # Combined steps (roughly equivalent to 5000 + 2000)
LEARNING_RATE = 1e-3
BATCH_SIZE = 16

# Context Settings
MAX_CTX_TOKENS = 256    # Max length of the Question

# Data
NUM_CACHE_PAIRS = 4000
DATA_BATCH_SIZE = 4
###########################

torch.manual_seed(42)
random.seed(42)


def build_prompt(question):
    return f"Question: {question}\nLet's think step by step.\nAnswer:"


# --- ResNet Translator ---
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 2)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x):
        return x + self.linear2(self.act(self.linear1(self.norm(x))))


class AdvancedTranslator(nn.Module):
    def __init__(self, input_size, output_size, target_heads, target_head_dim):
        super().__init__()
        self.target_heads = target_heads
        self.target_head_dim = target_head_dim
        hidden_size = 512
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size)
        )
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, cache_tensor):
        batch, num_heads, seq_len, head_dim = cache_tensor.shape
        x = cache_tensor.permute(0, 2, 1, 3).contiguous().view(batch * seq_len, -1)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        y = self.output_proj(x)
        y = y.view(batch, seq_len, self.target_heads, self.target_head_dim)
        return y.permute(0, 2, 1, 3).contiguous()


# --- Helper Functions ---
def get_model_dimensions(tokenizer, model, device, num_layers):
    dummy_input = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**dummy_input, use_cache=True)
    kv_cache = outputs.past_key_values
    dims = []
    if hasattr(kv_cache, 'key_cache'): 
        for i in range(num_layers):
            k, v = kv_cache.key_cache[i], kv_cache.value_cache[i]
            dims.append({
                'k_heads': k.shape[1], 'k_head_dim': k.shape[3],
                'v_heads': v.shape[1], 'v_head_dim': v.shape[3]
            })
    else:
        for i in range(num_layers):
            dims.append({
                'k_heads': kv_cache[i][0].shape[1], 'k_head_dim': kv_cache[i][0].shape[3],
                'v_heads': kv_cache[i][1].shape[1], 'v_head_dim': kv_cache[i][1].shape[3]
            })
    return dims


def generate_kv_cache_pair(prompt, tokenizer_a, tokenizer_b, model_a, model_b, device):
    inputs_a = tokenizer_a(
        prompt, return_tensors="pt", padding="max_length",
        truncation=True, max_length=MAX_CTX_TOKENS
    ).to(device)
    inputs_b = tokenizer_b(
        prompt, return_tensors="pt", padding="max_length",
        truncation=True, max_length=MAX_CTX_TOKENS
    ).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            out_a = model_a(**inputs_a, use_cache=True)
            out_b = model_b(**inputs_b, use_cache=True)
    
    kv_a = out_a.past_key_values
    kv_b = out_b.past_key_values
    
    if hasattr(kv_a, 'key_cache'):
        sk = [kv_a.key_cache[i].cpu() for i in range(NUM_LAYERS)]
        sv = [kv_a.value_cache[i].cpu() for i in range(NUM_LAYERS)]
    else:
        sk = [kv_a[i][0].cpu() for i in range(NUM_LAYERS)]
        sv = [kv_a[i][1].cpu() for i in range(NUM_LAYERS)]

    if hasattr(kv_b, 'key_cache'):
        tk = [kv_b.key_cache[i].cpu() for i in range(NUM_LAYERS)]
        tv = [kv_b.value_cache[i].cpu() for i in range(NUM_LAYERS)]
    else:
        tk = [kv_b[i][0].cpu() for i in range(NUM_LAYERS)]
        tv = [kv_b[i][1].cpu() for i in range(NUM_LAYERS)]
        
    return sk, sv, tk, tv


def main():
    print(f"Device: {DEVICE}")
    print("=" * 50)
    print("GSM8K KV Translator - MSE ONLY (No NTP)")
    print("=" * 50)
    
    print("\nLoading GSM8K...")
    dataset = load_dataset("gsm8k", "main")
    train_data = [x for x in dataset['train']]
    random.shuffle(train_data)
    print(f"Training samples: {len(train_data)}")
    
    print("\nLoading Models...")
    print(f"  Model A (source): {MODEL_A}")
    print(f"  Model B (target): {MODEL_B}")
    
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    
    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None:
        tokenizer_b.pad_token = tokenizer_b.eos_token
    
    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL_A, torch_dtype=torch.float16
    ).to(DEVICE)
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL_B, torch_dtype=torch.float16
    ).to(DEVICE)
    
    model_a.eval()
    model_b.eval()
    for p in model_a.parameters():
        p.requires_grad = False
    for p in model_b.parameters():
        p.requires_grad = False

    # Initialize translators
    dims_a = get_model_dimensions(tokenizer_a, model_a, DEVICE, NUM_LAYERS)
    dims_b = get_model_dimensions(tokenizer_b, model_b, DEVICE, NUM_LAYERS)
    
    translators_k = nn.ModuleList()
    translators_v = nn.ModuleList()
    
    print("\nCreating ResNet Translators...")
    for i in range(NUM_LAYERS):
        src, tgt = dims_a[i], dims_b[i]
        k_in = src['k_heads'] * src['k_head_dim']
        k_out = tgt['k_heads'] * tgt['k_head_dim']
        v_in = src['v_heads'] * src['v_head_dim']
        v_out = tgt['v_heads'] * tgt['v_head_dim']
        translators_k.append(
            AdvancedTranslator(k_in, k_out, tgt['k_heads'], tgt['k_head_dim']).to(DEVICE)
        )
        translators_v.append(
            AdvancedTranslator(v_in, v_out, tgt['v_heads'], tgt['v_head_dim']).to(DEVICE)
        )
    
    total_params = sum(p.numel() for p in translators_k.parameters()) + \
                   sum(p.numel() for p in translators_v.parameters())
    print(f"Total translator parameters: {total_params:,}")
    
    # =========================================
    # Generate Cache Pairs
    # =========================================
    print(f"\n{'=' * 50}")
    print(f"Generating {NUM_CACHE_PAIRS} cache pairs...")
    print(f"{'=' * 50}")
    
    data_by_layer = [[[] for _ in range(NUM_LAYERS)] for _ in range(4)]
    
    data_idx = 0
    for _ in trange(NUM_CACHE_PAIRS // DATA_BATCH_SIZE, desc="Generating caches"):
        for _ in range(DATA_BATCH_SIZE):
            ex = train_data[data_idx % len(train_data)]
            data_idx += 1
            prompt = build_prompt(ex['question'])
            sk, sv, tk, tv = generate_kv_cache_pair(
                prompt, tokenizer_a, tokenizer_b, model_a, model_b, DEVICE
            )
            for i in range(NUM_LAYERS):
                data_by_layer[0][i].append(sk[i])
                data_by_layer[1][i].append(sv[i])
                data_by_layer[2][i].append(tk[i])
                data_by_layer[3][i].append(tv[i])

    data_tensors = [[torch.cat(d) for d in layer] for layer in data_by_layer]
    num_samples = data_tensors[0][0].shape[0]
    print(f"Total cache samples: {num_samples}")
    
    # =========================================
    # MSE-Only Training
    # =========================================
    print(f"\n{'=' * 50}")
    print(f"MSE-Only Training ({TRAIN_STEPS} steps)")
    print(f"{'=' * 50}")
    
    params = list(translators_k.parameters()) + list(translators_v.parameters())
    optimizer = optim.AdamW(params, lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=True)
    
    indices = torch.randperm(num_samples)
    
    progress = trange(TRAIN_STEPS)
    for step in progress:
        start = (step * BATCH_SIZE) % num_samples
        end = start + BATCH_SIZE
        if end <= num_samples:
            batch_idx = indices[start:end]
        else:
            batch_idx = torch.cat((indices[start:], indices[:end - num_samples]))
        
        loss = 0
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16):
            for i in range(NUM_LAYERS):
                sk_b = data_tensors[0][i][batch_idx].to(DEVICE)
                sv_b = data_tensors[1][i][batch_idx].to(DEVICE)
                tk_b = data_tensors[2][i][batch_idx].to(DEVICE)
                tv_b = data_tensors[3][i][batch_idx].to(DEVICE)
                
                pred_k = translators_k[i](sk_b)
                pred_v = translators_v[i](sv_b)
                
                loss += loss_fn(pred_k, tk_b) + loss_fn(pred_v, tv_b)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if step % 100 == 0:
            progress.set_description(f"MSE Loss: {loss.item():.4f}")
    
    # Save
    print("\nSaving translators...")
    torch.save(translators_k.state_dict(), SAVE_PATH + "_k.pth")
    torch.save(translators_v.state_dict(), SAVE_PATH + "_v.pth")
    print(f"Saved to {SAVE_PATH}_k.pth and {SAVE_PATH}_v.pth")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
