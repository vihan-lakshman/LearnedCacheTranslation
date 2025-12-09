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
NUM_LAYERS = 28 # Qwen 1.5B layers
SAVE_PATH = "kv_translators_gsm8k.pth"

# Training Settings
STAGE1_STEPS = 5000     # MSE Alignment
STAGE1_LR = 1e-3
STAGE1_BATCH_SIZE = 16

STAGE2_STEPS = 2000     # NTP Fine-Tuning
STAGE2_LR = 5e-5
STAGE2_GRADIENT_ACCUMULATION = 4
LAMBDA_MSE = 2.0        # High anchor to keep reasoning stable

# Context Settings
MAX_CTX_TOKENS = 256      # Max length of the Question
MAX_REASONING_TOKENS = 64 # Length of reasoning chain to train on (Teacher Forcing)

# Data
NUM_CACHE_PAIRS = 4000
DATA_BATCH_SIZE = 4
###########################

torch.manual_seed(42)
random.seed(42)

def build_prompt(question):
    return f"Question: {question}\nLet's think step by step.\nAnswer:"

# --- ResNet Translator (Same as Advanced BoolQ) ---
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
        
        self.input_proj = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.res_blocks = nn.Sequential(ResidualBlock(hidden_size), ResidualBlock(hidden_size))
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
    # Handle Qwen/Mistral structure differences
    if hasattr(kv_cache, 'key_cache'): 
        for i in range(num_layers):
            k, v = kv_cache.key_cache[i], kv_cache.value_cache[i]
            dims.append({'k_heads': k.shape[1], 'k_head_dim': k.shape[3], 'v_heads': v.shape[1], 'v_head_dim': v.shape[3]})
    else:
        for i in range(num_layers):
            dims.append({'k_heads': kv_cache[i][0].shape[1], 'k_head_dim': kv_cache[i][0].shape[3], 'v_heads': kv_cache[i][1].shape[1], 'v_head_dim': kv_cache[i][1].shape[3]})
    return dims

def generate_kv_cache_pair(prompt, tokenizer_a, tokenizer_b, model_a, model_b, device):
    inputs_a = tokenizer_a(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_CTX_TOKENS).to(device)
    inputs_b = tokenizer_b(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_CTX_TOKENS).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            out_a = model_a(**inputs_a, use_cache=True)
            out_b = model_b(**inputs_b, use_cache=True)
    
    # Extract
    kv_a = out_a.past_key_values
    kv_b = out_b.past_key_values
    
    # Unpack to list of tensors
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

def compute_ntp_loss_cot(model_b, tokenizer_b, translated_cache, answer_text, device):
    """
    Computes loss on the answer text given the translated cache of the question.
    TEACHER FORCING: We feed the answer text and ask model to predict it.
    """
    # Tokenize the reasoning chain
    answer_ids = tokenizer_b(answer_text, return_tensors="pt", truncation=True, max_length=MAX_REASONING_TOKENS).input_ids.to(device)
    
    if answer_ids.shape[1] < 2: return None

    cache_seq_len = translated_cache.get_seq_length()
    ans_len = answer_ids.shape[1]
    
    # Mask: 1s for the cache + 1s for the answer tokens
    attention_mask = torch.ones(1, cache_seq_len + ans_len, device=device)
    
    # Position IDs: Continue from where cache left off
    position_ids = torch.arange(cache_seq_len, cache_seq_len + ans_len, device=device).unsqueeze(0)
    
    # Forward pass with labels
    outputs = model_b(
        input_ids=answer_ids,
        attention_mask=attention_mask,
        past_key_values=translated_cache,
        position_ids=position_ids,
        use_cache=False
    )
    
    # Standard Causal Loss
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = answer_ids[:, 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

def main():
    print(f"Device: {DEVICE}")
    print("Loading GSM8K...")
    dataset = load_dataset("gsm8k", "main")
    train_data = [x for x in dataset['train']]
    random.shuffle(train_data)
    
    print("Loading Models...")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    
    if tokenizer_a.pad_token is None: tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token
    
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16).to(DEVICE)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16).to(DEVICE)
    
    model_a.eval()
    model_b.eval()
    for p in model_a.parameters(): p.requires_grad = False
    for p in model_b.parameters(): p.requires_grad = False

    # Init Translators
    dims_a = get_model_dimensions(tokenizer_a, model_a, DEVICE, NUM_LAYERS)
    dims_b = get_model_dimensions(tokenizer_b, model_b, DEVICE, NUM_LAYERS)
    
    translators_k = nn.ModuleList()
    translators_v = nn.ModuleList()
    
    print("Creating ResNet Translators...")
    for i in range(NUM_LAYERS):
        src, tgt = dims_a[i], dims_b[i]
        k_in, k_out = src['k_heads']*src['k_head_dim'], tgt['k_heads']*tgt['k_head_dim']
        v_in, v_out = src['v_heads']*src['v_head_dim'], tgt['v_heads']*tgt['v_head_dim']
        translators_k.append(AdvancedTranslator(k_in, k_out, tgt['k_heads'], tgt['k_head_dim']).to(DEVICE))
        translators_v.append(AdvancedTranslator(v_in, v_out, tgt['v_heads'], tgt['v_head_dim']).to(DEVICE))
        
    # STAGE 1: MSE PRETRAINING
    # We generate cache only for the QUESTION part
    print(f"\nGenerating {NUM_CACHE_PAIRS} Question Caches...")
    data_by_layer = [[[] for _ in range(NUM_LAYERS)] for _ in range(4)]
    
    data_idx = 0
    for _ in trange(NUM_CACHE_PAIRS // DATA_BATCH_SIZE):
        for _ in range(DATA_BATCH_SIZE):
            ex = train_data[data_idx % len(train_data)]; data_idx += 1
            prompt = build_prompt(ex['question']) # Only the question!
            sk, sv, tk, tv = generate_kv_cache_pair(prompt, tokenizer_a, tokenizer_b, model_a, model_b, DEVICE)
            for i in range(NUM_LAYERS):
                data_by_layer[0][i].append(sk[i])
                data_by_layer[1][i].append(sv[i])
                data_by_layer[2][i].append(tk[i])
                data_by_layer[3][i].append(tv[i])

    data_tensors = [[torch.cat(d) for d in l] for l in data_by_layer]
    num_samples = data_tensors[0][0].shape[0]
    
    print("\n--- STAGE 1: MSE ---")
    params = list(translators_k.parameters()) + list(translators_v.parameters())
    optimizer = optim.AdamW(params, lr=STAGE1_LR)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=True)
    
    # Reuse random indices logic from before...
    indices = torch.randperm(num_samples)
    for step in trange(STAGE1_STEPS):
        start = (step * STAGE1_BATCH_SIZE) % num_samples
        end = start + STAGE1_BATCH_SIZE
        batch_idx = indices[start:end] if end <= num_samples else torch.cat((indices[start:], indices[:end-num_samples]))
        
        loss = 0
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16):
            for i in range(NUM_LAYERS):
                sk_b = data_tensors[0][i][batch_idx].to(DEVICE)
                sv_b = data_tensors[1][i][batch_idx].to(DEVICE)
                tk_b = data_tensors[2][i][batch_idx].to(DEVICE)
                tv_b = data_tensors[3][i][batch_idx].to(DEVICE)
                loss += loss_fn(translators_k[i](sk_b), tk_b) + loss_fn(translators_v[i](sv_b), tv_b)
        
        optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        
    del data_tensors, data_by_layer
    torch.cuda.empty_cache()
    
    # STAGE 2: NTP (CoT) + ANCHOR
    print("\n--- STAGE 2: CoT NTP + MSE Anchor ---")
    optimizer = optim.AdamW(params, lr=STAGE2_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE2_STEPS)
    
    progress = trange(STAGE2_STEPS)
    data_idx = 0
    
    for step in progress:
        optimizer.zero_grad()
        batch_ntp = 0; batch_mse = 0; batch_ok = 0
        
        for _ in range(STAGE2_GRADIENT_ACCUMULATION):
            ex = train_data[data_idx % len(train_data)]; data_idx += 1
            prompt = build_prompt(ex['question'])
            # The answer text to predict
            reasoning = ex['answer'] 
            
            try:
                with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16):
                    # 1. Generate Ground Truth Caches (for Anchor)
                    sk, sv, tk, tv = generate_kv_cache_pair(prompt, tokenizer_a, tokenizer_b, model_a, model_b, DEVICE)
                    
                    # 2. Translate & Calc MSE
                    trans_cache = DynamicCache()
                    mse_loss = 0
                    for i in range(NUM_LAYERS):
                        pk = translators_k[i](sk[i].to(DEVICE))
                        pv = translators_v[i](sv[i].to(DEVICE))
                        mse_loss += loss_fn(pk, tk[i].to(DEVICE)) + loss_fn(pv, tv[i].to(DEVICE))
                        trans_cache.update(pk, pv, i)
                    mse_loss /= (2*NUM_LAYERS)
                    
                    # 3. Calc NTP (Teacher Forcing on Reasoning)
                    ntp_loss = compute_ntp_loss_cot(model_b, tokenizer_b, trans_cache, reasoning, DEVICE)
                    
                    if ntp_loss is not None:
                        loss = ntp_loss + (LAMBDA_MSE * mse_loss)
                        scaler.scale(loss / STAGE2_GRADIENT_ACCUMULATION).backward()
                        batch_ntp += ntp_loss.item(); batch_mse += mse_loss.item(); batch_ok += 1
            except Exception: continue
            
        if batch_ok > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            progress.set_description(f"NTP={batch_ntp/batch_ok:.3f} MSE={batch_mse/batch_ok:.3f}")

    torch.save(translators_k.state_dict(), SAVE_PATH + "_k.pth")
    torch.save(translators_v.state_dict(), SAVE_PATH + "_v.pth")
    print("Done.")

if __name__ == "__main__":
    main()
