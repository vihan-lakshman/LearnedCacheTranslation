import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from tqdm import tqdm
import re

logging.set_verbosity_error()

###########################
# Config
###########################
MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
CHECKPOINT_PATH_K = "kv_translators_gsm8k.pth_k.pth"
CHECKPOINT_PATH_V = "kv_translators_gsm8k.pth_v.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_LAYERS = 28
MAX_NEW_TOKENS = 256 # Length of reasoning chain
NUM_SAMPLES = 200    # Number of test samples

###########################
# Architecture (Must match Training)
###########################
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

###########################
# Helpers
###########################
def extract_answer_number(text):
    """Robust extraction of the final number from reasoning text."""
    # If the model follows GSM8K format '#### 42'
    if "####" in text:
        text = text.split("####")[1]
    
    # Otherwise find the last number
    matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if matches:
        return matches[-1]
    return None

def build_prompt(question):
    return f"Question: {question}\nLet's think step by step.\nAnswer:"

def get_dims(tokenizer, model, device):
    dummy = tokenizer("h", return_tensors="pt").to(device)
    with torch.no_grad(): out = model(**dummy, use_cache=True)
    kv = out.past_key_values
    dims = []
    # Handle Qwen/Mistral structure
    if hasattr(kv, 'key_cache'):
        for i in range(NUM_LAYERS):
            k, v = kv.key_cache[i], kv.value_cache[i]
            dims.append({'k_heads': k.shape[1], 'k_head_dim': k.shape[3], 'v_heads': v.shape[1], 'v_head_dim': v.shape[3]})
    else:
        for i in range(NUM_LAYERS):
            dims.append({'k_heads': kv[i][0].shape[1], 'k_head_dim': kv[i][0].shape[3], 'v_heads': kv[i][1].shape[1], 'v_head_dim': kv[i][1].shape[3]})
    return dims

###########################
# Main Evaluation
###########################
def main():
    print(f"Loading Models onto {DEVICE}...")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model_a.eval(); model_b.eval()

    # Load Translators
    print("Loading Translators...")
    dims_a = get_dims(tokenizer_a, model_a, DEVICE)
    dims_b = get_dims(tokenizer_b, model_b, DEVICE)
    
    translators_k = nn.ModuleList()
    translators_v = nn.ModuleList()
    
    state_k = torch.load(CHECKPOINT_PATH_K, map_location=DEVICE)
    state_v = torch.load(CHECKPOINT_PATH_V, map_location=DEVICE)
    
    for i in range(NUM_LAYERS):
        src, tgt = dims_a[i], dims_b[i]
        k_in, k_out = src['k_heads']*src['k_head_dim'], tgt['k_heads']*tgt['k_head_dim']
        v_in, v_out = src['v_heads']*src['v_head_dim'], tgt['v_heads']*tgt['v_head_dim']
        
        tk = AdvancedTranslator(k_in, k_out, tgt['k_heads'], tgt['k_head_dim']).to(DEVICE).to(dtype=torch.float16)
        tv = AdvancedTranslator(v_in, v_out, tgt['v_heads'], tgt['v_head_dim']).to(DEVICE).to(dtype=torch.float16)
        translators_k.append(tk)
        translators_v.append(tv)
    
    translators_k.load_state_dict(state_k)
    translators_v.load_state_dict(state_v)

    # Load Data
    dataset = load_dataset("gsm8k", "main", split="test")
    if NUM_SAMPLES: dataset = dataset.select(range(NUM_SAMPLES))
    
    print(f"\nEvaluating on {len(dataset)} samples...")
    print("="*60)
    
    correct = 0
    total = 0
    
    for i, ex in enumerate(tqdm(dataset)):
        prompt = build_prompt(ex['question'])
        gt_ans = extract_answer_number(ex['answer'])
        
        # 1. Source Pass
        inputs_a = tokenizer_a(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out_a = model_a(**inputs_a, use_cache=True)
            
        # 2. Translation
        translated_cache = DynamicCache()
        kv_a = out_a.past_key_values
        
        with torch.no_grad():
            for layer_idx in range(NUM_LAYERS):
                # Extract
                if hasattr(kv_a, 'key_cache'):
                    k_src, v_src = kv_a.key_cache[layer_idx], kv_a.value_cache[layer_idx]
                else:
                    k_src, v_src = kv_a[layer_idx][0], kv_a[layer_idx][1]
                
                # Translate
                k_trans = translators_k[layer_idx](k_src)
                v_trans = translators_v[layer_idx](v_src)
                translated_cache.update(k_trans, v_trans, layer_idx)
        
        # 3. Target Generation (Blind Reasoning)
        # We start with a blank token and force it to reason based on the cache
        start_token = tokenizer_b(" ", return_tensors="pt").input_ids.to(DEVICE)
        seq_len = translated_cache.get_seq_length()
        attention_mask = torch.ones(1, seq_len + 1, device=DEVICE)
        
        with torch.no_grad():
            generated = model_b.generate(
                input_ids=start_token,
                attention_mask=attention_mask,
                past_key_values=translated_cache,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, # Deterministic evaluation
                pad_token_id=tokenizer_b.eos_token_id,
                cache_position=torch.arange(seq_len, seq_len + 1, device=DEVICE)
            )
        
        output_text = tokenizer_b.decode(generated[0], skip_special_tokens=True)
        pred_ans = extract_answer_number(output_text)
        
        if pred_ans == gt_ans:
            correct += 1
        total += 1
        
        if i < 3:
            print(f"\nQ: {ex['question'][:60]}...")
            print(f"Generated Reasoning: {repr(output_text[:100])}...")
            print(f"GT: {gt_ans} | Pred: {pred_ans} | Correct: {pred_ans == gt_ans}")

    accuracy = correct / total
    print("="*60)
    print(f"Final Accuracy: {accuracy:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()
