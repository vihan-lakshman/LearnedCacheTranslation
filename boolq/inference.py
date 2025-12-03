import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from tqdm import tqdm
import sys

logging.set_verbosity_error()

###########################
# Configuration
###########################
MODEL_TYPE = "qwen"  # 'qwen' or 'mistral'
CHECKPOINT_PATH = "kv_translators_boolq_qwen.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1 
MAX_SAMPLES = None # Set to None for full run, or e.g. 100 for test

if MODEL_TYPE == 'qwen':
    MODEL_A_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_B_ID = "Qwen/Qwen2.5-7B-Instruct"
    NUM_LAYERS = 28
else:
    MODEL_A_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_B_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    NUM_LAYERS = 32

###########################
# Classes
###########################
class SimpleDeepTranslator(nn.Module):
    def __init__(self, input_size, output_size, target_heads, target_head_dim):
        super().__init__()
        self.target_heads = target_heads
        self.target_head_dim = target_head_dim
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

    def forward(self, cache_tensor):
        batch, num_heads, seq_len, head_dim = cache_tensor.shape
        x = cache_tensor.permute(0, 2, 1, 3).contiguous().view(batch * seq_len, -1)
        y = self.net(x)
        y = y.view(batch, seq_len, self.target_heads, self.target_head_dim)
        return y.permute(0, 2, 1, 3).contiguous()

def build_prompt(passage, question):
    return f"""Read the passage and answer the question with just "Yes" or "No".

Passage: {passage}

Question: {question}

Answer:"""

###########################
# Helper Functions
###########################
def get_yes_no_token_ids(tokenizer):
    """Find the token IDs for 'Yes' and 'No'."""
    yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[-1]
    no_id = tokenizer.encode(" No", add_special_tokens=False)[-1]
    print(f"Target Token IDs -> Yes: {yes_id} | No: {no_id}")
    return yes_id, no_id

def load_translators(checkpoint_path, model_a, model_b, device):
    print(f"Loading translators from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    translators_k = nn.ModuleList()
    translators_v = nn.ModuleList()
    
    # Dummy run to get shapes
    dummy = torch.tensor([[100, 101]]).to(device)
    with torch.no_grad():
        out_a = model_a(dummy, use_cache=True)
        out_b = model_b(dummy, use_cache=True)
        
    def get_dims(kv, layer_idx):
        if hasattr(kv, 'key_cache'): 
            k = kv.key_cache[layer_idx]
            v = kv.value_cache[layer_idx]
            return k.shape[1], k.shape[3], v.shape[1], v.shape[3]
        else:
            return kv[layer_idx][0].shape[1], kv[layer_idx][0].shape[3], kv[layer_idx][1].shape[1], kv[layer_idx][1].shape[3]

    for i in range(NUM_LAYERS):
        ak_h, ak_d, av_h, av_d = get_dims(out_a.past_key_values, i)
        bk_h, bk_d, bv_h, bv_d = get_dims(out_b.past_key_values, i)
        
        k_in, k_out = ak_h * ak_d, bk_h * bk_d
        v_in, v_out = av_h * av_d, bv_h * bv_d
        
        # FIX: Explicitly cast to float16 to match Model A output
        t_k = SimpleDeepTranslator(k_in, k_out, bk_h, bk_d).to(device).to(dtype=torch.float16)
        t_v = SimpleDeepTranslator(v_in, v_out, bv_h, bv_d).to(device).to(dtype=torch.float16)
        
        translators_k.append(t_k)
        translators_v.append(t_v)

    translators_k.load_state_dict(checkpoint['translators_k_state_dict'])
    translators_v.load_state_dict(checkpoint['translators_v_state_dict'])
    
    translators_k.eval()
    translators_v.eval()
    
    return translators_k, translators_v

###########################
# Main Evaluation Loop
###########################
def main():
    print(f"Device: {DEVICE}")
    
    # 1. Load Data
    print("Loading BoolQ validation set...")
    dataset = load_dataset("google/boolq")
    eval_data = dataset['validation'] 
    
    if MAX_SAMPLES:
        eval_data = eval_data.select(range(min(len(eval_data), MAX_SAMPLES)))
    
    print(f"Evaluator samples: {len(eval_data)}")

    # 2. Load Models
    print("Loading LLMs...")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A_ID, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B_ID, trust_remote_code=True)
    
    # Ensure pad token exists
    if tokenizer_a.pad_token is None: tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token
    
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A_ID, torch_dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B_ID, torch_dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    
    model_a.eval()
    model_b.eval()

    # 3. Load Translators (FIX APPLIED HERE)
    translators_k, translators_v = load_translators(CHECKPOINT_PATH, model_a, model_b, DEVICE)
    
    # 4. Identify Constraint Tokens
    yes_id, no_id = get_yes_no_token_ids(tokenizer_b)

    # 5. Run Evaluation
    correct = 0
    total = 0
    
    print("\n" + "="*50)
    print("STARTING CONSTRAINED DECODING EVALUATION")
    print("="*50)
    
    # Use autocast context to safely handle any mixed types
    with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16):
        for i, ex in enumerate(tqdm(eval_data)):
            prompt = build_prompt(ex['passage'], ex['question'])
            ground_truth = ex['answer'] # True/False
            
            try:
                # A. Get Source Cache
                inputs_a = tokenizer_a(prompt, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    out_a = model_a(**inputs_a, use_cache=True)
                
                # B. Translate
                translated_cache = DynamicCache()
                kv_a = out_a.past_key_values
                
                with torch.no_grad():
                    for layer_idx in range(NUM_LAYERS):
                        if hasattr(kv_a, 'key_cache'):
                            k_src = kv_a.key_cache[layer_idx]
                            v_src = kv_a.value_cache[layer_idx]
                        else:
                            k_src = kv_a[layer_idx][0]
                            v_src = kv_a[layer_idx][1]
                        
                        k_trans = translators_k[layer_idx](k_src)
                        v_trans = translators_v[layer_idx](v_src)
                        translated_cache.update(k_trans, v_trans, layer_idx)
                
                # C. Constrained Scoring
                start_token = tokenizer_b(" ", return_tensors="pt").input_ids.to(DEVICE)
                seq_len = translated_cache.get_seq_length()
                
                attention_mask = torch.ones(1, seq_len + 1).to(DEVICE)
                position_ids = torch.arange(seq_len, seq_len + 1, device=DEVICE).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model_b(
                        input_ids=start_token,
                        attention_mask=attention_mask,
                        past_key_values=translated_cache,
                        position_ids=position_ids,
                        use_cache=False
                    )
                
                logits = outputs.logits[0, -1, :]
                score_yes = logits[yes_id].item()
                score_no = logits[no_id].item()
                
                prediction = True if score_yes > score_no else False
                
                if prediction == ground_truth:
                    correct += 1
                total += 1
                
            except Exception as e:
                # Print only the first error to avoid spam, or key errors
                if i < 3:
                    print(f"Error on sample {i}: {e}")
                continue

    accuracy = correct / total if total > 0 else 0
    print("\n" + "="*50)
    print(f"FINAL RESULTS")
    print(f"Total Samples: {total}")
    print(f"Correct:       {correct}")
    print(f"Accuracy:      {accuracy:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
