import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from tqdm import trange
import gc
import sys
import re
import string
from collections import Counter

logging.set_verbosity_error()

###########################
# Configuration
###########################
if len(sys.argv) < 3:
    print("Usage: python train_graduated_synthetic_qa.py <qwen|mistral> <num_facts>")
    print("  num_facts: 1, 2, 3, or 4 (number of facts in context)")
    sys.exit(1)

if sys.argv[1] not in ['qwen', 'mistral']:
    print("First argument must be 'qwen' or 'mistral'")
    sys.exit(1)

try:
    NUM_FACTS = int(sys.argv[2])
    if NUM_FACTS not in [1, 2, 3, 4]:
        raise ValueError()
except:
    print("Second argument must be 1, 2, 3, or 4")
    sys.exit(1)

if sys.argv[1] == 'qwen':
    MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
else:
    MODEL_A = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

NUM_LAYERS = 28 if 'Qwen' in MODEL_A else 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = f"kv_translators_synthetic_{NUM_FACTS}facts.pth"

# Configuration
NUM_PROMPTS = 2000
MAX_CTX_TOKENS = 32 + (NUM_FACTS * 32)  # Scale context window with number of facts
TRAIN_STEPS = 3000
BATCH_SIZE = 4
LR = 1e-3
SEED = 42
TRAINING_BATCH_SIZE = 32
COMPILE_MODEL = True
MIXED_PRECISION = True

# Evaluation settings
EVAL_EVERY = 1000
NUM_EVAL_SAMPLES = 100
MAX_NEW_TOKENS = 32
###########################

torch.manual_seed(SEED)
random.seed(SEED)

if not (hasattr(torch, 'compile') and torch.cuda.is_available()):
    COMPILE_MODEL = False

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

def generate_fact():
    """Generate a single fact."""
    subjects = ["Dr. Evans", "The pilot", "The scientist", "Agent Helix", "The cartographer",
                "Captain Rivera", "Professor Kim", "Detective Chen", "Colonel Hayes", "Dr. Martinez"]
    actions = ["placed", "hid", "secured", "delivered", "found", 
               "stored", "moved", "retrieved", "archived", "discovered"]
    objects = ["the blue folder", "the silver key", "the encrypted drive", "the heavy package", "the coded map",
               "the sealed envelope", "the metal case", "the classified document", "the ancient artifact", "the prototype device"]
    locations = ["in the library", "in the hangar", "in the laboratory", "at the north gate", "behind the console",
                 "in the archive room", "at the main entrance", "in the storage facility", "near the fountain", "in the vault"]
    
    subject = random.choice(subjects)
    action = random.choice(actions)
    obj = random.choice(objects)
    location = random.choice(locations)
    
    fact = f"{subject} {action} {obj} {location}."
    
    return {
        'text': fact,
        'subject': subject,
        'action': action,
        'object': obj,
        'location': location
    }

def generate_synthetic_qa(num_facts=1):
    """Generate synthetic QA with multiple facts in context."""
    facts = [generate_fact() for _ in range(num_facts)]
    
    # Build context from all facts
    context = " ".join([f['text'] for f in facts])
    
    # Randomly select one fact to ask about
    target_fact = random.choice(facts)
    
    # Generate question about the target fact
    q_type = random.choice(["who", "what", "where"])
    if q_type == "who":
        question = f"Who {target_fact['action']} {target_fact['object']}?"
        answer = target_fact['subject']
    elif q_type == "what":
        question = f"What did {target_fact['subject']} {target_fact['action']}?"
        answer = target_fact['object']
    else:
        question = f"Where was {target_fact['object']} {target_fact['action']}?"
        answer = target_fact['location']
    
    return context, question, answer

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

def generate_kv_cache(prompts, tokenizer_a, model_a, device, max_length):
    inputs_a = tokenizer_a(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            out_a = model_a(**inputs_a, use_cache=True)
        kv_a = out_a.past_key_values
    source_keys = [kv_a[i][0] for i in range(NUM_LAYERS)]
    source_vals = [kv_a[i][1] for i in range(NUM_LAYERS)]
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

def evaluate_translators(model_a, model_b, tokenizer_a, tokenizer_b, translators_k, translators_v, device, num_facts):
    """Evaluate translators on held-out synthetic examples."""
    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80)
    
    translators_k.eval()
    translators_v.eval()
    model_a.eval()
    model_b.eval()
    
    total_f1 = 0
    
    for i in range(NUM_EVAL_SAMPLES):
        context, question, ground_truth_answer = generate_synthetic_qa(num_facts)
        
        context_prompt = f"Context:\n{context}\n\nQuestion: {question} Answer:"
        question_prompt = " Answer:"
        
        try:
            sk, sv = generate_kv_cache([context_prompt], tokenizer_a, model_a, device, MAX_CTX_TOKENS)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
                    translated_k = [translators_k[i](sk[i]) for i in range(NUM_LAYERS)]
                    translated_v = [translators_v[i](sv[i]) for i in range(NUM_LAYERS)]
            
            translated_cache = tuple(zip(translated_k, translated_v))
            
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
            
            if i < 3:
                print(f"\nEval {i+1}/{NUM_EVAL_SAMPLES}")
                print(f"Context ({len(context.split())} words): {context[:100]}...")
                print(f"Q: {question}")
                print(f"GT: {ground_truth_answer}")
                print(f"Pred: {cleaned_response}")
                print(f"F1: {f1:.4f}")
        
        except Exception as e:
            print(f"Eval example {i+1} failed: {e}")
    
    avg_f1 = total_f1 / NUM_EVAL_SAMPLES
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE: Average F1 = {avg_f1:.4f}")
    print(f"{'='*80}\n")
    
    translators_k.train()
    translators_v.train()
    
    return avg_f1

def main():
    print(f"Device: {DEVICE}")
    print(f"Training with {NUM_FACTS} fact(s) per context")
    print(f"Max context tokens: {MAX_CTX_TOKENS}")
    print("--- Loading Models and Tokenizers ---")

    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)

    if tokenizer_a.pad_token is None: tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token
    model_a.eval()
    model_b.eval()

    print(f"--- Generating Training Data (Synthetic QA with {NUM_FACTS} facts) ---")
    data_by_layer = [[[] for _ in range(NUM_LAYERS)] for _ in range(4)]
    
    for _ in trange(NUM_PROMPTS // BATCH_SIZE, desc="Generating data"):
        prompts = []
        for _ in range(BATCH_SIZE):
            context, question, answer = generate_synthetic_qa(NUM_FACTS)
            prompt = f"Context:\n{context}\n\nQuestion: {question} Answer:"
            prompts.append(prompt)

        sk, sv, tk, tv = generate_kv_cache_pair_batch(prompts, tokenizer_a, tokenizer_b, model_a, model_b, DEVICE, MAX_CTX_TOKENS)
        for i in range(NUM_LAYERS):
            data_by_layer[0][i].append(sk[i])
            data_by_layer[1][i].append(sv[i])
            data_by_layer[2][i].append(tk[i])
            data_by_layer[3][i].append(tv[i])
    
    data_tensors = [[torch.cat(d) for d in layer_data] for layer_data in data_by_layer]
    NUM_PROMPTS_ACTUAL = data_tensors[0][0].shape[0]
    print(f"Generated {NUM_PROMPTS_ACTUAL} training examples.")
    
    # Sample context to show actual length
    sample_context, _, _ = generate_synthetic_qa(NUM_FACTS)
    print(f"Sample context ({len(sample_context.split())} words): {sample_context}")

    print("Keeping models loaded for periodic evaluation...")

    print("--- Creating Translators ---")
    translators_k, translators_v = nn.ModuleList(), nn.ModuleList()
    for i in range(NUM_LAYERS):
        sk, sv, tk, tv = data_tensors[0][i], data_tensors[1][i], data_tensors[2][i], data_tensors[3][i]
        k_in_size = sk.shape[1] * sk.shape[3]
        k_out_size = tk.shape[1] * tk.shape[3]
        v_in_size = sv.shape[1] * sv.shape[3]
        v_out_size = tv.shape[1] * tv.shape[3]
        translators_k.append(SimpleDeepTranslator(k_in_size, k_out_size, tk.shape[1], tk.shape[3]).to(device=DEVICE))
        translators_v.append(SimpleDeepTranslator(v_in_size, v_out_size, tv.shape[1], tv.shape[3]).to(device=DEVICE))
    
    total_params = sum(p.numel() for p in translators_k.parameters()) + sum(p.numel() for p in translators_v.parameters())
    print(f"Total translator parameters: {total_params:,}")

    if COMPILE_MODEL:
        print("Compiling translators for faster training...")
        for i in range(NUM_LAYERS):
            translators_k[i] = torch.compile(translators_k[i])
            translators_v[i] = torch.compile(translators_v[i])

    params = list(translators_k.parameters()) + list(translators_v.parameters())
    optimizer = optim.AdamW(params, lr=LR)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=(MIXED_PRECISION and DEVICE=='cuda'))

    print("--- Starting Training with Periodic Evaluation ---")
    indices = torch.randperm(NUM_PROMPTS_ACTUAL)
    
    eval_history = []
    
    for step in trange(TRAIN_STEPS, desc="Training"):
        translators_k.train()
        translators_v.train()
        
        start_idx = (step * TRAINING_BATCH_SIZE) % NUM_PROMPTS_ACTUAL
        end_idx = start_idx + TRAINING_BATCH_SIZE
        
        if end_idx > NUM_PROMPTS_ACTUAL:
            batch_indices = torch.cat((indices[start_idx:], indices[:end_idx - NUM_PROMPTS_ACTUAL]))
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
        
        if (step + 1) % EVAL_EVERY == 0:
            avg_loss = total_loss.item() / (2 * NUM_LAYERS)
            print(f"\n[Step {step+1}] Avg Layer Loss: {avg_loss:.6f}")
            
            avg_f1 = evaluate_translators(
                model_a, model_b, tokenizer_a, tokenizer_b,
                translators_k, translators_v, DEVICE, NUM_FACTS
            )
            eval_history.append((step + 1, avg_loss, avg_f1))
            
            checkpoint_path = f"kv_translators_synthetic_{NUM_FACTS}facts_step{step+1}.pth"
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
                'step': step + 1,
                'loss': avg_loss,
                'f1': avg_f1,
                'num_facts': NUM_FACTS,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("\nTraining complete.")
    
    print("\n" + "="*80)
    print(f"EVALUATION HISTORY ({NUM_FACTS} FACTS)")
    print("="*80)
    print(f"{'Step':<10} {'Loss':<12} {'F1 Score':<12}")
    print("-" * 80)
    for step, loss, f1 in eval_history:
        print(f"{step:<10} {loss:<12.6f} {f1:<12.4f}")
    print("="*80)
    
    print(f"\n--- Saving Final Translators to {SAVE_PATH} ---")
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
        'eval_history': eval_history,
        'num_facts': NUM_FACTS,
    }, SAVE_PATH)
    print(f"✅ Final translators saved successfully to {SAVE_PATH}")

if __name__ == "__main__":
    main()
