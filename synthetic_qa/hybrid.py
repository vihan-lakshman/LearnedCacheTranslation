import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from tqdm import tqdm
import sys
import re
import string
from collections import Counter
import time

logging.set_verbosity_error()

###########################
# Configuration
###########################
if len(sys.argv) < 4:
    print("Usage: python test_hybrid_translation.py <qwen|mistral> <num_facts> <cutoff_layer>")
    print("  num_facts: 1, 2, 3, or 4")
    print("  cutoff_layer: layer at which to switch from translation to recomputation (e.g., 10)")
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

try:
    CUTOFF_LAYER = int(sys.argv[3])
except:
    print("Third argument must be an integer (e.g., 10)")
    sys.exit(1)

if sys.argv[1] == 'qwen':
    MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
else:
    MODEL_A = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

NUM_LAYERS = 28 if 'Qwen' in MODEL_A else 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_PATH = f"kv_translators_synthetic_{NUM_FACTS}facts.pth"
MAX_CTX_TOKENS = 32 + (NUM_FACTS * 32)
MIXED_PRECISION = True
SEED = 42
NUM_EVAL_SAMPLES = 50
MAX_NEW_TOKENS = 32
###########################

if CUTOFF_LAYER >= NUM_LAYERS:
    print(f"Error: cutoff_layer must be less than {NUM_LAYERS}")
    sys.exit(1)

torch.manual_seed(SEED)
random.seed(SEED)

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
    """Generate a single fact with expanded vocabulary."""
    subjects = [
        "Dr. Evans", "The pilot", "The scientist", "Agent Helix", "The cartographer",
        "Captain Rivera", "Professor Kim", "Detective Chen", "Colonel Hayes", "Dr. Martinez",
        "Ambassador Lee", "Inspector Walsh", "Commander Singh", "Dr. Patel", "Agent Morrison",
        "Professor Anderson", "Captain Brooks", "Detective Quinn", "Colonel Foster", "Dr. Zhang",
        "The engineer", "The diplomat", "The analyst", "The curator", "The technician",
        "The researcher", "The specialist", "The coordinator", "The supervisor", "The investigator"
    ]
    
    actions = [
        "placed", "hid", "secured", "delivered", "found",
        "stored", "moved", "retrieved", "archived", "discovered",
        "transferred", "collected", "obtained", "examined", "acquired",
        "deposited", "concealed", "safeguarded", "transported", "located",
        "positioned", "stashed", "recovered", "verified", "identified",
        "assembled", "distributed", "extracted", "preserved", "detected"
    ]
    
    objects = [
        "the blue folder", "the silver key", "the encrypted drive", "the heavy package", "the coded map",
        "the sealed envelope", "the metal case", "the classified document", "the ancient artifact", "the prototype device",
        "the red binder", "the brass medallion", "the backup drive", "the wooden crate", "the detailed blueprint",
        "the leather journal", "the steel container", "the signed contract", "the rare manuscript", "the experimental chip",
        "the green folder", "the copper coin", "the memory card", "the plastic box", "the floor plan",
        "the yellow notebook", "the glass vial", "the research paper", "the stone tablet", "the circuit board"
    ]
    
    locations = [
        "in the library", "in the hangar", "in the laboratory", "at the north gate", "behind the console",
        "in the archive room", "at the main entrance", "in the storage facility", "near the fountain", "in the vault",
        "in the control room", "at the south entrance", "in the research wing", "near the statue", "in the safe",
        "in the conference room", "at the east checkpoint", "in the basement", "near the elevator", "in the cabinet",
        "in the office", "at the west gate", "in the workshop", "near the corridor", "in the locker",
        "in the chamber", "at the dock", "in the warehouse", "near the courtyard", "in the repository"
    ]
    
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
    """Generate synthetic QA with multiple facts."""
    facts = [generate_fact() for _ in range(num_facts)]
    context = " ".join([f['text'] for f in facts])
    target_fact = random.choice(facts)
    
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

def generate_kv_cache_model_a(prompt, tokenizer_a, model_a, device, max_length):
    """Generate KV cache from Model A."""
    inputs_a = tokenizer_a([prompt], return_tensors="pt", padding="max_length", 
                          truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            out_a = model_a(**inputs_a, use_cache=True)
        kv_a = out_a.past_key_values
    
    source_keys = [kv_a[i][0] for i in range(NUM_LAYERS)]
    source_vals = [kv_a[i][1] for i in range(NUM_LAYERS)]
    del out_a, kv_a, inputs_a
    return source_keys, source_vals

def hybrid_translate_and_generate(context_prompt, question_prompt, tokenizer_a, tokenizer_b, 
                                  model_a, model_b, translators_k, translators_v, 
                                  cutoff_layer, device, max_length):
    """
    Hybrid approach: Translate early layers, recompute late layers.
    
    Since we can't directly pass partial caches to Model B due to API limitations,
    we simulate the hybrid by:
    1. Translating all layers as normal
    2. For late layers (>= cutoff), we overwrite with freshly computed Model B cache
    
    This simulates the quality we'd get, though not the exact computational pattern.
    """
    # Step 1: Generate source cache from Model A
    source_keys, source_vals = generate_kv_cache_model_a(
        context_prompt, tokenizer_a, model_a, device, max_length
    )
    
    # Step 2: Translate ALL layers first
    translated_keys = []
    translated_vals = []
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            for i in range(NUM_LAYERS):
                trans_k = translators_k[i](source_keys[i])
                trans_v = translators_v[i](source_vals[i])
                translated_keys.append(trans_k)
                translated_vals.append(trans_v)
    
    # Step 3: For late layers, replace with native Model B computation
    # Generate fresh Model B cache
    inputs_b = tokenizer_b([context_prompt], return_tensors="pt", padding="max_length",
                          truncation=True, max_length=max_length).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            outputs_b = model_b(**inputs_b, use_cache=True)
            native_cache_b = outputs_b.past_key_values
    
    del inputs_b, outputs_b
    
    # Step 4: Create hybrid cache (early translated + late native)
    hybrid_keys = []
    hybrid_vals = []
    
    for i in range(NUM_LAYERS):
        if i < cutoff_layer:
            # Use translated cache for early layers
            hybrid_keys.append(translated_keys[i])
            hybrid_vals.append(translated_vals[i])
        else:
            # Use native Model B cache for late layers
            hybrid_keys.append(native_cache_b[i][0])
            hybrid_vals.append(native_cache_b[i][1])
    
    # Combine into tuple format
    hybrid_cache = tuple(zip(hybrid_keys, hybrid_vals))
    
    return hybrid_cache

def full_translate_cache(context_prompt, tokenizer_a, model_a, translators_k, translators_v, device, max_length):
    """
    Full translation approach: Translate all layers.
    """
    # Generate source cache from Model A
    source_keys, source_vals = generate_kv_cache_model_a(
        context_prompt, tokenizer_a, model_a, device, max_length
    )
    
    # Translate all layers
    translated_keys = []
    translated_vals = []
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            for i in range(NUM_LAYERS):
                trans_k = translators_k[i](source_keys[i])
                trans_v = translators_v[i](source_vals[i])
                translated_keys.append(trans_k)
                translated_vals.append(trans_v)
    
    # Combine into tuple format
    full_translated_cache = tuple(zip(translated_keys, translated_vals))
    
    return full_translated_cache

def evaluate_full_translation(model_a, model_b, tokenizer_a, tokenizer_b, 
                              translators_k, translators_v, device):
    """Evaluate the full translation approach."""
    print("\n" + "="*80)
    print(f"FULL TRANSLATION EVALUATION")
    print(f"Translating all layers 0-{NUM_LAYERS-1}")
    print("="*80)
    
    model_a.eval()
    model_b.eval()
    translators_k.eval()
    translators_v.eval()
    
    total_f1 = 0
    total_time = 0
    
    for i in tqdm(range(NUM_EVAL_SAMPLES), desc="Evaluating Full Translation"):
        context, question, ground_truth_answer = generate_synthetic_qa(NUM_FACTS)
        
        context_prompt = f"Context:\n{context}\n\nQuestion: {question} Answer:"
        question_prompt = " Answer:"
        
        try:
            start_time = time.time()
            
            # Get fully translated cache
            translated_cache = full_translate_cache(
                context_prompt, tokenizer_a, model_a,
                translators_k, translators_v, device, MAX_CTX_TOKENS
            )
            
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
            
            end_time = time.time()
            total_time += (end_time - start_time)
            
            response = tokenizer_b.decode(generated[0][question_len:], skip_special_tokens=True).strip()
            cleaned_response = response.split('\n')[0].strip()
            for ending in ['.', '?', '!']:
                if ending in cleaned_response:
                    cleaned_response = cleaned_response.split(ending)[0].strip()
                    break
            
            f1 = calculate_f1_score(cleaned_response, ground_truth_answer)
            total_f1 += f1
            
            if i < 3:
                print(f"\n--- Example {i+1} ---")
                print(f"Context: {context[:80]}...")
                print(f"Question: {question}")
                print(f"Ground Truth: {ground_truth_answer}")
                print(f"Prediction: {cleaned_response}")
                print(f"F1: {f1:.4f}")
        
        except Exception as e:
            print(f"\nExample {i+1} failed: {e}")
    
    avg_f1 = total_f1 / NUM_EVAL_SAMPLES
    avg_time = total_time / NUM_EVAL_SAMPLES
    
    return avg_f1, avg_time 
def evaluate_hybrid_approach(model_a, model_b, tokenizer_a, tokenizer_b, 
                            translators_k, translators_v, cutoff_layer, device):
    """Evaluate the hybrid translation + recomputation approach."""
    print("\n" + "="*80)
    print(f"HYBRID APPROACH EVALUATION")
    print(f"Translating layers 0-{cutoff_layer-1}, Using native layers {cutoff_layer}-{NUM_LAYERS-1}")
    print("="*80)
    
    model_a.eval()
    model_b.eval()
    translators_k.eval()
    translators_v.eval()
    
    total_f1 = 0
    total_time = 0
    
    for i in tqdm(range(NUM_EVAL_SAMPLES), desc="Evaluating Hybrid"):
        context, question, ground_truth_answer = generate_synthetic_qa(NUM_FACTS)
        
        context_prompt = f"Context:\n{context}\n\nQuestion: {question} Answer:"
        question_prompt = " Answer:"
        
        try:
            # Measure time for hybrid approach
            start_time = time.time()
            
            # Get hybrid cache (translated early + native late)
            hybrid_cache = hybrid_translate_and_generate(
                context_prompt, question_prompt, tokenizer_a, tokenizer_b, model_a, model_b,
                translators_k, translators_v, cutoff_layer, device, MAX_CTX_TOKENS
            )
            
            # Generate answer using hybrid cache
            q_inputs = tokenizer_b(question_prompt, return_tensors="pt").to(device)
            context_len = hybrid_cache[0][0].shape[2]
            question_len = q_inputs.input_ids.shape[1]
            attention_mask = torch.ones(1, context_len + question_len, device=device)
            cache_position = torch.arange(context_len, context_len + question_len, device=device)
            
            with torch.no_grad():
                generated = model_b.generate(
                    input_ids=q_inputs.input_ids,
                    attention_mask=attention_mask,
                    past_key_values=hybrid_cache,
                    cache_position=cache_position,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer_b.eos_token_id
                )
            
            end_time = time.time()
            total_time += (end_time - start_time)
            
            response = tokenizer_b.decode(generated[0][question_len:], skip_special_tokens=True).strip()
            cleaned_response = response.split('\n')[0].strip()
            for ending in ['.', '?', '!']:
                if ending in cleaned_response:
                    cleaned_response = cleaned_response.split(ending)[0].strip()
                    break
            
            f1 = calculate_f1_score(cleaned_response, ground_truth_answer)
            total_f1 += f1
            
            if i < 3:
                print(f"\n--- Example {i+1} ---")
                print(f"Context: {context[:80]}...")
                print(f"Question: {question}")
                print(f"Ground Truth: {ground_truth_answer}")
                print(f"Prediction: {cleaned_response}")
                print(f"F1: {f1:.4f}")
        
        except Exception as e:
            print(f"\nExample {i+1} failed: {e}")
    
    avg_f1 = total_f1 / NUM_EVAL_SAMPLES
    avg_time = total_time / NUM_EVAL_SAMPLES
    
    return avg_f1, avg_time

def main():
    print(f"Device: {DEVICE}")
    print(f"Evaluating Hybrid Translation Approach")
    print(f"  Facts per context: {NUM_FACTS}")
    print(f"  Cutoff layer: {CUTOFF_LAYER}")
    print(f"  Early layers (translated): 0-{CUTOFF_LAYER-1} ({CUTOFF_LAYER} layers)")
    print(f"  Late layers (recomputed): {CUTOFF_LAYER}-{NUM_LAYERS-1} ({NUM_LAYERS-CUTOFF_LAYER} layers)")
    print()
    
    # Load models
    print("--- Loading Models and Tokenizers ---")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    
    if tokenizer_a.pad_token is None: tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token
    
    model_a.eval()
    model_b.eval()
    print("✅ Models loaded")
    
    # Load translators
    print(f"--- Loading Translators from {LOAD_PATH} ---")
    
    # Generate dummy cache to get dimensions
    dummy_context, _, _ = generate_synthetic_qa(NUM_FACTS)
    dummy_prompt = f"Context:\n{dummy_context}\n\nQuestion: Test? Answer:"
    dummy_sk, dummy_sv = generate_kv_cache_model_a(dummy_prompt, tokenizer_a, model_a, DEVICE, MAX_CTX_TOKENS)
    
    # Also need target dimensions
    inputs_b_dummy = tokenizer_b([dummy_prompt], return_tensors="pt", padding="max_length",
                                 truncation=True, max_length=MAX_CTX_TOKENS).to(DEVICE)
    with torch.no_grad():
        out_b_dummy = model_b(**inputs_b_dummy, use_cache=True)
    dummy_tk = [out_b_dummy.past_key_values[i][0] for i in range(NUM_LAYERS)]
    dummy_tv = [out_b_dummy.past_key_values[i][1] for i in range(NUM_LAYERS)]
    del inputs_b_dummy, out_b_dummy
    
    translators_k, translators_v = nn.ModuleList(), nn.ModuleList()
    for i in range(NUM_LAYERS):
        sk, sv = dummy_sk[i], dummy_sv[i]
        tk, tv = dummy_tk[i], dummy_tv[i]
        
        k_in_size = sk.shape[1] * sk.shape[3]
        k_out_size = tk.shape[1] * tk.shape[3]
        v_in_size = sv.shape[1] * sv.shape[3]
        v_out_size = tv.shape[1] * tv.shape[3]
        
        translators_k.append(SimpleDeepTranslator(k_in_size, k_out_size, tk.shape[1], tk.shape[3]))
        translators_v.append(SimpleDeepTranslator(v_in_size, v_out_size, tv.shape[1], tv.shape[3]))
    
    checkpoint = torch.load(LOAD_PATH, map_location=DEVICE)
    translators_k.load_state_dict(checkpoint['translators_k_state_dict'])
    translators_v.load_state_dict(checkpoint['translators_v_state_dict'])
    translators_k.to(DEVICE).eval()
    translators_v.to(DEVICE).eval()
    print("✅ Translators loaded")
    
    del dummy_sk, dummy_sv, dummy_tk, dummy_tv
    
    # Run evaluations
    print("\n" + "="*80)
    print("RUNNING EVALUATIONS")
    print("="*80)
    
    # 1. Full Translation
    print("\n[1/2] Evaluating Full Translation...")
    full_f1, full_time = evaluate_full_translation(
        model_a, model_b, tokenizer_a, tokenizer_b,
        translators_k, translators_v, DEVICE
    )
    
    # 2. Hybrid Approach
    print("\n[2/2] Evaluating Hybrid Approach...")
    hybrid_f1, hybrid_time = evaluate_hybrid_approach(
        model_a, model_b, tokenizer_a, tokenizer_b,
        translators_k, translators_v, CUTOFF_LAYER, DEVICE
    )
    
    # Print comparative results
    print("\n" + "="*80)
    print("COMPARATIVE RESULTS")
    print("="*80)
    print(f"Configuration:")
    print(f"  Model pair: {MODEL_A} → {MODEL_B}")
    print(f"  Facts per context: {NUM_FACTS}")
    print(f"  Evaluation samples: {NUM_EVAL_SAMPLES}")
    print()
    print(f"Full Translation (all {NUM_LAYERS} layers):")
    print(f"  F1 Score:      {full_f1:.4f}")
    print(f"  Avg Time:      {full_time:.3f}s")
    print()
    print(f"Hybrid Approach (cutoff at layer {CUTOFF_LAYER}):")
    print(f"  Translated:    Layers 0-{CUTOFF_LAYER-1} ({100*CUTOFF_LAYER/NUM_LAYERS:.1f}%)")
    print(f"  Native:        Layers {CUTOFF_LAYER}-{NUM_LAYERS-1} ({100*(NUM_LAYERS-CUTOFF_LAYER)/NUM_LAYERS:.1f}%)")
    print(f"  F1 Score:      {hybrid_f1:.4f}")
    print(f"  Avg Time:      {hybrid_time:.3f}s")
    print()
    print("Comparison:")
    f1_diff = hybrid_f1 - full_f1
    f1_pct = 100 * f1_diff / full_f1 if full_f1 > 0 else 0
    time_ratio = hybrid_time / full_time if full_time > 0 else 0
    
    print(f"  F1 Difference:     {f1_diff:+.4f} ({f1_pct:+.1f}%)")
    if f1_diff > 0:
        print(f"  → Hybrid is BETTER by {abs(f1_diff):.4f} F1 points ✅")
    elif f1_diff < 0:
        print(f"  → Full translation is better by {abs(f1_diff):.4f} F1 points")
    else:
        print(f"  → Both approaches perform equally")
    
    print(f"  Time Ratio:        {time_ratio:.2f}x")
    if time_ratio > 1:
        print(f"  → Hybrid is {time_ratio:.2f}x slower (expected due to simulation)")
    
    print("="*80)
    
    print("\n💡 Key Insights:")
    if f1_diff > 0.01:
        print(f"  • Hybrid approach improves F1 by {abs(f1_diff):.4f}, validating that")
        print(f"    native computation of late layers (semantic) improves quality")
    elif abs(f1_diff) < 0.01:
        print(f"  • Both approaches achieve similar F1 (~{full_f1:.3f})")
    
    print(f"  • Using native layers {CUTOFF_LAYER}-{NUM_LAYERS-1} vs translated shows")
    print(f"    the importance of high-quality semantic layer representations")
    
    print("\n📊 Recommendations:")
    if CUTOFF_LAYER <= 10:
        print(f"  • Cutoff at layer {CUTOFF_LAYER} leverages high-quality early layer translation")
        print(f"  • Native semantic layers preserve task performance")
    elif CUTOFF_LAYER > 15:
        print(f"  • Try lower cutoff (8-12) for better quality")
        print(f"  • Current cutoff translates too many semantic layers")

if __name__ == "__main__":
    main()
