import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset
from tqdm import tqdm
import sys
import re
import string
from collections import Counter
import time
import traceback

logging.set_verbosity_error()

###########################
# Configuration
###########################
if len(sys.argv) < 3:
    print("Usage: python test_hybrid_translation_musique.py <qwen|mistral> <cutoff_layer>")
    print("  cutoff_layer: layer at which to switch from translation to recomputation (e.g., 10)")
    print("  Special: cutoff_layer=0 means use native Model B for all layers")
    sys.exit(1)

if sys.argv[1] not in ['qwen', 'mistral']:
    print("First argument must be 'qwen' or 'mistral'")
    sys.exit(1)

try:
    CUTOFF_LAYER = int(sys.argv[2])
except:
    print("Second argument must be an integer (e.g., 10)")
    sys.exit(1)

if sys.argv[1] == 'qwen':
    MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
else:
    MODEL_A = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

NUM_LAYERS = 28 if 'Qwen' in MODEL_A else 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_PATH = "kv_translators_musique.pth"
MAX_CTX_TOKENS = 1024
MIXED_PRECISION = True
SEED = 42
NUM_EVAL_SAMPLES = 50
MAX_NEW_TOKENS = 64
###########################

if CUTOFF_LAYER > NUM_LAYERS:
    print(f"Error: cutoff_layer must be <= {NUM_LAYERS}")
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

def format_musique_prompt(example):
    """Format a MuSiQue example into prompts for different evaluation modes."""
    supporting_idx = example['question_decomposition'][0]['paragraph_support_idx']
    
    supporting_paragraphs = [
        p['paragraph_text'] 
        for p in example['paragraphs'] 
        if p['idx'] == supporting_idx
    ]
    
    context_paragraphs = "\n\n".join(supporting_paragraphs)
    question_text = example['question_decomposition'][0]['question']
    
    full_prompt = f"Context:\n{context_paragraphs}\n\nQuestion: {question_text} Answer:"
    context_only = f"Context:\n{context_paragraphs}\n\nQuestion: {question_text}"
    
    return full_prompt, context_only, question_text, example['question_decomposition'][0].get('answer', '')

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
    inputs_a = tokenizer_a([prompt], return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            out_a = model_a(**inputs_a, use_cache=True)
        kv_a = out_a.past_key_values
    
    source_keys = [kv_a[i][0] for i in range(NUM_LAYERS)]
    source_vals = [kv_a[i][1] for i in range(NUM_LAYERS)]
    del out_a, kv_a, inputs_a
    return source_keys, source_vals

def hybrid_translate_and_generate(context_only, tokenizer_a, tokenizer_b, 
                                  model_a, model_b, translators_k, translators_v, 
                                  cutoff_layer, device, max_length):
    """Hybrid approach: Translate early layers, recompute late layers."""
    
    if cutoff_layer == 0:
        inputs_b = tokenizer_b([context_only], return_tensors="pt", truncation=True,
                              max_length=max_length).to(device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
                outputs_b = model_b(**inputs_b, use_cache=True)
                native_cache_b = outputs_b.past_key_values
        
        del inputs_b, outputs_b
        return native_cache_b
    
    source_keys, source_vals = generate_kv_cache_model_a(
        context_only, tokenizer_a, model_a, device, max_length
    )
    
    translated_keys = []
    translated_vals = []
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            for i in range(NUM_LAYERS):
                trans_k = translators_k[i](source_keys[i])
                trans_v = translators_v[i](source_vals[i])
                translated_keys.append(trans_k)
                translated_vals.append(trans_v)
    
    inputs_b = tokenizer_b([context_only], return_tensors="pt", truncation=True,
                          max_length=max_length).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            outputs_b = model_b(**inputs_b, use_cache=True)
            native_cache_b = outputs_b.past_key_values
    
    del inputs_b, outputs_b
    
    hybrid_keys = []
    hybrid_vals = []
    
    for i in range(NUM_LAYERS):
        if i < cutoff_layer:
            hybrid_keys.append(translated_keys[i])
            hybrid_vals.append(translated_vals[i])
        else:
            hybrid_keys.append(native_cache_b[i][0])
            hybrid_vals.append(native_cache_b[i][1])
    
    hybrid_cache = tuple(zip(hybrid_keys, hybrid_vals))
    
    return hybrid_cache

def full_translate_cache(context_only, tokenizer_a, model_a, translators_k, translators_v, device, max_length):
    """Full translation approach: Translate all layers."""
    source_keys, source_vals = generate_kv_cache_model_a(
        context_only, tokenizer_a, model_a, device, max_length
    )
    
    translated_keys = []
    translated_vals = []
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            for i in range(NUM_LAYERS):
                trans_k = translators_k[i](source_keys[i])
                trans_v = translators_v[i](source_vals[i])
                translated_keys.append(trans_k)
                translated_vals.append(trans_v)
    
    full_translated_cache = tuple(zip(translated_keys, translated_vals))
    
    return full_translated_cache

def evaluate_baseline_model_b(model_b, tokenizer_b, device, eval_dataset):
    """Evaluate Model B baseline (full text processing, no translation, single-pass)."""
    print("\n" + "="*80)
    print(f"BASELINE EVALUATION: Model B with Full Text (Single-Pass)")
    print("="*80)
    
    model_b.eval()
    
    total_f1 = 0
    total_time = 0
    num_evaluated = 0
    
    for i in tqdm(range(min(NUM_EVAL_SAMPLES, len(eval_dataset))), desc="Evaluating Baseline"):
        example = eval_dataset[i]
        full_prompt, context_only, question, ground_truth_answer = format_musique_prompt(example)
        
        if not ground_truth_answer:
            continue
        
        try:
            start_time = time.time()
            
            inputs = tokenizer_b(full_prompt, return_tensors="pt", truncation=True, max_length=MAX_CTX_TOKENS).to(device)
            
            with torch.no_grad():
                generated = model_b.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer_b.eos_token_id
                )
            
            end_time = time.time()
            total_time += (end_time - start_time)
            
            input_length = inputs.input_ids.shape[1]
            response = tokenizer_b.decode(generated[0][input_length:], skip_special_tokens=True).strip()
            cleaned_response = response.split('\n')[0].strip()
            for ending in ['.', '?', '!']:
                if ending in cleaned_response:
                    cleaned_response = cleaned_response.split(ending)[0].strip()
                    break
            
            f1 = calculate_f1_score(cleaned_response, ground_truth_answer)
            total_f1 += f1
            num_evaluated += 1
            
            if i < 2:
                print(f"\n--- Example {i+1} ---")
                print(f"Question: {question}")
                print(f"Ground Truth: {ground_truth_answer}")
                print(f"Prediction: {cleaned_response}")
                print(f"F1: {f1:.4f}")
        
        except Exception as e:
            print(f"\nExample {i+1} failed: {e}")
            traceback.print_exc()
    
    avg_f1 = total_f1 / num_evaluated if num_evaluated > 0 else 0
    avg_time = total_time / num_evaluated if num_evaluated > 0 else 0
    
    return avg_f1, avg_time, num_evaluated

def evaluate_native_cache_model_b(model_b, tokenizer_b, device, eval_dataset):
    """Evaluate Model B using native KV cache (two-pass, no translation)."""
    print("\n" + "="*80)
    print(f"NATIVE CACHE EVALUATION: Model B with Native KV Cache (Two-Pass)")
    print("="*80)
    
    model_b.eval()
    
    total_f1 = 0
    total_time = 0
    num_evaluated = 0
    
    for i in tqdm(range(min(NUM_EVAL_SAMPLES, len(eval_dataset))), desc="Evaluating Native Cache"):
        example = eval_dataset[i]
        full_prompt, context_only, question, ground_truth_answer = format_musique_prompt(example)
        
        if not ground_truth_answer:
            continue
        
        try:
            start_time = time.time()
            
            inputs_b = tokenizer_b([context_only], return_tensors="pt", truncation=True,
                                  max_length=MAX_CTX_TOKENS).to(device)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
                    outputs_b = model_b(**inputs_b, use_cache=True)
                    native_cache = outputs_b.past_key_values
            
            context_len = native_cache[0][0].shape[2]
            del inputs_b, outputs_b
            
            # Continue with " Answer:"
            answer_prompt = " Answer:"
            q_inputs = tokenizer_b(answer_prompt, return_tensors="pt").to(device)
            question_len = q_inputs.input_ids.shape[1]
            
            # Create proper attention mask and cache_position
            attention_mask = torch.ones(1, context_len + question_len, device=device, dtype=torch.long)
            cache_position = torch.arange(context_len, context_len + question_len, device=device, dtype=torch.long)
            
            with torch.no_grad():
                generated = model_b.generate(
                    input_ids=q_inputs.input_ids,
                    attention_mask=attention_mask,
                    past_key_values=native_cache,
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
            num_evaluated += 1
            
            if i < 2:
                print(f"\n--- Example {i+1} ---")
                print(f"Question: {question}")
                print(f"Ground Truth: {ground_truth_answer}")
                print(f"Prediction: {cleaned_response}")
                print(f"F1: {f1:.4f}")
        
        except Exception as e:
            print(f"\nExample {i+1} failed: {e}")
            traceback.print_exc()
    
    avg_f1 = total_f1 / num_evaluated if num_evaluated > 0 else 0
    avg_time = total_time / num_evaluated if num_evaluated > 0 else 0
    
    return avg_f1, avg_time, num_evaluated

def evaluate_full_translation(model_a, model_b, tokenizer_a, tokenizer_b, 
                              translators_k, translators_v, device, eval_dataset):
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
    num_evaluated = 0
    
    for i in tqdm(range(min(NUM_EVAL_SAMPLES, len(eval_dataset))), desc="Evaluating Full Translation"):
        example = eval_dataset[i]
        full_prompt, context_only, question, ground_truth_answer = format_musique_prompt(example)
        
        if not ground_truth_answer:
            continue
        
        try:
            start_time = time.time()
            
            translated_cache = full_translate_cache(
                context_only, tokenizer_a, model_a,
                translators_k, translators_v, device, MAX_CTX_TOKENS
            )
            
            context_len = translated_cache[0][0].shape[2]
            
            answer_prompt = " Answer:"
            q_inputs = tokenizer_b(answer_prompt, return_tensors="pt").to(device)
            question_len = q_inputs.input_ids.shape[1]
            
            attention_mask = torch.ones(1, context_len + question_len, device=device, dtype=torch.long)
            cache_position = torch.arange(context_len, context_len + question_len, device=device, dtype=torch.long)
            
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
            num_evaluated += 1
            
            if i < 2:
                print(f"\n--- Example {i+1} ---")
                print(f"Question: {question}")
                print(f"Ground Truth: {ground_truth_answer}")
                print(f"Prediction: {cleaned_response}")
                print(f"F1: {f1:.4f}")
        
        except Exception as e:
            print(f"\nExample {i+1} failed: {e}")
            traceback.print_exc()
    
    avg_f1 = total_f1 / num_evaluated if num_evaluated > 0 else 0
    avg_time = total_time / num_evaluated if num_evaluated > 0 else 0
    
    return avg_f1, avg_time, num_evaluated

def evaluate_hybrid_approach(model_a, model_b, tokenizer_a, tokenizer_b, 
                            translators_k, translators_v, cutoff_layer, device, eval_dataset):
    """Evaluate the hybrid translation + recomputation approach."""
    print("\n" + "="*80)
    print(f"HYBRID APPROACH EVALUATION")
    if cutoff_layer == 0:
        print(f"Using native Model B for ALL layers (cutoff=0)")
    else:
        print(f"Translating layers 0-{cutoff_layer-1}, Using native layers {cutoff_layer}-{NUM_LAYERS-1}")
    print("="*80)
    
    model_a.eval()
    model_b.eval()
    translators_k.eval()
    translators_v.eval()
    
    total_f1 = 0
    total_time = 0
    num_evaluated = 0
    
    for i in tqdm(range(min(NUM_EVAL_SAMPLES, len(eval_dataset))), desc="Evaluating Hybrid"):
        example = eval_dataset[i]
        full_prompt, context_only, question, ground_truth_answer = format_musique_prompt(example)
        
        if not ground_truth_answer:
            continue
        
        try:
            start_time = time.time()
            
            hybrid_cache = hybrid_translate_and_generate(
                context_only, tokenizer_a, tokenizer_b, model_a, model_b,
                translators_k, translators_v, cutoff_layer, device, MAX_CTX_TOKENS
            )
            
            context_len = hybrid_cache[0][0].shape[2]
            
            answer_prompt = " Answer:"
            q_inputs = tokenizer_b(answer_prompt, return_tensors="pt").to(device)
            question_len = q_inputs.input_ids.shape[1]
            
            attention_mask = torch.ones(1, context_len + question_len, device=device, dtype=torch.long)
            cache_position = torch.arange(context_len, context_len + question_len, device=device, dtype=torch.long)
            
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
            num_evaluated += 1
            
            if i < 2:
                print(f"\n--- Example {i+1} ---")
                print(f"Question: {question}")
                print(f"Ground Truth: {ground_truth_answer}")
                print(f"Prediction: {cleaned_response}")
                print(f"F1: {f1:.4f}")
        
        except Exception as e:
            print(f"\nExample {i+1} failed: {e}")
            traceback.print_exc()
    
    avg_f1 = total_f1 / num_evaluated if num_evaluated > 0 else 0
    avg_time = total_time / num_evaluated if num_evaluated > 0 else 0
    
    return avg_f1, avg_time, num_evaluated

def main():
    print(f"Device: {DEVICE}")
    print(f"Evaluating Hybrid Translation Approach on MuSiQue Dataset")
    print(f"  Cutoff layer: {CUTOFF_LAYER}")
    if CUTOFF_LAYER == 0:
        print(f"  Special mode: Using native Model B for ALL layers")
    else:
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
    
    # Load MuSiQue validation dataset
    print("--- Loading MuSiQue Validation Dataset ---")
    try:
        eval_dataset = load_dataset("dgslibisey/MuSiQue", split="validation").select(range(NUM_EVAL_SAMPLES))
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        print(f"Loading smaller subset...")
        eval_dataset = load_dataset("dgslibisey/MuSiQue", split="validation").select(range(min(NUM_EVAL_SAMPLES, 20)))
    
    print(f"Loaded {len(eval_dataset)} evaluation examples")
    
    # Load translators
    print(f"--- Loading Translators from {LOAD_PATH} ---")
    
    dummy_example = eval_dataset[0]
    _, dummy_context_only, _, _ = format_musique_prompt(dummy_example)
    dummy_sk, dummy_sv = generate_kv_cache_model_a(dummy_context_only, tokenizer_a, model_a, DEVICE, MAX_CTX_TOKENS)
    
    inputs_b_dummy = tokenizer_b([dummy_context_only], return_tensors="pt", truncation=True,
                                 max_length=MAX_CTX_TOKENS).to(DEVICE)
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
    
    print("\n[1/4] Evaluating Baseline (Model B, single-pass)...")
    baseline_f1, baseline_time, baseline_count = evaluate_baseline_model_b(
        model_b, tokenizer_b, DEVICE, eval_dataset
    )
    
    print("\n[2/4] Evaluating Native Cache (Model B, two-pass with cache)...")
    native_cache_f1, native_cache_time, native_cache_count = evaluate_native_cache_model_b(
        model_b, tokenizer_b, DEVICE, eval_dataset
    )
    
    print("\n[3/4] Evaluating Full Translation...")
    full_f1, full_time, full_count = evaluate_full_translation(
        model_a, model_b, tokenizer_a, tokenizer_b,
        translators_k, translators_v, DEVICE, eval_dataset
    )
    
    print("\n[4/4] Evaluating Hybrid Approach...")
    hybrid_f1, hybrid_time, hybrid_count = evaluate_hybrid_approach(
        model_a, model_b, tokenizer_a, tokenizer_b,
        translators_k, translators_v, CUTOFF_LAYER, DEVICE, eval_dataset
    )
    
    # Print results
    print("\n" + "="*80)
    print("COMPARATIVE RESULTS (MuSiQue Dataset)")
    print("="*80)
    print(f"  Evaluation samples: {baseline_count} (with valid answers)\n")
    
    print(f"Baseline (Single-Pass):        F1 = {baseline_f1:.4f}  ⭐")
    print(f"Native Cache (Two-Pass):       F1 = {native_cache_f1:.4f}  (diff: {native_cache_f1-baseline_f1:+.4f})")
    print(f"Full Translation:              F1 = {full_f1:.4f}  (diff: {full_f1-baseline_f1:+.4f})")
    print(f"Hybrid (cutoff={CUTOFF_LAYER}):           F1 = {hybrid_f1:.4f}  (diff: {hybrid_f1-baseline_f1:+.4f})")
    print("="*80)
    
    if CUTOFF_LAYER == 0 and abs(hybrid_f1 - native_cache_f1) < 0.001:
        print("✅ Validation: Hybrid cutoff=0 matches Native Cache!")
    
    if abs(native_cache_f1 - baseline_f1) < 0.05:
        print("✅ Two-pass cache works well!")
    else:
        print(f"⚠️  Two-pass cache has {abs(native_cache_f1-baseline_f1):.3f} F1 gap vs baseline")

if __name__ == "__main__":
    main()
