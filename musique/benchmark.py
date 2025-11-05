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
    print("Usage: python test_hybrid_speedup_simple.py <qwen|mistral> <cutoff_layer>")
    print("  cutoff_layer: layer at which to switch from translation to recomputation (e.g., 8)")
    sys.exit(1)

if sys.argv[1] not in ['qwen', 'mistral']:
    print("First argument must be 'qwen' or 'mistral'")
    sys.exit(1)

try:
    CUTOFF_LAYER = int(sys.argv[2])
except:
    print("Second argument must be an integer (e.g., 8)")
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

def partial_forward_model_b(model_b, input_ids, start_layer, end_layer, 
                            past_key_values, attention_mask, position_ids):
    """
    Forward pass through Model B from start_layer to end_layer.
    
    For layers 0 to start_layer-1: Uses past_key_values (translated KV)
    For layers start_layer to end_layer-1: Computes natively
    
    Args:
        model_b: The Model B instance
        input_ids: Input token IDs [batch, seq_len]
        start_layer: First layer to compute (0 to start_layer-1 use past_key_values)
        end_layer: Last layer to compute (exclusive)
        past_key_values: KV cache for layers 0 to start_layer-1
        attention_mask: Attention mask
        position_ids: Position IDs
    
    Returns:
        new_kv_cache: List of (key, value) tuples for layers start_layer to end_layer-1
    """
    # Start from Model B's embeddings
    hidden_states = model_b.model.embed_tokens(input_ids)
    
    # We need to go through ALL layers, but:
    # - Layers 0 to start_layer-1: Use past_key_values, don't store output
    # - Layers start_layer to end_layer-1: Compute and store KV cache
    
    new_kv_cache = []
    
    for layer_idx in range(end_layer):
        layer = model_b.model.layers[layer_idx]
        
        if layer_idx < start_layer:
            # Use translated KV cache, don't save output
            past_kv_layer = past_key_values[layer_idx]
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv_layer,
                use_cache=False,  # Don't generate new KV
            )
            hidden_states = layer_outputs[0]
            
        else:
            # Compute natively and save KV cache
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=True,  # Generate new KV
            )
            hidden_states = layer_outputs[0]
            new_kv_cache.append(layer_outputs[1])  # (key, value) tuple
    
    return new_kv_cache

def true_hybrid_inference(context_only, tokenizer_a, tokenizer_b, 
                         model_a, model_b, translators_k, translators_v,
                         cutoff_layer, device, max_length):
    """
    True hybrid inference with actual speedup:
    1. Model A generates KV cache
    2. Translate KV for layers 0 to cutoff-1
    3. Feed tokens + translated KV to Model B
    4. Model B computes layers 0 to cutoff-1 using translated KV (cheap - just attention)
    5. Model B computes layers cutoff to NUM_LAYERS natively (generates KV)
    6. Combine translated + native KV cache
    """
    
    timings = {}
    
    # Step 1: Model A forward pass
    t0 = time.time()
    source_keys, source_vals = generate_kv_cache_model_a(
        context_only, tokenizer_a, model_a, device, max_length
    )
    timings['model_a_forward'] = time.time() - t0
    
    # Step 2: Translate KV cache for layers 0 to cutoff-1
    t0 = time.time()
    translated_keys = []
    translated_vals = []
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            for i in range(cutoff_layer):
                trans_k = translators_k[i](source_keys[i])
                trans_v = translators_v[i](source_vals[i])
                translated_keys.append(trans_k)
                translated_vals.append(trans_v)
    
    timings['translate_kv'] = time.time() - t0
    
    # Step 3: Tokenize context for Model B
    inputs_b = tokenizer_b([context_only], return_tensors="pt", truncation=True,
                          max_length=max_length).to(device)
    
    input_ids = inputs_b.input_ids
    seq_len = input_ids.shape[1]
    
    # Create attention mask and position IDs
    attention_mask = torch.ones((1, seq_len), device=device, dtype=torch.long)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    
    # Step 4: Prepare past_key_values with translated KV
    past_kv_translated = [(translated_keys[i], translated_vals[i]) for i in range(cutoff_layer)]
    
    # Step 5: Partial forward pass through Model B
    # Layers 0 to cutoff-1: Use translated KV (fast)
    # Layers cutoff to NUM_LAYERS: Compute natively (slow)
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            native_kv_late = partial_forward_model_b(
                model_b,
                input_ids,
                start_layer=cutoff_layer,
                end_layer=NUM_LAYERS,
                past_key_values=past_kv_translated,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
    
    timings['model_b_partial'] = time.time() - t0
    
    # Step 6: Combine translated (early) + native (late) KV cache
    hybrid_cache = past_kv_translated + native_kv_late
    
    del inputs_b, source_keys, source_vals
    
    return tuple(hybrid_cache), timings

def baseline_full_prefill(context_only, tokenizer_b, model_b, device, max_length):
    """Baseline: Full prefill on Model B."""
    t0 = time.time()
    
    inputs_b = tokenizer_b([context_only], return_tensors="pt", truncation=True,
                          max_length=max_length).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            outputs_b = model_b(**inputs_b, use_cache=True)
            baseline_cache = outputs_b.past_key_values
    
    prefill_time = time.time() - t0
    
    del inputs_b, outputs_b
    
    return baseline_cache, prefill_time

def evaluate_true_hybrid(model_a, model_b, tokenizer_a, tokenizer_b,
                        translators_k, translators_v,
                        cutoff_layer, device, eval_dataset):
    """Evaluate true hybrid approach with timing breakdown."""
    print("\n" + "="*80)
    print(f"TRUE HYBRID EVALUATION (Cutoff={cutoff_layer})")
    print("="*80)
    
    model_a.eval()
    model_b.eval()
    translators_k.eval()
    translators_v.eval()
    
    total_f1 = 0
    total_timings = {
        'model_a_forward': 0,
        'translate_kv': 0,
        'model_b_partial': 0,
        'generation': 0,
        'total': 0
    }
    num_evaluated = 0
    
    for i in tqdm(range(min(NUM_EVAL_SAMPLES, len(eval_dataset))), desc="Evaluating True Hybrid"):
        example = eval_dataset[i]
        full_prompt, context_only, question, ground_truth_answer = format_musique_prompt(example)
        
        if not ground_truth_answer:
            continue
        
        try:
            start_time = time.time()
            
            # True hybrid inference
            hybrid_cache, timings = true_hybrid_inference(
                context_only, tokenizer_a, tokenizer_b,
                model_a, model_b,
                translators_k, translators_v,
                cutoff_layer, device, MAX_CTX_TOKENS
            )
            
            context_len = hybrid_cache[0][0].shape[2]
            
            # Continue with " Answer:"
            answer_prompt = " Answer:"
            q_inputs = tokenizer_b(answer_prompt, return_tensors="pt").to(device)
            question_len = q_inputs.input_ids.shape[1]
            
            # Create proper attention mask and cache_position
            attention_mask = torch.ones(1, context_len + question_len, device=device, dtype=torch.long)
            cache_position = torch.arange(context_len, context_len + question_len, device=device, dtype=torch.long)
            
            t_gen = time.time()
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
            timings['generation'] = time.time() - t_gen
            
            end_time = time.time()
            timings['total'] = end_time - start_time
            
            response = tokenizer_b.decode(generated[0][question_len:], skip_special_tokens=True).strip()
            cleaned_response = response.split('\n')[0].strip()
            for ending in ['.', '?', '!']:
                if ending in cleaned_response:
                    cleaned_response = cleaned_response.split(ending)[0].strip()
                    break
            
            f1 = calculate_f1_score(cleaned_response, ground_truth_answer)
            total_f1 += f1
            num_evaluated += 1
            
            # Accumulate timings
            for key in total_timings:
                total_timings[key] += timings[key]
            
            if i < 2:
                print(f"\n--- Example {i+1} ---")
                print(f"Question: {question}")
                print(f"Ground Truth: {ground_truth_answer}")
                print(f"Prediction: {cleaned_response}")
                print(f"F1: {f1:.4f}")
                print(f"Timing breakdown:")
                print(f"  Model A forward:    {timings['model_a_forward']:.3f}s")
                print(f"  Translate KV:       {timings['translate_kv']:.3f}s")
                print(f"  Model B partial:    {timings['model_b_partial']:.3f}s")
                print(f"  Generation:         {timings['generation']:.3f}s")
                print(f"  Total:              {timings['total']:.3f}s")
        
        except Exception as e:
            print(f"\nExample {i+1} failed: {e}")
            traceback.print_exc()
    
    avg_f1 = total_f1 / num_evaluated if num_evaluated > 0 else 0
    avg_timings = {k: v / num_evaluated for k, v in total_timings.items()}
    
    return avg_f1, avg_timings, num_evaluated

def evaluate_baseline(model_b, tokenizer_b, device, eval_dataset):
    """Evaluate baseline full prefill."""
    print("\n" + "="*80)
    print(f"BASELINE EVALUATION: Full Prefill on Model B")
    print("="*80)
    
    model_b.eval()
    
    total_f1 = 0
    total_prefill_time = 0
    total_generation_time = 0
    total_time = 0
    num_evaluated = 0
    
    for i in tqdm(range(min(NUM_EVAL_SAMPLES, len(eval_dataset))), desc="Evaluating Baseline"):
        example = eval_dataset[i]
        full_prompt, context_only, question, ground_truth_answer = format_musique_prompt(example)
        
        if not ground_truth_answer:
            continue
        
        try:
            start_time = time.time()
            
            # Full prefill
            baseline_cache, prefill_time = baseline_full_prefill(
                context_only, tokenizer_b, model_b, device, MAX_CTX_TOKENS
            )
            
            context_len = baseline_cache[0][0].shape[2]
            
            # Continue with " Answer:"
            answer_prompt = " Answer:"
            q_inputs = tokenizer_b(answer_prompt, return_tensors="pt").to(device)
            question_len = q_inputs.input_ids.shape[1]
            
            attention_mask = torch.ones(1, context_len + question_len, device=device, dtype=torch.long)
            cache_position = torch.arange(context_len, context_len + question_len, device=device, dtype=torch.long)
            
            t_gen = time.time()
            with torch.no_grad():
                generated = model_b.generate(
                    input_ids=q_inputs.input_ids,
                    attention_mask=attention_mask,
                    past_key_values=baseline_cache,
                    cache_position=cache_position,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer_b.eos_token_id
                )
            generation_time = time.time() - t_gen
            
            end_time = time.time()
            
            response = tokenizer_b.decode(generated[0][question_len:], skip_special_tokens=True).strip()
            cleaned_response = response.split('\n')[0].strip()
            for ending in ['.', '?', '!']:
                if ending in cleaned_response:
                    cleaned_response = cleaned_response.split(ending)[0].strip()
                    break
            
            f1 = calculate_f1_score(cleaned_response, ground_truth_answer)
            total_f1 += f1
            total_prefill_time += prefill_time
            total_generation_time += generation_time
            total_time += (end_time - start_time)
            num_evaluated += 1
            
            if i < 2:
                print(f"\n--- Example {i+1} ---")
                print(f"Question: {question}")
                print(f"Ground Truth: {ground_truth_answer}")
                print(f"Prediction: {cleaned_response}")
                print(f"F1: {f1:.4f}")
                print(f"Prefill time: {prefill_time:.3f}s")
                print(f"Generation time: {generation_time:.3f}s")
                print(f"Total time: {end_time - start_time:.3f}s")
        
        except Exception as e:
            print(f"\nExample {i+1} failed: {e}")
            traceback.print_exc()
    
    avg_f1 = total_f1 / num_evaluated if num_evaluated > 0 else 0
    avg_prefill_time = total_prefill_time / num_evaluated if num_evaluated > 0 else 0
    avg_generation_time = total_generation_time / num_evaluated if num_evaluated > 0 else 0
    avg_total_time = total_time / num_evaluated if num_evaluated > 0 else 0
    
    return avg_f1, avg_prefill_time, avg_generation_time, avg_total_time, num_evaluated

def main():
    print(f"Device: {DEVICE}")
    print(f"Evaluating True Hybrid Approach with Speedup (No Hidden State Translation)")
    print(f"  Cutoff layer: {CUTOFF_LAYER}")
    print(f"  Early layers (translated KV): 0-{CUTOFF_LAYER-1} ({CUTOFF_LAYER} layers)")
    print(f"  Late layers (native compute): {CUTOFF_LAYER}-{NUM_LAYERS-1} ({NUM_LAYERS-CUTOFF_LAYER} layers)")
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
    
    # Load dataset
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
    
    # Create KV translators
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
    
    # Load KV translator weights
    checkpoint = torch.load(LOAD_PATH, map_location=DEVICE)
    translators_k.load_state_dict(checkpoint['translators_k_state_dict'])
    translators_v.load_state_dict(checkpoint['translators_v_state_dict'])
    translators_k.to(DEVICE).eval()
    translators_v.to(DEVICE).eval()
    print("✅ KV Translators loaded")
    
    del dummy_sk, dummy_sv, dummy_tk, dummy_tv
    
    # Run evaluations
    print("\n" + "="*80)
    print("RUNNING EVALUATIONS")
    print("="*80)
    
    print("\n[1/2] Evaluating Baseline (Full Prefill)...")
    baseline_f1, baseline_prefill, baseline_gen, baseline_total, baseline_count = evaluate_baseline(
        model_b, tokenizer_b, DEVICE, eval_dataset
    )
    
    print("\n[2/2] Evaluating True Hybrid Approach...")
    hybrid_f1, hybrid_timings, hybrid_count = evaluate_true_hybrid(
        model_a, model_b, tokenizer_a, tokenizer_b,
        translators_k, translators_v,
        CUTOFF_LAYER, DEVICE, eval_dataset
    )
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS: True Hybrid vs Baseline")
    print("="*80)
    print(f"Evaluation samples: {baseline_count}\n")
    
    print("QUALITY:")
    print(f"  Baseline F1:     {baseline_f1:.4f}")
    print(f"  Hybrid F1:       {hybrid_f1:.4f}")
    print(f"  Difference:      {hybrid_f1 - baseline_f1:+.4f}")
    
    print("\nTIMING (Prefill only):")
    hybrid_prefill_total = (hybrid_timings['model_a_forward'] + 
                           hybrid_timings['translate_kv'] + 
                           hybrid_timings['model_b_partial'])
    
    print(f"  Baseline prefill:         {baseline_prefill:.3f}s")
    print(f"  Hybrid prefill total:     {hybrid_prefill_total:.3f}s")
    print(f"    - Model A forward:      {hybrid_timings['model_a_forward']:.3f}s")
    print(f"    - Translate KV:         {hybrid_timings['translate_kv']:.3f}s")
    print(f"    - Model B partial:      {hybrid_timings['model_b_partial']:.3f}s")
    
    speedup_prefill = baseline_prefill / hybrid_prefill_total if hybrid_prefill_total > 0 else 0
    print(f"\n  PREFILL SPEEDUP:          {speedup_prefill:.2f}x")
    
    print("\nTIMING (End-to-end):")
    print(f"  Baseline total:           {baseline_total:.3f}s")
    print(f"  Hybrid total:             {hybrid_timings['total']:.3f}s")
    
    speedup_total = baseline_total / hybrid_timings['total'] if hybrid_timings['total'] > 0 else 0
    print(f"\n  END-TO-END SPEEDUP:       {speedup_total:.2f}x")
    
    print("\nCOMPUTATION SAVINGS:")
    layers_saved = CUTOFF_LAYER
    layers_computed = NUM_LAYERS - CUTOFF_LAYER
    print(f"  Model B layers saved:     {layers_saved}/{NUM_LAYERS} ({100*layers_saved/NUM_LAYERS:.1f}%)")
    print(f"  Model B layers computed:  {layers_computed}/{NUM_LAYERS} ({100*layers_computed/NUM_LAYERS:.1f}%)")
    
    theoretical_speedup = NUM_LAYERS / layers_computed
    print(f"  Theoretical speedup:      {theoretical_speedup:.2f}x (ignoring Model A cost)")
    
    print("="*80)
    
    if speedup_prefill > 1.0:
        print(f"\n✅ SUCCESS: Achieved {speedup_prefill:.2f}x prefill speedup!")
        if abs(hybrid_f1 - baseline_f1) < 0.05:
            print(f"✅ Quality preserved: F1 difference is only {abs(hybrid_f1-baseline_f1):.4f}")
        else:
            print(f"⚠️  Quality gap: {abs(hybrid_f1-baseline_f1):.4f} F1 difference")
    else:
        print(f"\n⚠️  No speedup achieved. Hybrid is slower by {1/speedup_prefill:.2f}x")
        print(f"   This may be due to Model A overhead or translation cost.")

if __name__ == "__main__":
    main()
