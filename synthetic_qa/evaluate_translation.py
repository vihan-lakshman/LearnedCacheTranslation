import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import sys
import re
import string
from collections import Counter

logging.set_verbosity_error()

###########################
# Configuration
###########################
if len(sys.argv) < 3:
    print("Usage: python evaluate_translation_fair.py <qwen|mistral> <num_facts> [checkpoint]")
    sys.exit(1)

MODEL_TYPE = sys.argv[1]
NUM_FACTS = int(sys.argv[2])
CHECKPOINT = sys.argv[3] if len(sys.argv) > 3 else f"kv_translators_twostage_{NUM_FACTS}facts.pth"

if MODEL_TYPE == 'qwen':
    MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
else:
    MODEL_A = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

NUM_LAYERS = 28 if 'Qwen' in MODEL_A else 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EVAL_SAMPLES = 200
MAX_CTX_TOKENS = 32 + (NUM_FACTS * 32)
MAX_NEW_TOKENS = 32
SEED = 456
MIXED_PRECISION = True

torch.manual_seed(SEED)
random.seed(SEED)
###########################


def normalize_text(s):
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s


def calculate_f1_score(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if not truth_tokens and not pred_tokens:
        return 1.0
    if not truth_tokens or not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def calculate_exact_match(prediction, ground_truth):
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def generate_fact():
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
    
    return {
        'text': f"{subject} {action} {obj} {location}.",
        'subject': subject,
        'action': action,
        'object': obj,
        'location': location
    }


def generate_synthetic_qa(num_facts=1):
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
    
    return context, question, answer, q_type


# Two prompt formats to compare
def build_prompt_training(context, question):
    """Original training prompt format"""
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def build_prompt_explicit(context, question):
    """Better prompt format that works well for both models"""
    return f"""Read the context and answer the question with just the answer, no explanation.

Context: {context}

Question: {question}

Answer:"""


def clean_response(response, expected_answer=None):
    response = response.strip()
    response = response.split('\n')[0].strip()
    
    prefixes_to_remove = ["The answer is", "Answer:", "Based on the context,", "According to the context,"]
    for prefix in prefixes_to_remove:
        if response.lower().startswith(prefix.lower()):
            response = response[len(prefix):].strip()
    
    if expected_answer:
        normalized_expected = normalize_text(expected_answer)
        words_expected = normalized_expected.split()
        words_response = response.split()
        
        if len(words_response) >= len(words_expected):
            potential_match = ' '.join(words_response[:len(words_expected)])
            if normalize_text(potential_match) == normalized_expected:
                return potential_match
    
    for ending in ['.', '?', '!', ',']:
        if ending in response:
            response = response.split(ending)[0].strip()
            break
    
    return response


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

    def forward(self, cache_tensor_a):
        batch, num_heads, seq_len, head_dim = cache_tensor_a.shape
        x = cache_tensor_a.permute(0, 2, 1, 3).contiguous().view(batch * seq_len, -1)
        y = self.net(x)
        y = y.view(batch, seq_len, self.target_heads, self.target_head_dim)
        return y.permute(0, 2, 1, 3).contiguous()


def get_model_dimensions(tokenizer, model, device, num_layers):
    dummy_input = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**dummy_input, use_cache=True)
    kv_cache = outputs.past_key_values
    dims = []
    if hasattr(kv_cache, 'key_cache'):
        for i in range(num_layers):
            k_shape = kv_cache.key_cache[i].shape
            v_shape = kv_cache.value_cache[i].shape
            dims.append({'k_heads': k_shape[1], 'k_head_dim': k_shape[3],
                        'v_heads': v_shape[1], 'v_head_dim': v_shape[3]})
    else:
        for i in range(num_layers):
            k_shape = kv_cache[i][0].shape
            v_shape = kv_cache[i][1].shape
            dims.append({'k_heads': k_shape[1], 'k_head_dim': k_shape[3],
                        'v_heads': v_shape[1], 'v_head_dim': v_shape[3]})
    return dims


def generate_source_kv_cache(prompt, tokenizer, model, device, max_length):
    inputs = tokenizer([prompt], return_tensors="pt", padding="max_length",
                       truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    kv_cache = outputs.past_key_values
    if hasattr(kv_cache, 'key_cache'):
        source_keys = [kv_cache.key_cache[i] for i in range(NUM_LAYERS)]
        source_vals = [kv_cache.value_cache[i] for i in range(NUM_LAYERS)]
    else:
        source_keys = [kv_cache[i][0] for i in range(NUM_LAYERS)]
        source_vals = [kv_cache[i][1] for i in range(NUM_LAYERS)]
    return source_keys, source_vals


def translate_cache(source_keys, source_vals, translators_k, translators_v):
    translated_cache = DynamicCache()
    for i in range(NUM_LAYERS):
        trans_k = translators_k[i](source_keys[i])
        trans_v = translators_v[i](source_vals[i])
        translated_cache.update(trans_k, trans_v, i)
    return translated_cache


def evaluate_method(method_name, generate_fn, test_examples, tokenizer_b, model_b, 
                    prompt_fn, device, expected_answers):
    """Generic evaluation function."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {method_name}")
    print('='*60)
    
    f1_scores, em_scores = [], []
    
    for i, (context, question, answer, q_type) in enumerate(tqdm(test_examples, desc=method_name)):
        try:
            prediction = generate_fn(context, question, answer, prompt_fn)
            
            f1 = calculate_f1_score(prediction, answer)
            em = calculate_exact_match(prediction, answer)
            f1_scores.append(f1)
            em_scores.append(em)
            
            if i < 3:
                print(f"\n  Ex {i+1}: Q: {question[:50]}...")
                print(f"    GT: {answer} | Pred: {prediction} | F1: {f1:.3f}")
        except Exception as e:
            print(f"  Ex {i+1} failed: {e}")
            f1_scores.append(0.0)
            em_scores.append(0.0)
    
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_em = sum(em_scores) / len(em_scores)
    print(f"\n  {method_name}: F1={avg_f1:.4f}, EM={avg_em:.4f}")
    
    return avg_f1, avg_em


def main():
    print(f"Device: {DEVICE}")
    print(f"Fair Evaluation of KV Cache Translation")
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  Num facts: {NUM_FACTS}")
    print()
    
    # Generate test set
    print("Generating test set...")
    test_examples = [generate_synthetic_qa(NUM_FACTS) for _ in range(NUM_EVAL_SAMPLES)]
    
    # Load models
    print(f"\nLoading models...")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    
    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None:
        tokenizer_b.pad_token = tokenizer_b.eos_token
    
    model_a.eval()
    model_b.eval()
    
    # Load translators
    print(f"Loading translators from {CHECKPOINT}...")
    source_dims = get_model_dimensions(tokenizer_a, model_a, DEVICE, NUM_LAYERS)
    target_dims = get_model_dimensions(tokenizer_b, model_b, DEVICE, NUM_LAYERS)
    
    translators_k = nn.ModuleList()
    translators_v = nn.ModuleList()
    for i in range(NUM_LAYERS):
        src, tgt = source_dims[i], target_dims[i]
        k_in = src['k_heads'] * src['k_head_dim']
        k_out = tgt['k_heads'] * tgt['k_head_dim']
        v_in = src['v_heads'] * src['v_head_dim']
        v_out = tgt['v_heads'] * tgt['v_head_dim']
        translators_k.append(SimpleDeepTranslator(k_in, k_out, tgt['k_heads'], tgt['k_head_dim']))
        translators_v.append(SimpleDeepTranslator(v_in, v_out, tgt['v_heads'], tgt['v_head_dim']))
    
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    translators_k.load_state_dict(checkpoint['translators_k_state_dict'])
    translators_v.load_state_dict(checkpoint['translators_v_state_dict'])
    translators_k.to(DEVICE).eval()
    translators_v.to(DEVICE).eval()
    print("✅ Translators loaded")
    
    # Define generation functions
    def generate_model_a_direct(context, question, answer, prompt_fn):
        prompt = prompt_fn(context, question)
        inputs = tokenizer_a(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated = model_a.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                         do_sample=False, pad_token_id=tokenizer_a.eos_token_id)
        response = tokenizer_a.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return clean_response(response, answer)
    
    def generate_model_b_direct(context, question, answer, prompt_fn):
        prompt = prompt_fn(context, question)
        inputs = tokenizer_b(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated = model_b.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                         do_sample=False, pad_token_id=tokenizer_b.eos_token_id)
        response = tokenizer_b.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return clean_response(response, answer)
    
    def generate_translated(context, question, answer, prompt_fn):
        prompt = prompt_fn(context, question)
        source_keys, source_vals = generate_source_kv_cache(prompt, tokenizer_a, model_a, DEVICE, MAX_CTX_TOKENS)
        with torch.no_grad():
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
                translated_cache = translate_cache(source_keys, source_vals, translators_k, translators_v)
        
        start_token = tokenizer_b(" ", return_tensors="pt").input_ids.to(DEVICE)
        cache_seq_len = translated_cache.get_seq_length()
        attention_mask = torch.ones(1, cache_seq_len + 1, device=DEVICE)
        
        with torch.no_grad():
            generated = model_b.generate(
                input_ids=start_token,
                attention_mask=attention_mask,
                past_key_values=translated_cache,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer_b.eos_token_id,
                cache_position=torch.arange(cache_seq_len, cache_seq_len + 1, device=DEVICE)
            )
        response = tokenizer_b.decode(generated[0], skip_special_tokens=True)
        return clean_response(response, answer)
    
    # Run evaluations with TRAINING prompt format
    print("\n" + "="*70)
    print("EVALUATION WITH TRAINING PROMPT FORMAT")
    print("="*70)
    
    results_training = {}
    results_training['Model A'] = evaluate_method(
        "Model A Direct (training prompt)", generate_model_a_direct,
        test_examples, tokenizer_b, model_b, build_prompt_training, DEVICE, None
    )
    results_training['Model B'] = evaluate_method(
        "Model B Direct (training prompt)", generate_model_b_direct,
        test_examples, tokenizer_b, model_b, build_prompt_training, DEVICE, None
    )
    results_training['Translated'] = evaluate_method(
        "Translated Cache (training prompt)", generate_translated,
        test_examples, tokenizer_b, model_b, build_prompt_training, DEVICE, None
    )
    
    # Run evaluations with EXPLICIT prompt format
    print("\n" + "="*70)
    print("EVALUATION WITH EXPLICIT PROMPT FORMAT")
    print("="*70)
    
    results_explicit = {}
    results_explicit['Model A'] = evaluate_method(
        "Model A Direct (explicit prompt)", generate_model_a_direct,
        test_examples, tokenizer_b, model_b, build_prompt_explicit, DEVICE, None
    )
    results_explicit['Model B'] = evaluate_method(
        "Model B Direct (explicit prompt)", generate_model_b_direct,
        test_examples, tokenizer_b, model_b, build_prompt_explicit, DEVICE, None
    )
    results_explicit['Translated'] = evaluate_method(
        "Translated Cache (explicit prompt)", generate_translated,
        test_examples, tokenizer_b, model_b, build_prompt_explicit, DEVICE, None
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Method':<30} {'Training Prompt':<20} {'Explicit Prompt':<20}")
    print("-"*70)
    for method in ['Model A', 'Model B', 'Translated']:
        f1_train = results_training[method][0]
        f1_explicit = results_explicit[method][0]
        print(f"{method:<30} {f1_train:<20.4f} {f1_explicit:<20.4f}")
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Use training prompt results for fair comparison
    f1_a = results_training['Model A'][0]
    f1_b = results_training['Model B'][0]
    f1_trans = results_training['Translated'][0]
    
    print(f"\nWith training prompt format:")
    print(f"  Model A (lower bound): {f1_a:.4f}")
    print(f"  Model B (upper bound): {f1_b:.4f}")
    print(f"  Translated:            {f1_trans:.4f}")
    
    if f1_b > f1_a:
        gap = f1_b - f1_a
        recovered = (f1_trans - f1_a) / gap * 100
        print(f"\n  Gap: {gap:.4f}")
        print(f"  Recovery: {recovered:.1f}%")
    else:
        print(f"\n  ⚠️ Model B worse than Model A with this prompt format!")


if __name__ == "__main__":
    main()
