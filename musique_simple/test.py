import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
import traceback
import sys
import re
import string
from collections import Counter
from datasets import load_dataset
import gc

logging.set_verbosity_error()

###########################
# Configuration
# Usage: python test_musique_translation.py <qwen|mistral>
###########################
if len(sys.argv) < 2 or sys.argv[1] not in ['qwen', 'mistral']:
    print("Usage: python test_musique_translation.py <qwen|mistral>")
    sys.exit(1)

if sys.argv[1] == 'qwen':
    MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
else:
    MODEL_A = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

LOAD_PATH = "kv_translators_musique.pth" # Updated load path
NUM_LAYERS = 28 if 'Qwen' in MODEL_A else 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Settings for MuSiQue task
MAX_CTX_TOKENS = 1024 # Contexts are longer
MAX_NEW_TOKENS = 64
NUM_TESTS = 50
MIXED_PRECISION = True
SEED = 42
###########################

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

class FastKVTranslator(nn.Module):
    def __init__(self, input_size, output_size, target_heads, target_head_dim):
        super().__init__()
        self.target_heads, self.target_head_dim = target_heads, target_head_dim
        hidden_size = (input_size + output_size) // 2
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size, bias=False), nn.GELU(), nn.Linear(hidden_size, output_size, bias=False))

    def forward(self, cache_tensor_a):
        batch, _, seq_len, _ = cache_tensor_a.shape
        x = cache_tensor_a.permute(0, 2, 1, 3).contiguous().view(batch * seq_len, -1)
        y = self.net(x).view(batch, seq_len, self.target_heads, self.target_head_dim)
        return y.permute(0, 2, 1, 3).contiguous()

def generate_kv_cache(prompts, tokenizer_a, model_a, device, max_length):
    # Generates KV cache for the contextual prompt using Model A
    inputs_a = tokenizer_a(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            out_a = model_a(**inputs_a, use_cache=True)
        kv_a = out_a.past_key_values
    source_keys = [kv_a[i][0] for i in range(NUM_LAYERS)]; source_vals = [kv_a[i][1] for i in range(NUM_LAYERS)]
    del out_a, kv_a, inputs_a
    return source_keys, source_vals

def run_test():
    print(f"Device: {DEVICE}")
    print("--- Loading MuSiQue Dataset ---")
    
    # Load test split for evaluation
    try:
        dataset = load_dataset("dgslibisey/MuSiQue", split="validation")
    except Exception:
        # Fallback to train if test split is restricted
        print("Could not load 'test' split. Falling back to a subset of 'train'.")
        dataset = load_dataset("dgslibisey/MuSiQue", split="train").select(range(500))

    print(f"Loaded {len(dataset)} examples for testing.")

    print("\n--- Loading Base Models and Tokenizers ---")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)

    if tokenizer_a.pad_token is None: tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token
    model_a.eval(); model_b.eval()

    print("\n--- Determining Translator Dimensions ---")
    # Use a dummy contextual prompt for dimension determination
    dummy_context_prompt = "Context: This is a test paragraph.\n\nQuestion: What is the word?"
    dummy_sk, dummy_sv = generate_kv_cache([dummy_context_prompt], tokenizer_a, model_a, DEVICE, MAX_CTX_TOKENS)
    
    with torch.no_grad():
        dummy_inputs_b = tokenizer_b([dummy_context_prompt], return_tensors="pt", padding="max_length", max_length=MAX_CTX_TOKENS).to(DEVICE)
        dummy_out_b = model_b(**dummy_inputs_b, use_cache=True)
        dummy_kv_b = dummy_out_b.past_key_values
    dummy_tk = [dummy_kv_b[i][0] for i in range(NUM_LAYERS)]; dummy_tv = [dummy_kv_b[i][1] for i in range(NUM_LAYERS)]
    del dummy_out_b, dummy_kv_b, dummy_inputs_b; gc.collect(); torch.cuda.empty_cache()


    print("--- Creating Empty Translator Models ---")
    translators_k, translators_v = nn.ModuleList(), nn.ModuleList()
    for i in range(NUM_LAYERS):
        sk, sv, tk, tv = dummy_sk[i], dummy_sv[i], dummy_tk[i], dummy_tv[i]
        k_in_size = sk.shape[1] * sk.shape[3]; k_out_size = tk.shape[1] * tk.shape[3]
        v_in_size = sv.shape[1] * sv.shape[3]; v_out_size = tv.shape[1] * tv.shape[3]
        translators_k.append(FastKVTranslator(k_in_size, k_out_size, tk.shape[1], tk.shape[3]))
        translators_v.append(FastKVTranslator(v_in_size, v_out_size, tv.shape[1], tv.shape[3]))

    print(f"--- Loading Trained Weights from {LOAD_PATH} ---")
    checkpoint = torch.load(LOAD_PATH, map_location=DEVICE)
    k_state_dict = checkpoint['translators_k_state_dict']; v_state_dict = checkpoint['translators_v_state_dict']
    clean_k_state_dict = {k.replace('_orig_mod.', ''): v for k, v in k_state_dict.items()}
    clean_v_state_dict = {k.replace('_orig_mod.', ''): v for k, v in v_state_dict.items()}
    translators_k.load_state_dict(clean_k_state_dict); translators_v.load_state_dict(clean_v_state_dict)
    translators_k.to(DEVICE).eval(); translators_v.to(DEVICE).eval()
    print("✅ Translators loaded and in evaluation mode.")
    
    del dummy_sk, dummy_sv, dummy_tk, dummy_tv; gc.collect(); torch.cuda.empty_cache()


    print("\n--- Starting MuSiQue Translation Evaluation ---")
    total_f1_score = 0

    random.seed(SEED)
    test_indices = random.sample(range(len(dataset)), NUM_TESTS)

    for i, example_idx in enumerate(test_indices):
        example = dataset[example_idx]
        print(f"\n--- Test {i + 1}/{NUM_TESTS} ---")

        question_text = example['question_decomposition'][0]['question']
        ground_truth_answer = example['question_decomposition'][0]['answer']

        supporting_idx = example['question_decomposition'][0]['paragraph_support_idx']
        # --- MODIFICATION: Filter for only supporting paragraphs ---
        supporting_paragraphs = [
            p['paragraph_text']
            for p in example['paragraphs']
            if p['idx'] == supporting_idx
        ]

        context_paragraphs = "\n\n".join(supporting_paragraphs) 
        
        # 2. Context Prompt (used for KV Cache generation by Model A)
        context_prompt = f"Context:\n{context_paragraphs}\n\nQuestion: {question_text}"
        
        # 3. Question Prompt (used for decoding by Model B)
        # We only pass the question/instruction for the decoding step
        messages_b = [{"role": "user", "content": f"Answer the following question with just the direct answer: {question_text}"}]
        prompt_for_b = tokenizer_b.apply_chat_template(messages_b, tokenize=False, add_generation_prompt=True)
        q_inputs = tokenizer_b(prompt_for_b, return_tensors="pt").to(DEVICE)

        print(f"Question: '{question_text}'")
        print(f"Ground Truth Answer: '{ground_truth_answer}'")
        print(f"Source Context Length: {len(context_prompt.split()):,} words")


        try:
            # Step 1: Generate Source KV Cache from the FULL CONTEXT PROMPT using Model A
            sk, sv = generate_kv_cache([context_prompt], tokenizer_a, model_a, DEVICE, MAX_CTX_TOKENS)
            
            # Step 2: Translate the Source KV Cache to Target KV Cache format
            with torch.no_grad():
                with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
                    translated_k = [translators_k[i](sk[i]) for i in range(NUM_LAYERS)]
                    translated_v = [translators_v[i](sv[i]) for i in range(NUM_LAYERS)]
            translated_cache = tuple(zip(translated_k, translated_v))

            # Step 3: Configure Model B Generation
            context_len = translated_cache[0][0].shape[2] # Length of the translated context cache
            question_len = q_inputs.input_ids.shape[1]   # Length of the question prompt for Model B
            
            # Since the context is translated, the question prompt starts where the context ends.
            attention_mask = torch.ones(1, context_len + question_len, device=DEVICE)
            cache_position = torch.arange(context_len, context_len + question_len, device=DEVICE)

            # Step 4: Model B generates the answer using the translated context
            with torch.no_grad():
                generated = model_b.generate(
                    input_ids=q_inputs.input_ids, attention_mask=attention_mask,
                    past_key_values=translated_cache, cache_position=cache_position,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer_b.eos_token_id
                )

            # Step 5: Decode and calculate F1 score
            response_start_index = len(tokenizer_b.decode(q_inputs.input_ids[0], skip_special_tokens=True))
            full_decoded_response = tokenizer_b.decode(generated[0], skip_special_tokens=True)
            
            # Extract response part and clean
            response = full_decoded_response[response_start_index:].strip()
            cleaned_response = response.split('\n')[0].split('.')[0].strip()

            print(f"Model Response (cleaned): '{cleaned_response}'")

            f1 = calculate_f1_score(cleaned_response, ground_truth_answer)
            total_f1_score += f1
            print(f"F1 Score: {f1:.4f}")
        except Exception as e:
            print(f"Test failed with an error: {e}"); traceback.print_exc()

    average_f1 = total_f1_score / len(test_indices)
    print(f"\n--- Final Results ---")
    print(f"Average F1 Score (MuSiQue Translation) across {len(test_indices)} samples: {average_f1:.4f}")

if __name__ == "__main__":
    run_test()
