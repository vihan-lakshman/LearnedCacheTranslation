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
from tqdm import tqdm
import os # Added for path handling
import gc

logging.set_verbosity_error()

###########################
# Configuration
###########################
# UPDATED: Added 'gemma' option
if len(sys.argv) < 2 or sys.argv[1] not in ['qwen', 'mistral', 'gemma']:
    print("Usage: python test_musique.py <qwen|mistral|gemma>")
    sys.exit(1)

if sys.argv[1] == 'qwen':
    MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
elif sys.argv[1] == 'mistral':
    MODEL_A = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"
else: # gemma
    MODEL_A = "google/gemma-2-9b"
    MODEL_B = "google/gemma-2-9b-it"


LOAD_PATH = "kv_translators_musique.pth" # Updated path to match training script
# UPDATED: NUM_LAYERS logic
if "Qwen" in MODEL_A:
    NUM_LAYERS = 28
elif "Mistral" in MODEL_A:
    NUM_LAYERS = 32
else:
    NUM_LAYERS = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_CTX_TOKENS = 512
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
    # (FastKVTranslator remains the same)
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
    # Generate the initial cache for the first part of the prompt
    inputs_a = tokenizer_a(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            out_a = model_a(**inputs_a, use_cache=True)
        kv_a = out_a.past_key_values
    source_keys = [kv_a[i][0] for i in range(len(kv_a))]; source_vals = [kv_a[i][1] for i in range(len(kv_a))]
    del out_a, kv_a, inputs_a
    return source_keys, source_vals

def load_models_and_translators():
    print("--- Loading Base Models and Tokenizers ---")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B)
    # Ensure trust_remote_code is set to False by default unless absolutely necessary
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16).to(DEVICE)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16).to(DEVICE)

    if tokenizer_a.pad_token is None: tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token
    model_a.eval(); model_b.eval()

    print("\n--- Determining Translator Dimensions ---")
    # Use MAX_CTX_TOKENS for more robust dimension check
    dummy_sk, dummy_sv = generate_kv_cache(["dummy prompt for size check"], tokenizer_a, model_a, DEVICE, MAX_CTX_TOKENS)
    with torch.no_grad():
        dummy_inputs_b = tokenizer_b(["dummy prompt for size check"], return_tensors="pt", padding=True, max_length=MAX_CTX_TOKENS).to(DEVICE)
        dummy_out_b = model_b(**dummy_inputs_b, use_cache=True)
        dummy_kv_b = dummy_out_b.past_key_values
    dummy_tk = [dummy_kv_b[i][0] for i in range(len(dummy_kv_b))]; dummy_tv = [dummy_kv_b[i][1] for i in range(len(dummy_kv_b))]
    del dummy_out_b, dummy_kv_b, dummy_inputs_b
    gc.collect(); torch.cuda.empty_cache() # Clean up dummy tensors

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
    
    # Updated: Robust cleanup of keys potentially added by torch.compile in training
    clean_k_state_dict = {k.replace('_orig_mod.', ''): v for k, v in k_state_dict.items()}
    clean_v_state_dict = {k.replace('_orig_mod.', ''): v for k, v in v_state_dict.items()}
    
    translators_k.load_state_dict(clean_k_state_dict); translators_v.load_state_dict(clean_v_state_dict)
    translators_k.to(DEVICE).eval(); translators_v.to(DEVICE).eval()
    print("✅ Translators loaded and in evaluation mode.")

    return model_a, model_b, tokenizer_a, tokenizer_b, translators_k, translators_v

def run_test():
    # Load models, clean up dummy tensors
    model_a, model_b, tokenizer_a, tokenizer_b, translators_k, translators_v = load_models_and_translators()
    gc.collect(); torch.cuda.empty_cache()

    print("\n--- Loading MuSiQue Validation Dataset ---")
    dataset = load_dataset("dgslibisey/MuSiQue", split="validation")
    random.seed(SEED)
    test_indices = random.sample(range(len(dataset)), NUM_TESTS)
    print(f"Loaded {len(dataset)} examples, testing {NUM_TESTS} random samples.")

    total_f1_score = 0
    for i, example_idx in enumerate(tqdm(test_indices, desc="Running Multi-Hop Evaluation")):
        example = dataset[example_idx]
        
        try:
            # Step 1: First hop on Model A using the context + first question part
            context_text = " ".join([p["paragraph_text"] for p in example["paragraphs"]])
            # The prompt includes the context and the first hop of the decomposed question
            first_hop_prompt = f"{context_text} {example['question_decomposition'][0]}"
            
            # Use generate_kv_cache to get the KV cache from Model A
            sk, sv = generate_kv_cache([first_hop_prompt], tokenizer_a, model_a, DEVICE, MAX_CTX_TOKENS)
            
            # Step 2: Translate the KV cache
            with torch.no_grad():
                with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
                    translated_k = [translators_k[j](sk[j]) for j in range(NUM_LAYERS)]
                    translated_v = [translators_v[j](sv[j]) for j in range(NUM_LAYERS)]
            
            translated_cache = tuple(zip(translated_k, translated_v))
            del sk, sv # Free Model A's cache immediately
            gc.collect(); torch.cuda.empty_cache()

            # Step 3: Prepare the final question for Model B
            full_question = example['question']
            
            # Prepare the final question prompt for Model B (only the instruction/question)
            # The critical part is to ONLY tokenize the instruction/question text, NOT the full context,
            # as the context is already represented by the translated_cache.
            
            # Use a simple instruction template for Model B
            question_only_prompt = f"Answer the following question based on the context provided in your memory (KV cache):\n\nQuestion: {full_question}\n\nAnswer:"
            
            q_inputs = tokenizer_b(question_only_prompt, return_tensors="pt").to(DEVICE)
            
            if q_inputs.input_ids.shape[1] == 0:
                tqdm.write(f"Skipping example {example['id']} due to empty tokenized question.")
                continue

            # Calculate generation parameters based on the combined length
            context_len = translated_cache[0][0].shape[2] # Length of the sequence in the cache
            question_len = q_inputs.input_ids.shape[1] # Length of the sequence of the new input (the final question)
            
            # Create attention mask covering the cache + the new question tokens
            attention_mask = torch.ones(1, context_len + question_len, device=DEVICE)
            # The cache_position must start at the length of the cache
            cache_position = torch.arange(context_len, context_len + question_len, device=DEVICE)

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

            # Step 4: Decode and Evaluate
            # The response starts *after* the prompt we fed to Model B
            response_start_index = q_inputs.input_ids.shape[1]
            full_decoded_response = tokenizer_b.decode(generated[0], skip_special_tokens=True)
            
            # Extract only the generated part
            response = tokenizer_b.decode(generated[0, response_start_index:], skip_special_tokens=True).strip()
            
            # Clean the response (take first sentence/line)
            cleaned_response = response.split('\n')[0].split('.')[0] if '.' in response.split('\n')[0] else response.split('\n')[0]

            ground_truth_answer = example['answer'].strip()
            
            f1 = calculate_f1_score(cleaned_response, ground_truth_answer)
            total_f1_score += f1
            
            tqdm.write(f"\nExample ID: {example['id']}")
            tqdm.write(f"Question: {full_question}")
            tqdm.write(f"Model A First Hop Input (Initial Context): {first_hop_prompt[:100]}...")
            # tqdm.write(f"Model B Final Input: {question_only_prompt.split('\n')[0]}...")
            tqdm.write(f"Model B Output: {cleaned_response}")
            tqdm.write(f"Ground Truth: {ground_truth_answer}")
            tqdm.write(f"F1 Score: {f1:.4f}")
            gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            tqdm.write(f"\nAn error occurred on example ID {example.get('id', 'N/A')}: {e}")
            traceback.print_exc()

    average_f1 = total_f1_score / NUM_TESTS
    print(f"\n--- Final Results ---")
    print(f"Average F1 Score across {NUM_TESTS} samples: {average_f1:.4f}")

if __name__ == "__main__":
    run_test()
