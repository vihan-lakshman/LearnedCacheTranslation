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
import os 
import gc

logging.set_verbosity_error()

MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_B = "Qwen/Qwen2.5-7B-Instruct"

LOAD_PATH = "kv_translators_musique_improved.pth"

NUM_LAYERS = 28
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_CTX_TOKENS = 2048
MAX_NEW_TOKENS = 64
NUM_TESTS = 50
MIXED_PRECISION = True
SEED = 42

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


def calculate_exact_match_score(prediction, ground_truth):
    em = 1 if ground_truth in prediction else 0
    return em


class FastKVTranslator(nn.Module):
    def __init__(self, input_size, output_size, target_heads, target_head_dim):
        super().__init__()
        self.target_heads, self.target_head_dim = target_heads, target_head_dim
        hidden_size = int((input_size + output_size) * 0.75) 
        
        self.net = nn.Sequential(
            # Layer 1: Input to Hidden
            nn.Linear(input_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size), 
            nn.GELU(),
            # Layer 2: Hidden to Hidden
            nn.Linear(hidden_size, hidden_size, bias=False), 
            nn.LayerNorm(hidden_size), 
            nn.GELU(),
            # Layer 3: Hidden to Output
            nn.Linear(hidden_size, output_size, bias=False) 
        )

    def forward(self, cache_tensor_a):
        batch, _, seq_len, _ = cache_tensor_a.shape
        # Flatten (Batch*SeqLen, InputDim)
        x = cache_tensor_a.permute(0, 2, 1, 3).contiguous().view(batch * seq_len, -1)
        # Apply network, reshape back to (Batch, SeqLen, Heads, HeadDim)
        y = self.net(x).view(batch, seq_len, self.target_heads, self.target_head_dim)
        # Permute back to standard KV cache format (Batch, Heads, SeqLen, HeadDim)
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
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16).to(DEVICE)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16).to(DEVICE)

    if tokenizer_a.pad_token is None: tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token
    model_a.eval(); model_b.eval()

    print("\n--- Determining Translator Dimensions ---")
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
    
    clean_k_state_dict = {k.replace('_orig_mod.', ''): v for k, v in k_state_dict.items()}
    clean_v_state_dict = {k.replace('_orig_mod.', ''): v for k, v in v_state_dict.items()}
    
    translators_k.load_state_dict(clean_k_state_dict); translators_v.load_state_dict(clean_v_state_dict)
    translators_k.to(DEVICE).eval(); translators_v.to(DEVICE).eval()
    print("✅ Translators loaded and in evaluation mode.")

    return model_a, model_b, tokenizer_a, tokenizer_b, translators_k, translators_v

def run_test():
    model_a, model_b, tokenizer_a, tokenizer_b, translators_k, translators_v = load_models_and_translators()
    gc.collect(); torch.cuda.empty_cache()

    print("\n--- Loading MuSiQue Validation Dataset ---")
    dataset = load_dataset("dgslibisey/MuSiQue", split="validation")
    random.seed(SEED)
    test_indices = random.sample(range(len(dataset)), NUM_TESTS)
    print(f"Loaded {len(dataset)} examples, testing {NUM_TESTS} random samples.")

    total_f1_score = 0
    total_em_score = 0
    for i, example_idx in enumerate(tqdm(test_indices, desc="Running KV Translation (Corrected Setup)")):
        example = dataset[example_idx]
        if len(example['question_decomposition']) != 2:
            continue
        # --- Context Preparation (Matching Corrected Baseline) ---
        paragraph_idx_a = example['question_decomposition'][0]['paragraph_support_idx']
        paragraph_idx_b = example['question_decomposition'][1]['paragraph_support_idx']
        
        # Get 3 distractors
        distractor_text = [p["paragraph_text"] for p in example["paragraphs"] if p['is_supporting'] == False]
        distractor_text = random.sample(distractor_text, min(3, len(distractor_text))) 
        
        # Context for Model A (First Hop)
        context_text_a = [example['paragraphs'][paragraph_idx_a]['paragraph_text']] + distractor_text
        random.shuffle(context_text_a)
        context_text_a = "\n\n".join(context_text_a)
        
        # --- Step 1: Generate KV Cache from Model A ---
        
        first_hop_prompt = (
            f"Extract the key fact required to answer the following question from the context. Be concise and state only the fact.\n\n"
            f"Context: {context_text_a}\n\n"
            f"Question: {example['question_decomposition'][0]['question']}\n\n"
            f"Fact:"
        )
        
        # Use generate_kv_cache to get the KV cache from Model A
        sk, sv = generate_kv_cache([first_hop_prompt], tokenizer_a, model_a, DEVICE, MAX_CTX_TOKENS)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
                translated_k = [translators_k[j](sk[j]) for j in range(NUM_LAYERS)]
                translated_v = [translators_v[j](sv[j]) for j in range(NUM_LAYERS)]
        
        translated_cache = tuple(zip(translated_k, translated_v))
        del sk, sv
        gc.collect(); torch.cuda.empty_cache()

        # Context for Model B (Second Hop)
        context_text_b = [example['paragraphs'][paragraph_idx_b]['paragraph_text']] + distractor_text
        random.shuffle(context_text_b)
        model_b_context = "\n\n".join(context_text_b)
        
        second_hop_prompt = (
            f"Use the Context below and the facts implicitly encoded in the prompt history to answer the Final Question.\n\n"
            f"Context: {model_b_context}\n\n"
            f"Final Question: {example['question']}\n\n"
            f"Final Answer:"
        )
        
        q_inputs = tokenizer_b(second_hop_prompt, return_tensors="pt", truncation=True, max_length=MAX_CTX_TOKENS).to(DEVICE)
        
        if q_inputs.input_ids.shape[1] == 0:
            tqdm.write(f"Skipping example {example['id']} due to empty tokenized question.")
            continue

        # Calculate generation parameters based on the combined length
        context_len = translated_cache[0][0].shape[2] 
        question_len = q_inputs.input_ids.shape[1]
        
        # Create attention mask covering the cache + the new question tokens
        attention_mask = torch.ones(1, context_len + question_len, device=DEVICE)
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

        # The response starts *after* the prompt we fed to Model B
        response_start_index = q_inputs.input_ids.shape[1]
        response = tokenizer_b.decode(generated[0, response_start_index:], skip_special_tokens=True).strip()
        
        cleaned_response = response.split('\n')[0].split('.')[0] if '.' in response.split('\n')[0] else response.split('\n')[0]

        ground_truth_answer = example['answer'].strip()
        
        f1 = calculate_f1_score(cleaned_response, ground_truth_answer)
        total_f1_score += f1
        em = calculate_exact_match_score(cleaned_response, ground_truth_answer)
        total_em_score += em
        
        tqdm.write(f"{example['id']}")
        split_prompt_a = first_hop_prompt.split('\n')[0]
        split_prompt_b = second_hop_prompt.split('\n')[0]
        first_question = first_hop_prompt.split('\n')[-3]
        final_question = second_hop_prompt.split('\n')[-3]
        
        tqdm.write(f"Model A Encoded Prompt: {split_prompt_a}")
        tqdm.write(f"{first_question}")
        tqdm.write(f"Model B Final Prompt: {split_prompt_b}")
        tqdm.write(f"{final_question}")
        tqdm.write(f"Model B Output: {cleaned_response}")
        tqdm.write(f"Ground Truth: {ground_truth_answer}")
        
        gc.collect(); torch.cuda.empty_cache()


    average_f1 = total_f1_score / NUM_TESTS
    average_em = total_em_score / NUM_TESTS
    print(f"\n--- Final KV Translation (Corrected Setup) Results ---")
    print(f"Average F1 Score across {NUM_TESTS} samples: {average_f1:.4f}")
    print(f"Average Exact Match Score across {NUM_TESTS} samples: {average_em:.4f}")


if __name__ == "__main__":
    run_test()
