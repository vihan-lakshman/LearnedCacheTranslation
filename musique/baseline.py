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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_CTX_TOKENS = 2048 # Increased to 2048 to prevent prompt truncation issues
MAX_NEW_TOKENS = 64
NUM_TESTS = 50
MIXED_PRECISION = True
SEED = 42

# --- Utility Functions (Normalize Text, F1, EM) remain the same ---

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
    # Using the original, potentially permissive EM logic for direct comparison
    em = 1 if ground_truth in prediction else 0 
    return em

# --- Model Loading (Simplified) ---

def load_models_only():
    print("--- Loading Base Models and Tokenizers ---")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B)
    
    # Load Models in float16
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16).to(DEVICE)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16).to(DEVICE)

    if tokenizer_a.pad_token is None: tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token
    model_a.eval(); model_b.eval()
    
    print("✅ Models loaded and in evaluation mode.")
    return model_a, model_b, tokenizer_a, tokenizer_b


# --- CORRECTED Baseline Test Function ---

def run_corrected_baseline_test():
    model_a, model_b, tokenizer_a, tokenizer_b = load_models_only()
    gc.collect(); torch.cuda.empty_cache()

    print("\n--- Loading MuSiQue Validation Dataset ---")
    dataset = load_dataset("dgslibisey/MuSiQue", split="validation")
    random.seed(SEED)
    test_indices = random.sample(range(len(dataset)), NUM_TESTS)
    print(f"Loaded {len(dataset)} examples, testing {NUM_TESTS} random samples.")

    total_f1_score = 0
    total_em_score = 0
    
    for i, example_idx in enumerate(tqdm(test_indices, desc="Running CORRECTED Baseline Evaluation")):
        example = dataset[example_idx]
        if len(example['question_decomposition']) != 2:
            continue
        try:
            # Get supporting context for Hop A and Hop B separately
            paragraph_idx_a = example['question_decomposition'][0]['paragraph_support_idx']
            paragraph_idx_b = example['question_decomposition'][1]['paragraph_support_idx']
            
            # Get 3 distractors (common for both contexts for simplicity)
            distractor_text = [p["paragraph_text"] for p in example["paragraphs"] if p['is_supporting'] == False]
            distractor_text = random.sample(distractor_text, min(3, len(distractor_text))) 
            
            # Context for Model A (First Hop)
            context_text_a = [example['paragraphs'][paragraph_idx_a]['paragraph_text']] + distractor_text
            random.shuffle(context_text_a)
            context_text_a = "\n\n".join(context_text_a)
            
            # --- Step 1: Model A Generates Text Answer for the First Hop ---
            
            # REFINED PROMPT: Instruct Model A to extract the key fact
            first_hop_prompt = (
                f"Extract the key fact required to answer the following question from the context. Be concise and state only the fact.\n\n"
                f"Context: {context_text_a}\n\n"
                f"Question: {example['question_decomposition'][0]['question']}\n\n"
                f"Fact:"
            )
            
            inputs_a = tokenizer_a(first_hop_prompt, return_tensors="pt", truncation=True, max_length=MAX_CTX_TOKENS).to(DEVICE)

            with torch.no_grad():
                 with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
                    generated_a = model_a.generate(
                        **inputs_a,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer_a.eos_token_id
                    )

            # Decode and clean Model A's answer
            response_start_index_a = inputs_a.input_ids.shape[1]
            first_hop_answer = tokenizer_a.decode(generated_a[0, response_start_index_a:], skip_special_tokens=True).strip()
            cleaned_answer_a = first_hop_answer.split('\n')[0].split('.')[0] if '.' in first_hop_answer.split('\n')[0] else first_hop_answer.split('\n')[0]

            del inputs_a, generated_a
            gc.collect(); torch.cuda.empty_cache()

            # --- Step 2: Model B Generates Final Answer using Model A's Text ---

            # Context for Model B (Second Hop)
            context_text_b = [example['paragraphs'][paragraph_idx_b]['paragraph_text']] + distractor_text
            random.shuffle(context_text_b)
            context_text_b = "\n\n".join(context_text_b)

            # CORRECTED PROMPT: Only include the Intermediate Fact and Context B
            second_hop_prompt = (
                f"Use the Intermediate Fact and the Context below to answer the Final Question.\n\n"
                f"Intermediate Fact: {cleaned_answer_a}\n\n"
                f"Context: {context_text_b}\n\n"
                f"Final Question: {example['question']}\n\n"
                f"Final Answer:"
            )
            
            q_inputs = tokenizer_b(second_hop_prompt, return_tensors="pt", truncation=True, max_length=MAX_CTX_TOKENS).to(DEVICE)
            
            if q_inputs.input_ids.shape[1] == 0:
                tqdm.write(f"Skipping example {example['id']} due to empty tokenized question.")
                continue

            with torch.no_grad():
                with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
                    generated_b = model_b.generate(
                        **q_inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer_b.eos_token_id
                    )

            # Decode and clean Model B's final answer
            response_start_index_b = q_inputs.input_ids.shape[1]
            final_response = tokenizer_b.decode(generated_b[0, response_start_index_b:], skip_special_tokens=True).strip()
            cleaned_response = final_response.split('\n')[0].split('.')[0] if '.' in final_response.split('\n')[0] else final_response.split('\n')[0]
            
            del q_inputs, generated_b
            
            # --- Step 3: Evaluate ---
            ground_truth_answer = example['answer'].strip()
            
            f1 = calculate_f1_score(cleaned_response, ground_truth_answer)
            total_f1_score += f1
            em = calculate_exact_match_score(cleaned_response, ground_truth_answer)
            total_em_score += em

            tqdm.write(f"ID: {example['id']}")
            tqdm.write(f"Intermediate Fact: {cleaned_answer_a}")
            tqdm.write(f"Model B Output: {cleaned_response}")
            tqdm.write(f"Ground Truth: {ground_truth_answer}")
            
            gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            tqdm.write(f"\nAn error occurred on example ID {example.get('id', 'N/A')}: {e}")
            traceback.print_exc()

    average_f1 = total_f1_score / NUM_TESTS
    average_em = total_em_score / NUM_TESTS
    print(f"\n--- Final CORRECTED Text-Based Baseline Results ---")
    print(f"Average F1 Score across {NUM_TESTS} samples: {average_f1:.4f}")
    print(f"Average Exact Match Score across {NUM_TESTS} samples: {average_em:.4f}")


if __name__ == "__main__":
    run_corrected_baseline_test()
