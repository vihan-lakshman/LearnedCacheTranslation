# (Imports and setup constants remain the same)
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

# Only Model B is required for inference
MODEL_B = "Qwen/Qwen2.5-7B-Instruct" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Increased context tokens to better fit the full QA context
MAX_CTX_TOKENS = 2048 
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
    # Using the original, potentially permissive EM logic
    em = 1 if ground_truth in prediction else 0 
    return em

# --- Model Loading (Only Model B) ---

def load_model_b_only():
    print("--- Loading Model B and Tokenizer ---")
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B)
    
    # Load Model B in float16
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16).to(DEVICE)

    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token
    model_b.eval()
    
    print("✅ Model B loaded and in evaluation mode.")
    return model_b, tokenizer_b


# --- Zero-Shot Test Function ---

def run_zero_shot_test():
    # Load model B only
    model_b, tokenizer_b = load_model_b_only()
    gc.collect(); torch.cuda.empty_cache()

    print("\n--- Loading MuSiQue Validation Dataset ---")
    dataset = load_dataset("dgslibisey/MuSiQue", split="validation")
    random.seed(SEED)
    test_indices = random.sample(range(len(dataset)), NUM_TESTS)
    print(f"Loaded {len(dataset)} examples, testing {NUM_TESTS} random samples.")

    total_f1_score = 0
    total_em_score = 0
    
    for i, example_idx in enumerate(tqdm(test_indices, desc="Running Zero-Shot Evaluation")):
        example = dataset[example_idx]
        
        # We need both supporting paragraphs for a multi-hop answer
        supporting_paras = [p["paragraph_text"] for p in example["paragraphs"] if p['is_supporting'] == True]
        
        # Use 3 distractors as in the original script, combined with supporting paras
        distractor_text = [p["paragraph_text"] for p in example["paragraphs"] if p['is_supporting'] == False]
        distractor_text = random.sample(distractor_text, min(3, len(distractor_text))) 
        
        full_context_text = supporting_paras + distractor_text
        random.shuffle(full_context_text)
        full_context_text = "\n\n".join(full_context_text)
        
        try:
            # --- Step 1: Model B Generates Final Answer in One Go ---
            
            zero_shot_prompt = (
                f"Answer the final question by synthesizing facts from the provided context.\n\n"
                f"Context:\n{full_context_text}\n\n"
                f"Question: {example['question']}\n\n"
                f"Answer:"
            )
            
            # Truncate to the new, larger context window
            q_inputs = tokenizer_b(zero_shot_prompt, return_tensors="pt", truncation=True, max_length=MAX_CTX_TOKENS).to(DEVICE)

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
            # Simple cleanup: take the first sentence/line
            cleaned_response = final_response.split('\n')[0].split('.')[0] if '.' in final_response.split('\n')[0] else final_response.split('\n')[0]
            
            del q_inputs, generated_b
            
            # --- Step 2: Evaluate ---
            ground_truth_answer = example['answer'].strip()
            
            f1 = calculate_f1_score(cleaned_response, ground_truth_answer)
            total_f1_score += f1
            em = calculate_exact_match_score(cleaned_response, ground_truth_answer)
            total_em_score += em

            tqdm.write(f"ID: {example['id']}")
            tqdm.write(f"Model B Output: {cleaned_response}")
            tqdm.write(f"Ground Truth: {ground_truth_answer}")
            
            gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            tqdm.write(f"\nAn error occurred on example ID {example.get('id', 'N/A')}: {e}")
            traceback.print_exc()

    average_f1 = total_f1_score / NUM_TESTS
    average_em = total_em_score / NUM_TESTS
    print(f"\n--- Final Zero-Shot (Model B Only) Results ---")
    print(f"Average F1 Score across {NUM_TESTS} samples: {average_f1:.4f}")
    print(f"Average Exact Match Score across {NUM_TESTS} samples: {average_em:.4f}")


if __name__ == "__main__":
    run_zero_shot_test()
