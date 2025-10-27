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
from tqdm import trange

# Suppress informational warnings
logging.set_verbosity_error()

###########################
# Configuration
# Usage: python test_musique_baseline.py <qwen|mistral>
###########################
if len(sys.argv) < 2 or sys.argv[1] not in ['qwen', 'mistral']:
    print("Usage: python test_musique_baseline.py <qwen|mistral>")
    sys.exit(1)

if sys.argv[1] == 'qwen':
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
else:
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Settings for MuSiQue task
NUM_PROMPTS = 500 # Use more samples since the dataset is large
MAX_CTX_TOKENS = 1024 # MuSiQue contexts are long, use a larger limit
MAX_NEW_TOKENS = 64
NUM_TESTS = 50
SEED = 42
###########################

def normalize_text(s):
    # Standard text normalization for F1 calculation in QA
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s


def calculate_f1_score(prediction, ground_truth):
    # F1 score calculation
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


def run_test():
    print(f"Device: {DEVICE}")
    print("--- Loading MuSiQue Dataset ---")
    
    # Load and select a subset of the training split for testing
    try:
        dataset = load_dataset("dgslibisey/MuSiQue", split="train").select(range(NUM_PROMPTS))
    except Exception:
        print("Could not load full dataset. Trying 'small' version or reducing NUM_PROMPTS.")
        dataset = load_dataset("dgslibisey/MuSiQue", split="train").select(range(min(NUM_PROMPTS, 100)))

    print(f"Loaded {len(dataset)} examples for evaluation.")

    print("\n--- Loading Baseline Model and Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)

    # Use bfloat16 for modern GPUs, otherwise float16
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading model with dtype: {dtype}")
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=dtype, trust_remote_code=True).to(DEVICE)

    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model_b.eval()

    print("\n--- Starting MuSiQue Baseline Evaluation ---")
    total_f1_score = 0
    
    # Randomly sample NUM_TESTS indices for evaluation
    random.seed(SEED)
    test_indices = random.sample(range(len(dataset)), min(NUM_TESTS, len(dataset)))

    for i, example_idx in enumerate(test_indices):
        example = dataset[example_idx]
        print(f"\n--- Test {i + 1}/{len(test_indices)} ---")

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
        
        print(f"Question: '{question_text}'")
        print(f"Ground Truth Answer: '{ground_truth_answer}'")
        print(f"Using {len(supporting_paragraphs)} supporting paragraphs as context.")
        
        try:
            # Create a contextual prompt for the baseline model
            messages = [{
                "role": "user",
                "content": f"Based on the following context, answer the question with just the direct answer. "
                           f"Context:\n{context_paragraphs}\n\nQuestion: {question_text}"
            }]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Tokenize the entire prompt, truncating if it exceeds MAX_CTX_TOKENS
            inputs = tokenizer(full_prompt, return_tensors="pt", max_length=MAX_CTX_TOKENS, truncation=True).to(DEVICE)

            # Generate the answer from scratch (Full Prefill)
            with torch.no_grad():
                generated = model_b.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode the generated response, excluding the input prompt
            response = tokenizer.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            # Clean the response to only take the first line/sentence
            cleaned_response = response.split('\n')[0].split('.')[0].strip()

            print(f"Model Response (cleaned): '{cleaned_response}'")

            f1 = calculate_f1_score(cleaned_response, ground_truth_answer)
            total_f1_score += f1
            print(f"F1 Score: {f1:.4f}")

        except Exception as e:
            print(f"Test failed with an error: {e}")
            traceback.print_exc()

    average_f1 = total_f1_score / len(test_indices)
    print(f"\n--- Final Results ---")
    print(f"Average F1 Score (MuSiQue CQA Baseline) across {len(test_indices)} samples: {average_f1:.4f}")

if __name__ == "__main__":
    run_test()
