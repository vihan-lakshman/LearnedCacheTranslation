import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from tqdm import trange
import sys
import re
import string
from collections import Counter

logging.set_verbosity_error()

###########################
# Configuration
###########################
if len(sys.argv) < 3:
    print("Usage: python baseline_graduated_synthetic_qa.py <qwen|mistral> <num_facts>")
    print("  num_facts: 1, 2, 3, or 4 (number of facts in context)")
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

if sys.argv[1] == 'qwen':
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
else:
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EVAL_SAMPLES = 100  # More samples for baseline
MAX_NEW_TOKENS = 32
SEED = 42
###########################

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
    """Generate a single fact."""
    subjects = ["Dr. Evans", "The pilot", "The scientist", "Agent Helix", "The cartographer",
                "Captain Rivera", "Professor Kim", "Detective Chen", "Colonel Hayes", "Dr. Martinez"]
    actions = ["placed", "hid", "secured", "delivered", "found", 
               "stored", "moved", "retrieved", "archived", "discovered"]
    objects = ["the blue folder", "the silver key", "the encrypted drive", "the heavy package", "the coded map",
               "the sealed envelope", "the metal case", "the classified document", "the ancient artifact", "the prototype device"]
    locations = ["in the library", "in the hangar", "in the laboratory", "at the north gate", "behind the console",
                 "in the archive room", "at the main entrance", "in the storage facility", "near the fountain", "in the vault"]
    
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
    """Generate synthetic QA with multiple facts in context."""
    facts = [generate_fact() for _ in range(num_facts)]
    
    # Build context from all facts
    context = " ".join([f['text'] for f in facts])
    
    # Randomly select one fact to ask about
    target_fact = random.choice(facts)
    
    # Generate question about the target fact
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

def main():
    print(f"Device: {DEVICE}")
    print(f"Running BASELINE evaluation with {NUM_FACTS} fact(s) per context")
    print("Model B receives the full context directly (no translation)")
    print()
    
    print("--- Loading Model B and Tokenizer ---")
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    
    if tokenizer_b.pad_token is None:
        tokenizer_b.pad_token = tokenizer_b.eos_token
    
    model_b.eval()
    print("✅ Model loaded")
    
    print(f"\n--- Running Baseline Evaluation on {NUM_EVAL_SAMPLES} Examples ---")
    
    total_f1 = 0
    f1_scores = []
    
    for i in trange(NUM_EVAL_SAMPLES, desc="Evaluating"):
        context, question, ground_truth_answer = generate_synthetic_qa(NUM_FACTS)
        
        # Model B gets the full context directly (no KV translation)
        full_prompt = f"Context:\n{context}\n\nQuestion: {question} Answer:"
        
        try:
            inputs = tokenizer_b(full_prompt, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                generated = model_b.generate(
                    input_ids=inputs.input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer_b.eos_token_id
                )
            
            # Decode only the generated part (skip the input prompt)
            response = tokenizer_b.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Clean the response
            cleaned_response = response.split('\n')[0].strip()
            for ending in ['.', '?', '!']:
                if ending in cleaned_response:
                    cleaned_response = cleaned_response.split(ending)[0].strip()
                    break
            
            f1 = calculate_f1_score(cleaned_response, ground_truth_answer)
            total_f1 += f1
            f1_scores.append(f1)
            
            # Print first few examples
            if i < 5:
                print(f"\n--- Example {i+1} ---")
                print(f"Context ({len(context.split())} words): {context}")
                print(f"Question: {question}")
                print(f"Ground Truth: {ground_truth_answer}")
                print(f"Prediction: {cleaned_response}")
                print(f"F1: {f1:.4f}")
        
        except Exception as e:
            print(f"\nExample {i+1} failed: {e}")
            f1_scores.append(0.0)
    
    avg_f1 = total_f1 / NUM_EVAL_SAMPLES
    
    # Calculate statistics
    import numpy as np
    f1_array = np.array(f1_scores)
    std_f1 = np.std(f1_array)
    min_f1 = np.min(f1_array)
    max_f1 = np.max(f1_array)
    
    print("\n" + "="*80)
    print(f"BASELINE RESULTS ({NUM_FACTS} FACTS)")
    print("="*80)
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Std Dev:          {std_f1:.4f}")
    print(f"Min F1:           {min_f1:.4f}")
    print(f"Max F1:           {max_f1:.4f}")
    print(f"Total Examples:   {NUM_EVAL_SAMPLES}")
    print("="*80)
    
    # Print distribution
    print("\nF1 Score Distribution:")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(bins)-1):
        count = sum(1 for f1 in f1_scores if bins[i] <= f1 < bins[i+1])
        pct = 100 * count / NUM_EVAL_SAMPLES
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {count:3d} ({pct:5.1f}%)")
    
    # Perfect matches
    perfect = sum(1 for f1 in f1_scores if f1 == 1.0)
    print(f"\nPerfect Matches (F1=1.0): {perfect}/{NUM_EVAL_SAMPLES} ({100*perfect/NUM_EVAL_SAMPLES:.1f}%)")
    
    print("\n" + "="*80)
    print("COMPARISON TO TRANSLATION")
    print("="*80)
    print("This baseline shows Model B's performance when given the full context directly.")
    print("Compare this to the translation results to see how much information is lost.")
    print("="*80)

if __name__ == "__main__":
    main()
