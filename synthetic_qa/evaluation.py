import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
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
    print("Usage: python measure_baselines_fixed.py <qwen|mistral> <num_facts>")
    sys.exit(1)

if sys.argv[1] == 'qwen':
    MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
else:
    MODEL_A = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

NUM_FACTS = int(sys.argv[2])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EVAL_SAMPLES = 200
MAX_NEW_TOKENS = 32
SEED = 456

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


def build_prompt(context, question):
    """Build prompt that works well for both models."""
    return f"""Read the context and answer the question with just the answer, no explanation.

Context: {context}

Question: {question}

Answer:"""


def clean_response(response, expected_answer=None):
    """
    Clean model response to extract answer.
    More aggressive cleaning to handle verbose models.
    """
    response = response.strip()
    
    # Split on newline and take first line
    response = response.split('\n')[0].strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "The answer is",
        "Answer:",
        "Based on the context,",
        "According to the context,",
    ]
    for prefix in prefixes_to_remove:
        if response.lower().startswith(prefix.lower()):
            response = response[len(prefix):].strip()
    
    # If response contains the expected answer at the start, extract it
    # This handles cases like "Dr. Evans placed the folder" -> "Dr. Evans"
    if expected_answer:
        normalized_expected = normalize_text(expected_answer)
        words_expected = normalized_expected.split()
        words_response = response.split()
        
        # Check if response starts with the expected answer words
        if len(words_response) >= len(words_expected):
            # Try to match the expected answer at the beginning
            potential_match = ' '.join(words_response[:len(words_expected)])
            if normalize_text(potential_match) == normalized_expected:
                return potential_match
    
    # Fallback: truncate at sentence endings
    for ending in ['.', '?', '!', ',']:
        if ending in response:
            response = response.split(ending)[0].strip()
            break
    
    return response


def evaluate_model(model, tokenizer, test_examples, model_name, device):
    """Evaluate a model on test examples."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print('='*60)
    
    f1_scores = []
    em_scores = []
    by_type = {'who': [], 'what': [], 'where': []}
    
    for i, (context, question, answer, q_type) in enumerate(tqdm(test_examples, desc=model_name)):
        prompt = build_prompt(context, question)
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            prediction = clean_response(response, expected_answer=answer)
            
            f1 = calculate_f1_score(prediction, answer)
            em = calculate_exact_match(prediction, answer)
            
            f1_scores.append(f1)
            em_scores.append(em)
            by_type[q_type].append(f1)
            
            # Show first few examples
            if i < 5:
                print(f"\n  Example {i+1}:")
                print(f"    Q: {question}")
                print(f"    GT: {answer}")
                print(f"    Raw: {repr(response[:80])}")
                print(f"    Cleaned: {prediction}")
                print(f"    F1: {f1:.3f}, EM: {em:.0f}")
        
        except Exception as e:
            print(f"  Example {i+1} failed: {e}")
            f1_scores.append(0.0)
            em_scores.append(0.0)
    
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_em = sum(em_scores) / len(em_scores)
    
    print(f"\n  Results for {model_name}:")
    print(f"    Average F1: {avg_f1:.4f}")
    print(f"    Average EM: {avg_em:.4f}")
    print(f"    By type - Who: {sum(by_type['who'])/len(by_type['who']):.3f}, "
          f"What: {sum(by_type['what'])/len(by_type['what']):.3f}, "
          f"Where: {sum(by_type['where'])/len(by_type['where']):.3f}")
    
    return avg_f1, avg_em, by_type


def main():
    print(f"Device: {DEVICE}")
    print(f"Measuring baseline performance (FIXED)")
    print(f"  Model A: {MODEL_A}")
    print(f"  Model B: {MODEL_B}")
    print(f"  Num facts: {NUM_FACTS}")
    print(f"  Num samples: {NUM_EVAL_SAMPLES}")
    print()
    
    # Generate test set
    print("Generating test set...")
    test_examples = [generate_synthetic_qa(NUM_FACTS) for _ in range(NUM_EVAL_SAMPLES)]
    type_counts = Counter([ex[3] for ex in test_examples])
    print(f"Question types: {dict(type_counts)}")
    
    # Load and evaluate Model A
    print(f"\n--- Loading Model A: {MODEL_A} ---")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL_A, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token
    model_a.eval()
    
    f1_a, em_a, by_type_a = evaluate_model(
        model_a, tokenizer_a, test_examples, 
        f"Model A ({MODEL_A.split('/')[-1]})", DEVICE
    )
    
    del model_a
    torch.cuda.empty_cache()
    
    # Load and evaluate Model B
    print(f"\n--- Loading Model B: {MODEL_B} ---")
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL_B, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    if tokenizer_b.pad_token is None:
        tokenizer_b.pad_token = tokenizer_b.eos_token
    model_b.eval()
    
    f1_b, em_b, by_type_b = evaluate_model(
        model_b, tokenizer_b, test_examples,
        f"Model B ({MODEL_B.split('/')[-1]})", DEVICE
    )
    
    # Summary
    print("\n" + "="*70)
    print("BASELINE SUMMARY (FIXED)")
    print("="*70)
    print(f"Task: {NUM_FACTS}-fact synthetic QA ({NUM_EVAL_SAMPLES} samples)")
    print()
    print(f"{'Model':<40} {'F1 Score':<12} {'Exact Match':<12}")
    print("-"*65)
    print(f"{'Model A (Source: ' + MODEL_A.split('/')[-1] + ')':<40} {f1_a:<12.4f} {em_a:<12.4f}")
    print(f"{'Model B (Target: ' + MODEL_B.split('/')[-1] + ')':<40} {f1_b:<12.4f} {em_b:<12.4f}")
    print("="*70)
    
    # Analysis
    print("\n📊 Analysis:")
    if f1_b > f1_a:
        gap = f1_b - f1_a
        print(f"  ✅ Model B is better by {gap:.3f} F1 points")
        print(f"  • Model A F1 ({f1_a:.3f}): Lower bound")
        print(f"  • Model B F1 ({f1_b:.3f}): Upper bound (target)")
        
        your_f1 = 0.596  # Your translated result
        recovered = (your_f1 - f1_a) / gap * 100 if gap > 0 else 0
        print(f"\n  Your translation (F1={your_f1:.3f}):")
        print(f"    Recovers {recovered:.1f}% of the performance gap")
        print(f"    Room for improvement: {f1_b - your_f1:.3f} F1 points")
    else:
        print(f"  ⚠️  Model A ({f1_a:.3f}) outperforms Model B ({f1_b:.3f})")
        print(f"  Consider swapping translation direction (B → A)")


if __name__ == "__main__":
    main()
