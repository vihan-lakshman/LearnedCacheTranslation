import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset
from tqdm import tqdm
import sys

logging.set_verbosity_error()

###########################
# Configuration
###########################
if len(sys.argv) < 2:
    print("Usage: python eval_boolq_baselines.py <qwen|mistral> [num_samples]")
    sys.exit(1)

if sys.argv[1] == 'qwen':
    MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
else:
    MODEL_A = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

NUM_EVAL_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 8
SEED = 456

torch.manual_seed(SEED)
###########################


def build_prompt(passage, question):
    """Best prompt format for BoolQ."""
    return f"""Read the passage and answer the question with just "Yes" or "No".

Passage: {passage}

Question: {question}

Answer:"""


def parse_yes_no(response):
    """Parse model response to Yes/No."""
    response = response.strip().lower()
    
    if response.startswith('yes'):
        return True
    if response.startswith('no'):
        return False
    
    first_word = response.split()[0] if response.split() else ""
    first_word = first_word.strip('.,!?')
    
    if first_word == 'yes':
        return True
    if first_word == 'no':
        return False
    
    if 'yes' in response and 'no' not in response:
        return True
    if 'no' in response and 'yes' not in response:
        return False
    
    return None


def evaluate_model(model, tokenizer, eval_data, model_name, device):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print('='*60)
    
    correct = 0
    total = 0
    unparseable = 0
    
    for i, ex in enumerate(tqdm(eval_data, desc=model_name)):
        prompt = build_prompt(ex['passage'], ex['question'])
        ground_truth = ex['answer']
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
            
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            prediction = parse_yes_no(response)
            
            if prediction is None:
                unparseable += 1
                if unparseable <= 5:
                    print(f"\n  Unparseable: {repr(response[:50])}")
            else:
                total += 1
                if prediction == ground_truth:
                    correct += 1
            
            if i < 5:
                print(f"\n  Example {i+1}:")
                print(f"    Q: {ex['question'][:60]}...")
                print(f"    GT: {'Yes' if ground_truth else 'No'}")
                print(f"    Response: {repr(response[:50])}")
                print(f"    Parsed: {prediction}")
        
        except Exception as e:
            print(f"  Example {i+1} failed: {e}")
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'unparseable': unparseable
    }


def main():
    print("="*70)
    print("BOOLQ BASELINE EVALUATION")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Model A: {MODEL_A}")
    print(f"Model B: {MODEL_B}")
    print(f"Num samples: {NUM_EVAL_SAMPLES}")
    print()
    
    # Load dataset - FIXED: iterate properly
    print("Loading BoolQ dataset...")
    dataset = load_dataset("google/boolq")
    num_samples = min(NUM_EVAL_SAMPLES, len(dataset['validation']))
    eval_data = [dataset['validation'][i] for i in range(num_samples)]
    print(f"Loaded {len(eval_data)} validation examples")
    
    # Check label balance
    yes_count = sum(1 for ex in eval_data if ex['answer'])
    print(f"Label distribution: Yes={yes_count}, No={len(eval_data)-yes_count}")
    
    results = {}
    
    # Evaluate Model A
    print(f"\n--- Loading Model A: {MODEL_A} ---")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL_A, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token
    model_a.eval()
    
    results['Model A'] = evaluate_model(model_a, tokenizer_a, eval_data,
                                        f"Model A ({MODEL_A.split('/')[-1]})", DEVICE)
    
    del model_a
    torch.cuda.empty_cache()
    
    # Evaluate Model B
    print(f"\n--- Loading Model B: {MODEL_B} ---")
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL_B, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    if tokenizer_b.pad_token is None:
        tokenizer_b.pad_token = tokenizer_b.eos_token
    model_b.eval()
    
    results['Model B'] = evaluate_model(model_b, tokenizer_b, eval_data,
                                        f"Model B ({MODEL_B.split('/')[-1]})", DEVICE)
    
    # Print summary
    print("\n" + "="*70)
    print("BOOLQ BASELINE RESULTS")
    print("="*70)
    print(f"Dataset: BoolQ validation set")
    print(f"Samples: {len(eval_data)}")
    print()
    
    print(f"{'Model':<45} {'Accuracy':<12} {'Correct':<12} {'Unparseable':<12}")
    print("-"*80)
    for name, res in results.items():
        model_str = MODEL_A.split('/')[-1] if name == 'Model A' else MODEL_B.split('/')[-1]
        print(f"{name + ' (' + model_str + ')':<45} {res['accuracy']:<12.4f} {res['correct']}/{res['total']:<8} {res['unparseable']:<12}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    gap = results['Model B']['accuracy'] - results['Model A']['accuracy']
    print(f"  Model A accuracy: {results['Model A']['accuracy']:.4f} (lower bound)")
    print(f"  Model B accuracy: {results['Model B']['accuracy']:.4f} (upper bound)")
    print(f"  Gap: {gap:.4f} ({gap*100:.1f} percentage points)")
    print("="*70)


if __name__ == "__main__":
    main()
