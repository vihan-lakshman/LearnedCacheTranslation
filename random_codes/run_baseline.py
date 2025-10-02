import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
import traceback
import sys

logging.set_verbosity_error()

if len(sys.argv) < 2 or sys.argv[1] not in ['qwen', 'mistral']:
    print("Usage: python test_random_codes_baseline.py <qwen|mistral>")
    sys.exit(1)

if sys.argv[1] == 'qwen':
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
else:
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CTX_TOKENS = 30 # Max length for the combined context and question
MAX_NEW_TOKENS = 30 # Max length for the generated answer
NUM_TESTS = 100
MIXED_PRECISION = True
SEED = 42

def gen_secret_phrase():
    templates = ["The agent's secret activation code is {}-{}-{}."]
    codenames = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
    template = random.choice(templates)
    a, b = random.sample(codenames, 2)
    c = random.randint(100, 999)
    return template.format(a, b, c)

def run_test():
    print(f"Device: {DEVICE}")
    print("--- Loading Baseline Model and Tokenizer ---")
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)

    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token
    model_b.eval()
    
    print("\n--- Starting Baseline Test ---")
    success_count = 0
    random.seed(SEED)
    
    for test_i in range(NUM_TESTS):
        print(f"\n--- Test {test_i + 1}/{NUM_TESTS} ---")
        unseen_secret = gen_secret_phrase()
        print(f"Secret Context: '{unseen_secret}'")

        try:
            # Create a single, full prompt for the baseline model to process
            question = " Repeat the secret context"
            full_prompt = unseen_secret + question
            
            inputs = tokenizer_b(full_prompt, return_tensors="pt", max_length=MAX_CTX_TOKENS, truncation=True).to(DEVICE)
            
            # Generate the answer from scratch (Full Prefill)
            with torch.no_grad():
                with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
                    generated = model_b.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer_b.eos_token_id
                    )
            
            # Decode only the newly generated tokens
            response = tokenizer_b.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            print(f"Model Response: '{response}'")

            core_code = [p for p in unseen_secret.split() if "-" in p][0].replace('.','')
            if core_code.lower() in response.lower():
                success_count += 1
                print("SUCCESS")
            else:
                print("FAILURE")
        except Exception as e:
            print(f"Test failed with an error: {e}")
            traceback.print_exc()

    print(f"\n--- Final Results ---")
    print(f"Success Rate (Full Prefill Baseline): {success_count}/{NUM_TESTS}")

if __name__ == "__main__":
    run_test()
