import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging

logging.set_verbosity_error()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_B = "Qwen/Qwen2.5-7B-Instruct"

random.seed(42)

def generate_fact():
    subjects = ["Dr. Evans", "The pilot", "The scientist", "Agent Helix"]
    actions = ["placed", "hid", "secured", "delivered"]
    objects = ["the blue folder", "the silver key", "the encrypted drive", "the heavy package"]
    locations = ["in the library", "in the hangar", "in the laboratory", "at the north gate"]
    
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

def generate_synthetic_qa(num_facts=4):
    facts = [generate_fact() for _ in range(num_facts)]
    context = " ".join([f['text'] for f in facts])
    target_fact = facts[0]  # Use first fact for consistency
    
    question = f"Who {target_fact['action']} {target_fact['object']}?"
    answer = target_fact['subject']
    
    return context, question, answer

# Generate a fixed test case
context, question, answer = generate_synthetic_qa(4)

print("="*80)
print("TEST CASE")
print("="*80)
print(f"Context: {context}")
print(f"Question: {question}")
print(f"Expected Answer: {answer}")
print()

# Different prompt formats to try
prompt_formats = {
    "basic": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    
    "basic_space": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer: ",
    
    "instructed": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer with only the name or phrase, nothing else:",
    
    "short": f"{context}\n\nQ: {question}\nA:",
    
    "explicit": f"""Read the context and answer the question with just the answer, no explanation.

Context: {context}

Question: {question}

Answer:""",
}

def test_model(model_name, tokenizer, model):
    print(f"\n{'='*80}")
    print(f"TESTING: {model_name}")
    print("="*80)
    
    for format_name, prompt in prompt_formats.items():
        print(f"\n--- Format: {format_name} ---")
        print(f"Prompt ends with: ...{repr(prompt[-30:])}")
        
        # Test raw generation
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        print(f"Input tokens: {inputs.input_ids.shape[1]}")
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_tokens = output[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response_with_special = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        
        print(f"Generated tokens: {len(generated_tokens)}")
        print(f"Raw response: {repr(response[:200])}")
        print(f"With special tokens: {repr(response_with_special[:200])}")
        
        # Also try with chat template
        if format_name == "basic":
            print(f"\n--- Format: chat_template ---")
            messages = [
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer with just the answer:"}
            ]
            chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"Chat prompt ends with: ...{repr(chat_prompt[-50:])}")
            
            inputs = tokenizer(chat_prompt, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_tokens = output[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"Chat response: {repr(response[:200])}")

# Load and test Model A
print("\nLoading Model A...")
tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
if tokenizer_a.pad_token is None:
    tokenizer_a.pad_token = tokenizer_a.eos_token
model_a.eval()

test_model("Model A (1.5B)", tokenizer_a, model_a)

# Free memory
del model_a
torch.cuda.empty_cache()

# Load and test Model B
print("\nLoading Model B...")
tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
if tokenizer_b.pad_token is None:
    tokenizer_b.pad_token = tokenizer_b.eos_token
model_b.eval()

test_model("Model B (7B)", tokenizer_b, model_b)

# Additional diagnostic: check tokenizer differences
print("\n" + "="*80)
print("TOKENIZER COMPARISON")
print("="*80)

test_text = "Dr. Evans"
tokens_a = tokenizer_a.encode(test_text)
tokens_b = tokenizer_b.encode(test_text)
print(f"'{test_text}' tokenized by A: {tokens_a} -> {[tokenizer_a.decode([t]) for t in tokens_a]}")
print(f"'{test_text}' tokenized by B: {tokens_b} -> {[tokenizer_b.decode([t]) for t in tokens_b]}")

# Check if they're using the same vocab
print(f"\nTokenizer A vocab size: {tokenizer_a.vocab_size}")
print(f"Tokenizer B vocab size: {tokenizer_b.vocab_size}")
print(f"Same tokenizer? {tokenizer_a.vocab_size == tokenizer_b.vocab_size and tokens_a == tokens_b}")
