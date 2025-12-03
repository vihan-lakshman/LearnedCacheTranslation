import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# Config
MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 200 # Set to None for full test

def extract_answer_number(text):
    """Finds the last number in the text (standard GSM8K eval)."""
    # Look for '####' which marks the answer in GSM8K gold lines
    if "####" in text:
        text = text.split("####")[1]
    
    # Extract all numbers (integers or floats)
    matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if matches:
        return matches[-1]
    return None

def eval_model(model_name, dataset):
    print(f"\nEvaluating {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=DEVICE)
    model.eval()
    
    correct = 0
    total = 0
    
    for i, ex in enumerate(tqdm(dataset)):
        prompt = f"Question: {ex['question']}\nLet's think step by step.\nAnswer:"
        gt_ans = extract_answer_number(ex['answer'])
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            generated = model.generate(
                **inputs, 
                max_new_tokens=256, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False # Greedy for baseline
            )
        
        output = tokenizer.decode(generated[0], skip_special_tokens=True)
        # Remove the prompt from the output to parse only the answer
        response = output[len(prompt):]
        pred_ans = extract_answer_number(response)
        
        if pred_ans == gt_ans:
            correct += 1
        total += 1
        
        if i < 3:
            print(f"Q: {ex['question'][:50]}...")
            print(f"GT: {gt_ans} | Pred: {pred_ans}")

    acc = correct / total
    print(f"Accuracy for {model_name}: {acc:.4f}")
    del model, tokenizer
    torch.cuda.empty_cache()
    return acc

def main():
    data = load_dataset("gsm8k", "main", split="test")
    if NUM_SAMPLES:
        data = data.select(range(NUM_SAMPLES))
        
    acc_a = eval_model(MODEL_A, data)
    acc_b = eval_model(MODEL_B, data)
    
    print("\n" + "="*40)
    print(f"BASELINE RESULTS (GSM8K - {NUM_SAMPLES} samples)")
    print(f"Model A (1.5B): {acc_a:.4f}")
    print(f"Model B (7.0B): {acc_b:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
