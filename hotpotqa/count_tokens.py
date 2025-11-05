import torch
from transformers import AutoTokenizer, logging
from datasets import load_dataset
from tqdm import tqdm
import sys

logging.set_verbosity_error()

###########################
# Configuration
###########################
if len(sys.argv) < 2:
    print("Usage: python compute_context_lengths.py <qwen|mistral>")
    sys.exit(1)

if sys.argv[1] not in ['qwen', 'mistral']:
    print("First argument must be 'qwen' or 'mistral'")
    sys.exit(1)

if sys.argv[1] == 'qwen':
    MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
else:
    MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

NUM_SAMPLES = 100  # Number of samples to analyze
###########################

def format_musique_prompt(example):
    """Format a MuSiQue example into a contextual QA prompt."""
    supporting_idx = example['question_decomposition'][0]['paragraph_support_idx']
    
    supporting_paragraphs = [
        p['paragraph_text'] 
        for p in example['paragraphs'] 
        if p['idx'] == supporting_idx
    ]
    
    context_paragraphs = "\n\n".join(supporting_paragraphs)
    question_text = example['question_decomposition'][0]['question']
    
    prompt = f"Context:\n{context_paragraphs}\n\nQuestion: {question_text} Answer:"
    
    return prompt

def format_hotpotqa_prompt(example, use_supporting_only=True):
    """Format a HotpotQA example into a contextual QA prompt."""
    
    # Extract supporting fact titles if filtering
    if use_supporting_only and "supporting_facts" in example and example["supporting_facts"]:
        supporting_titles = set(example["supporting_facts"]["title"])
        
        # Filter context to only supporting paragraphs
        context_texts = []
        for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
            if title in supporting_titles:
                context_texts.append(f"{title}: {' '.join(sentences)}")
    else:
        # Use all context
        context_texts = []
        for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
            context_texts.append(f"{title}: {' '.join(sentences)}")
    
    context_combined = "\n\n".join(context_texts)
    question_text = example['question']
    
    prompt = f"Context:\n{context_combined}\n\nQuestion: {question_text} Answer:"
    
    return prompt

def analyze_dataset_lengths(dataset_name, format_fn, tokenizer, num_samples):
    """Analyze context lengths for a dataset."""
    print(f"\n{'='*80}")
    print(f"Analyzing {dataset_name} Dataset")
    print(f"{'='*80}")
    
    # Load dataset
    if dataset_name == "MuSiQue":
        try:
            dataset = load_dataset("dgslibisey/MuSiQue", split="validation")
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return
    elif dataset_name == "HotpotQA":
        try:
            dataset = load_dataset("hotpot_qa", "distractor", split="validation")
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return
    else:
        print(f"Unknown dataset: {dataset_name}")
        return
    
    # Select subset
    num_samples = min(num_samples, len(dataset))
    dataset = dataset.select(range(num_samples))
    
    print(f"Analyzing {num_samples} examples...")
    
    token_counts = []
    word_counts = []
    char_counts = []
    
    for i in tqdm(range(len(dataset)), desc=f"Processing {dataset_name}"):
        try:
            example = dataset[i]
            prompt = format_fn(example)
            
            # Count tokens
            tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
            num_tokens = tokens.input_ids.shape[1]
            token_counts.append(num_tokens)
            
            # Count words
            num_words = len(prompt.split())
            word_counts.append(num_words)
            
            # Count characters
            num_chars = len(prompt)
            char_counts.append(num_chars)
            
        except Exception as e:
            print(f"\nError processing example {i}: {e}")
            continue
    
    if not token_counts:
        print("No valid examples processed!")
        return
    
    # Compute statistics
    avg_tokens = sum(token_counts) / len(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    median_tokens = sorted(token_counts)[len(token_counts) // 2]
    
    avg_words = sum(word_counts) / len(word_counts)
    min_words = min(word_counts)
    max_words = max(word_counts)
    
    avg_chars = sum(char_counts) / len(char_counts)
    min_chars = min(char_counts)
    max_chars = max(char_counts)
    
    # Count examples exceeding common limits
    over_512 = sum(1 for x in token_counts if x > 512)
    over_1024 = sum(1 for x in token_counts if x > 1024)
    over_2048 = sum(1 for x in token_counts if x > 2048)
    
    # Print results
    print(f"\n{'-'*80}")
    print(f"Results for {dataset_name} ({len(token_counts)} examples)")
    print(f"{'-'*80}")
    
    print(f"\nToken Statistics:")
    print(f"  Average:  {avg_tokens:.1f} tokens")
    print(f"  Median:   {median_tokens} tokens")
    print(f"  Min:      {min_tokens} tokens")
    print(f"  Max:      {max_tokens} tokens")
    
    print(f"\nWord Statistics:")
    print(f"  Average:  {avg_words:.1f} words")
    print(f"  Min:      {min_words} words")
    print(f"  Max:      {max_words} words")
    
    print(f"\nCharacter Statistics:")
    print(f"  Average:  {avg_chars:.1f} characters")
    print(f"  Min:      {min_chars} characters")
    print(f"  Max:      {max_chars} characters")
    
    print(f"\nContext Length Distribution:")
    print(f"  Examples > 512 tokens:  {over_512} ({over_512/len(token_counts)*100:.1f}%)")
    print(f"  Examples > 1024 tokens: {over_1024} ({over_1024/len(token_counts)*100:.1f}%)")
    print(f"  Examples > 2048 tokens: {over_2048} ({over_2048/len(token_counts)*100:.1f}%)")
    
    # Show some example lengths
    print(f"\nFirst 10 Token Counts:")
    print(f"  {token_counts[:10]}")
    
    return {
        'avg_tokens': avg_tokens,
        'median_tokens': median_tokens,
        'min_tokens': min_tokens,
        'max_tokens': max_tokens,
        'avg_words': avg_words,
        'avg_chars': avg_chars,
        'over_512': over_512,
        'over_1024': over_1024,
        'over_2048': over_2048,
        'total_samples': len(token_counts)
    }

def main():
    print(f"Loading tokenizer: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded\n")
    
    # Analyze MuSiQue
    musique_stats = analyze_dataset_lengths("MuSiQue", format_musique_prompt, tokenizer, NUM_SAMPLES)
    
    # Analyze HotpotQA
    hotpotqa_stats = analyze_dataset_lengths("HotpotQA", format_hotpotqa_prompt, tokenizer, NUM_SAMPLES)
    
    # Comparative summary
    if musique_stats and hotpotqa_stats:
        print(f"\n{'='*80}")
        print(f"COMPARATIVE SUMMARY")
        print(f"{'='*80}")
        print(f"\n{'Dataset':<15} {'Avg Tokens':<15} {'Median':<15} {'Max':<15} {'>1024':<15}")
        print(f"{'-'*80}")
        print(f"{'MuSiQue':<15} {musique_stats['avg_tokens']:<15.1f} {musique_stats['median_tokens']:<15} {musique_stats['max_tokens']:<15} {musique_stats['over_1024']:<15}")
        print(f"{'HotpotQA':<15} {hotpotqa_stats['avg_tokens']:<15.1f} {hotpotqa_stats['median_tokens']:<15} {hotpotqa_stats['max_tokens']:<15} {hotpotqa_stats['over_1024']:<15}")
        print(f"{'-'*80}")
        
        # Determine which is longer on average
        if musique_stats['avg_tokens'] > hotpotqa_stats['avg_tokens']:
            diff = musique_stats['avg_tokens'] - hotpotqa_stats['avg_tokens']
            pct = (diff / hotpotqa_stats['avg_tokens']) * 100
            print(f"\nMuSiQue contexts are {diff:.1f} tokens ({pct:.1f}%) longer on average than HotpotQA")
        else:
            diff = hotpotqa_stats['avg_tokens'] - musique_stats['avg_tokens']
            pct = (diff / musique_stats['avg_tokens']) * 100
            print(f"\nHotpotQA contexts are {diff:.1f} tokens ({pct:.1f}%) longer on average than MuSiQue")

if __name__ == "__main__":
    main()
