import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import gc

logging.set_verbosity_error()

###########################
# Configuration
###########################
if len(sys.argv) < 2:
    print("Usage: python visualize_kv_translation_musique.py <qwen|mistral>")
    sys.exit(1)

if sys.argv[1] not in ['qwen', 'mistral']:
    print("First argument must be 'qwen' or 'mistral'")
    sys.exit(1)

if sys.argv[1] == 'qwen':
    MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
else:
    MODEL_A = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

NUM_LAYERS = 28 if 'Qwen' in MODEL_A else 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_PATH = "kv_translators_musique.pth"
MAX_CTX_TOKENS = 1024
MIXED_PRECISION = True
SEED = 42
NUM_EXAMPLES = 20  # Analyze multiple examples for statistical significance
###########################

torch.manual_seed(SEED)
random.seed(SEED)

def format_musique_prompt(example):
    """Format a MuSiQue example into a contextual QA prompt."""
    # Get supporting paragraph index from first question decomposition
    supporting_idx = example['question_decomposition'][0]['paragraph_support_idx']
    
    # Filter for only supporting paragraphs
    supporting_paragraphs = [
        p['paragraph_text'] 
        for p in example['paragraphs'] 
        if p['idx'] == supporting_idx
    ]
    
    context_paragraphs = "\n\n".join(supporting_paragraphs)
    
    # Get the question text
    question_text = example['question_decomposition'][0]['question']
    
    # Format as contextual QA
    prompt = f"Context:\n{context_paragraphs}\n\nQuestion: {question_text} Answer:"
    
    return prompt, question_text, example['question_decomposition'][0].get('answer', '')

class SimpleDeepTranslator(nn.Module):
    def __init__(self, input_size, output_size, target_heads, target_head_dim):
        super().__init__()
        self.target_heads, self.target_head_dim = target_heads, target_head_dim
        hidden_size = max(input_size, output_size)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, output_size, bias=False),
        )

    def forward(self, cache_tensor_a):
        batch, _, seq_len, _ = cache_tensor_a.shape
        x = cache_tensor_a.permute(0, 2, 1, 3).contiguous().view(batch * seq_len, -1)
        y = self.net(x)
        y = y.view(batch, seq_len, self.target_heads, self.target_head_dim)
        return y.permute(0, 2, 1, 3).contiguous()

def generate_kv_caches(prompt, tokenizer_a, tokenizer_b, model_a, model_b, device, max_length):
    """Generate source and ground truth target KV caches."""
    inputs_a = tokenizer_a([prompt], return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    inputs_b = tokenizer_b([prompt], return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            out_a = model_a(**inputs_a, use_cache=True)
            out_b = model_b(**inputs_b, use_cache=True)
        kv_a = out_a.past_key_values
        kv_b = out_b.past_key_values

    source_keys = [kv_a[i][0] for i in range(NUM_LAYERS)]
    source_vals = [kv_a[i][1] for i in range(NUM_LAYERS)]
    target_keys = [kv_b[i][0] for i in range(NUM_LAYERS)]
    target_vals = [kv_b[i][1] for i in range(NUM_LAYERS)]
    
    del out_a, out_b, kv_a, kv_b, inputs_a, inputs_b
    return source_keys, source_vals, target_keys, target_vals

def compute_cache_statistics(translated_cache, target_cache, cache_type="keys"):
    """Compute various statistics comparing translated and target caches."""
    stats = {
        'mse_per_layer': [],
        'cosine_sim_per_layer': [],
        'correlation_per_layer': [],
        'relative_error_per_layer': []
    }
    
    for i in range(NUM_LAYERS):
        trans = translated_cache[i].cpu().float()
        target = target_cache[i].cpu().float()
        
        # MSE
        mse = torch.mean((trans - target) ** 2).item()
        stats['mse_per_layer'].append(mse)
        
        # Cosine similarity (flatten)
        trans_flat = trans.reshape(-1)
        target_flat = target.reshape(-1)
        cos_sim = torch.nn.functional.cosine_similarity(trans_flat.unsqueeze(0), target_flat.unsqueeze(0)).item()
        stats['cosine_sim_per_layer'].append(cos_sim)
        
        # Correlation
        trans_np = trans_flat.numpy()
        target_np = target_flat.numpy()
        corr = np.corrcoef(trans_np, target_np)[0, 1]
        stats['correlation_per_layer'].append(corr)
        
        # Relative error
        rel_error = (torch.norm(trans - target) / torch.norm(target)).item()
        stats['relative_error_per_layer'].append(rel_error)
    
    return stats

def visualize_caches(all_translated_keys, all_target_keys, all_translated_vals, all_target_vals, 
                     prompts, num_examples):
    """Create comprehensive visualizations of cache comparisons across multiple examples."""
    
    # Compute statistics for all examples
    all_key_stats = []
    all_val_stats = []
    
    print(f"Computing statistics across {num_examples} examples...")
    for ex_idx in range(num_examples):
        key_stats = compute_cache_statistics(
            [all_translated_keys[ex_idx][i] for i in range(NUM_LAYERS)],
            [all_target_keys[ex_idx][i] for i in range(NUM_LAYERS)],
            "keys"
        )
        val_stats = compute_cache_statistics(
            [all_translated_vals[ex_idx][i] for i in range(NUM_LAYERS)],
            [all_target_vals[ex_idx][i] for i in range(NUM_LAYERS)],
            "values"
        )
        all_key_stats.append(key_stats)
        all_val_stats.append(val_stats)
    
    # Aggregate statistics (mean and std across examples)
    def aggregate_stats(stats_list, metric):
        per_layer_values = np.array([s[metric] for s in stats_list])  # [num_examples, num_layers]
        mean_per_layer = np.mean(per_layer_values, axis=0)
        std_per_layer = np.std(per_layer_values, axis=0)
        return mean_per_layer, std_per_layer
    
    key_mse_mean, key_mse_std = aggregate_stats(all_key_stats, 'mse_per_layer')
    key_cos_mean, key_cos_std = aggregate_stats(all_key_stats, 'cosine_sim_per_layer')
    key_rel_mean, key_rel_std = aggregate_stats(all_key_stats, 'relative_error_per_layer')
    
    val_mse_mean, val_mse_std = aggregate_stats(all_val_stats, 'mse_per_layer')
    val_cos_mean, val_cos_std = aggregate_stats(all_val_stats, 'cosine_sim_per_layer')
    val_rel_mean, val_rel_std = aggregate_stats(all_val_stats, 'relative_error_per_layer')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Add title
    fig.suptitle(f'KV Cache Translation Analysis (MuSiQue Dataset, {num_examples} Examples)\nAggregated Statistics', 
                 fontsize=14, fontweight='bold')
    
    # 1. MSE per layer (with error bars)
    ax1 = fig.add_subplot(gs[0, 0])
    layers = list(range(NUM_LAYERS))
    ax1.errorbar(layers, key_mse_mean, yerr=key_mse_std, fmt='o-', label='Keys', 
                 linewidth=2, capsize=3, alpha=0.8)
    ax1.errorbar(layers, val_mse_mean, yerr=val_mse_std, fmt='s-', label='Values', 
                 linewidth=2, capsize=3, alpha=0.8)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('MSE')
    ax1.set_title(f'Mean Squared Error per Layer (n={num_examples})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cosine similarity per layer (with error bars)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.errorbar(layers, key_cos_mean, yerr=key_cos_std, fmt='o-', label='Keys', 
                 linewidth=2, capsize=3, alpha=0.8)
    ax2.errorbar(layers, val_cos_mean, yerr=val_cos_std, fmt='s-', label='Values', 
                 linewidth=2, capsize=3, alpha=0.8)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title(f'Cosine Similarity per Layer (n={num_examples})')
    ax2.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.8, 1.05])
    
    # 3. Relative error per layer (with error bars)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.errorbar(layers, key_rel_mean, yerr=key_rel_std, fmt='o-', label='Keys', 
                 linewidth=2, capsize=3, alpha=0.8)
    ax3.errorbar(layers, val_rel_mean, yerr=val_rel_std, fmt='s-', label='Values', 
                 linewidth=2, capsize=3, alpha=0.8)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Relative Error')
    ax3.set_title(f'Relative Error per Layer (n={num_examples})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4-6. Heatmaps from first example (representative)
    mid_layer = NUM_LAYERS // 2
    
    # 4. Heatmap of translated keys (middle layer, first example)
    ax4 = fig.add_subplot(gs[1, 0])
    trans_k_sample = all_translated_keys[0][mid_layer][0, :, :, :].mean(dim=0).cpu().float().numpy()
    sns.heatmap(trans_k_sample, cmap='coolwarm', center=0, ax=ax4, cbar_kws={'label': 'Value'})
    ax4.set_title(f'Translated Keys (Layer {mid_layer}, Example 1)')
    ax4.set_xlabel('Head Dimension')
    ax4.set_ylabel('Sequence Position')
    
    # 5. Heatmap of target keys (middle layer, first example)
    ax5 = fig.add_subplot(gs[1, 1])
    target_k_sample = all_target_keys[0][mid_layer][0, :, :, :].mean(dim=0).cpu().float().numpy()
    sns.heatmap(target_k_sample, cmap='coolwarm', center=0, ax=ax5, cbar_kws={'label': 'Value'})
    ax5.set_title(f'Ground Truth Keys (Layer {mid_layer}, Example 1)')
    ax5.set_xlabel('Head Dimension')
    ax5.set_ylabel('Sequence Position')
    
    # 6. Heatmap of difference (averaged across all examples)
    ax6 = fig.add_subplot(gs[1, 2])
    diffs = []
    for ex_idx in range(num_examples):
        trans = all_translated_keys[ex_idx][mid_layer][0, :, :, :].mean(dim=0).cpu().float().numpy()
        targ = all_target_keys[ex_idx][mid_layer][0, :, :, :].mean(dim=0).cpu().float().numpy()
        diffs.append(trans - targ)
    avg_diff = np.mean(diffs, axis=0)
    max_diff = np.abs(avg_diff).max()
    sns.heatmap(avg_diff, cmap='RdBu_r', center=0, vmin=-max_diff, vmax=max_diff, ax=ax6, 
                cbar_kws={'label': 'Avg Difference'})
    ax6.set_title(f'Average Difference (Layer {mid_layer}, n={num_examples})')
    ax6.set_xlabel('Head Dimension')
    ax6.set_ylabel('Sequence Position')
    
    # 7. Box plot of MSE across layers
    ax7 = fig.add_subplot(gs[2, 0])
    mse_data_keys = [all_key_stats[i]['mse_per_layer'] for i in range(num_examples)]
    mse_data_vals = [all_val_stats[i]['mse_per_layer'] for i in range(num_examples)]
    
    positions = np.arange(NUM_LAYERS)
    bp1 = ax7.boxplot([np.array(mse_data_keys)[:, i] for i in range(NUM_LAYERS)], 
                       positions=positions-0.2, widths=0.3, patch_artist=True,
                       boxprops=dict(facecolor='lightblue'))
    bp2 = ax7.boxplot([np.array(mse_data_vals)[:, i] for i in range(NUM_LAYERS)], 
                       positions=positions+0.2, widths=0.3, patch_artist=True,
                       boxprops=dict(facecolor='lightcoral'))
    ax7.set_xlabel('Layer')
    ax7.set_ylabel('MSE')
    ax7.set_title(f'MSE Distribution Across Layers (n={num_examples})')
    ax7.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Keys', 'Values'])
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Distribution comparison across all examples
    ax8 = fig.add_subplot(gs[2, 1])
    # Sample some translated and target values from all examples
    trans_samples = []
    target_samples = []
    for ex_idx in range(min(5, num_examples)):  # Use first 5 examples to avoid memory issues
        trans_samples.extend(all_translated_keys[ex_idx][mid_layer].flatten().cpu().float().numpy())
        target_samples.extend(all_target_keys[ex_idx][mid_layer].flatten().cpu().float().numpy())
    
    ax8.hist(trans_samples, bins=50, alpha=0.6, label='Translated', edgecolor='black', density=True)
    ax8.hist(target_samples, bins=50, alpha=0.6, label='Ground Truth', edgecolor='black', density=True)
    ax8.set_xlabel('Value')
    ax8.set_ylabel('Density')
    ax8.set_title(f'Value Distribution Comparison (Layer {mid_layer})')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary statistics table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_data = [
        ['Metric', 'Keys', 'Values'],
        ['Avg MSE', 
         f'{np.mean(key_mse_mean):.4f}±{np.mean(key_mse_std):.4f}', 
         f'{np.mean(val_mse_mean):.4f}±{np.mean(val_mse_std):.4f}'],
        ['Avg Cosine Sim', 
         f'{np.mean(key_cos_mean):.4f}±{np.mean(key_cos_std):.4f}', 
         f'{np.mean(val_cos_mean):.4f}±{np.mean(val_cos_std):.4f}'],
        ['Avg Rel Error', 
         f'{np.mean(key_rel_mean):.4f}±{np.mean(key_rel_std):.4f}', 
         f'{np.mean(val_rel_mean):.4f}±{np.mean(val_rel_std):.4f}'],
        ['', '', ''],
        ['Min Cosine Sim', 
         f'{np.min(key_cos_mean):.4f}', 
         f'{np.min(val_cos_mean):.4f}'],
        ['Max MSE', 
         f'{np.max(key_mse_mean):.4f}', 
         f'{np.max(val_mse_mean):.4f}'],
    ]
    
    table = ax9.table(cellText=summary_data, cellLoc='center', loc='center',
                      colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax9.set_title(f'Summary Statistics (n={num_examples})', fontweight='bold', pad=20)
    
    # Save figure
    output_filename = f'kv_cache_visualization_musique_{num_examples}examples.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_filename}")
    
    return all_key_stats, all_val_stats

def main():
    print(f"Device: {DEVICE}")
    print(f"Visualizing KV cache translation for MuSiQue dataset")
    print(f"Analyzing {NUM_EXAMPLES} examples for statistical robustness")
    print()
    
    # Load models
    print("--- Loading Models and Tokenizers ---")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    
    if tokenizer_a.pad_token is None: tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token
    
    model_a.eval()
    model_b.eval()
    
    # Load MuSiQue validation dataset
    print("--- Loading MuSiQue Validation Dataset ---")
    try:
        musique_dataset = load_dataset("dgslibisey/MuSiQue", split="validation").select(range(NUM_EXAMPLES))
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        print(f"Loading smaller subset...")
        musique_dataset = load_dataset("dgslibisey/MuSiQue", split="validation").select(range(min(NUM_EXAMPLES, 20)))
    
    print(f"Loaded {len(musique_dataset)} examples")
    
    # Generate prompts from MuSiQue examples
    print(f"--- Generating {len(musique_dataset)} Sample Contexts ---")
    prompts = []
    questions = []
    answers = []
    
    for i in range(len(musique_dataset)):
        example = musique_dataset[i]
        prompt, question, answer = format_musique_prompt(example)
        prompts.append(prompt)
        questions.append(question)
        answers.append(answer)
        
        if i < 3:  # Print first 3
            print(f"\nExample {i+1}:")
            print(f"  Context: {prompt[:150]}...")
            print(f"  Question: {question}")
            print(f"  Answer: {answer}")
    print()
    
    # Generate KV caches for all examples
    print(f"--- Generating KV Caches for {len(musique_dataset)} Examples ---")
    all_source_keys = []
    all_source_vals = []
    all_target_keys = []
    all_target_vals = []
    
    for i, prompt in enumerate(prompts):
        if (i + 1) % 5 == 0:
            print(f"Processing example {i+1}/{len(musique_dataset)}...")
        source_keys, source_vals, target_keys, target_vals = generate_kv_caches(
            prompt, tokenizer_a, tokenizer_b, model_a, model_b, DEVICE, MAX_CTX_TOKENS
        )
        all_source_keys.append(source_keys)
        all_source_vals.append(source_vals)
        all_target_keys.append(target_keys)
        all_target_vals.append(target_vals)
    
    # Load translators (using dimensions from first example)
    print(f"--- Loading Translators from {LOAD_PATH} ---")
    
    translators_k, translators_v = nn.ModuleList(), nn.ModuleList()
    for i in range(NUM_LAYERS):
        sk, sv = all_source_keys[0][i], all_source_vals[0][i]
        tk, tv = all_target_keys[0][i], all_target_vals[0][i]
        
        k_in_size = sk.shape[1] * sk.shape[3]
        k_out_size = tk.shape[1] * tk.shape[3]
        v_in_size = sv.shape[1] * sv.shape[3]
        v_out_size = tv.shape[1] * tv.shape[3]
        
        translators_k.append(SimpleDeepTranslator(k_in_size, k_out_size, tk.shape[1], tk.shape[3]))
        translators_v.append(SimpleDeepTranslator(v_in_size, v_out_size, tv.shape[1], tv.shape[3]))
    
    # Load weights
    checkpoint = torch.load(LOAD_PATH, map_location=DEVICE)
    translators_k.load_state_dict(checkpoint['translators_k_state_dict'])
    translators_v.load_state_dict(checkpoint['translators_v_state_dict'])
    translators_k.to(DEVICE).eval()
    translators_v.to(DEVICE).eval()
    print("✅ Translators loaded")
    
    # Translate caches for all examples
    print(f"--- Translating KV Caches for {len(musique_dataset)} Examples ---")
    all_translated_keys = []
    all_translated_vals = []
    
    for ex_idx in range(len(musique_dataset)):
        if (ex_idx + 1) % 5 == 0:
            print(f"Translating example {ex_idx+1}/{len(musique_dataset)}...")
        
        translated_keys = []
        translated_vals = []
        
        with torch.no_grad():
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
                for i in range(NUM_LAYERS):
                    trans_k = translators_k[i](all_source_keys[ex_idx][i])
                    trans_v = translators_v[i](all_source_vals[ex_idx][i])
                    translated_keys.append(trans_k)
                    translated_vals.append(trans_v)
        
        all_translated_keys.append(translated_keys)
        all_translated_vals.append(translated_vals)
    
    print("✅ Translation complete for all examples")
    
    # Visualize aggregated results
    print("\n--- Creating Aggregated Visualizations ---")
    all_key_stats, all_val_stats = visualize_caches(
        all_translated_keys, all_target_keys, 
        all_translated_vals, all_target_vals, 
        prompts, len(musique_dataset)
    )
    
    # Print detailed summary
    print("\n" + "="*80)
    print(f"SUMMARY STATISTICS (Aggregated across {len(musique_dataset)} MuSiQue examples)")
    print("="*80)
    
    # Aggregate across all examples and layers
    all_key_mse = [stat['mse_per_layer'] for stat in all_key_stats]
    all_val_mse = [stat['mse_per_layer'] for stat in all_val_stats]
    all_key_cos = [stat['cosine_sim_per_layer'] for stat in all_key_stats]
    all_val_cos = [stat['cosine_sim_per_layer'] for stat in all_val_stats]
    
    print(f"\nKeys:")
    print(f"  Average MSE:          {np.mean(all_key_mse):.6f} ± {np.std(all_key_mse):.6f}")
    print(f"  Average Cosine Sim:   {np.mean(all_key_cos):.4f} ± {np.std(all_key_cos):.4f}")
    print(f"  Min Cosine Sim:       {np.min(all_key_cos):.4f}")
    print(f"  Max Cosine Sim:       {np.max(all_key_cos):.4f}")
    
    print(f"\nValues:")
    print(f"  Average MSE:          {np.mean(all_val_mse):.6f} ± {np.std(all_val_mse):.6f}")
    print(f"  Average Cosine Sim:   {np.mean(all_val_cos):.4f} ± {np.std(all_val_cos):.4f}")
    print(f"  Min Cosine Sim:       {np.min(all_val_cos):.4f}")
    print(f"  Max Cosine Sim:       {np.max(all_val_cos):.4f}")
    
    print("\n" + "="*80)
    print(f"\n✅ Done! Check the generated PNG file for detailed visualizations.")
    print(f"   File: kv_cache_visualization_musique_{len(musique_dataset)}examples.png")

if __name__ == "__main__":
    main()
