import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from tqdm import trange
import sys

# Suppress warnings
logging.set_verbosity_error()

###########################
# Configuration
###########################
if len(sys.argv) < 2:
    print("Usage: python train_boolq_anchored.py <qwen|mistral>")
    sys.exit(1)

if sys.argv[1] == 'qwen':
    MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
else:
    MODEL_A = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

NUM_LAYERS = 28 if 'Qwen' in MODEL_A else 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = f"kv_translators_boolq_{sys.argv[1]}.pth"

# Stage 1: MSE pretraining
STAGE1_STEPS = 3000
STAGE1_LR = 1e-3
STAGE1_BATCH_SIZE = 16

# Stage 2: NTP fine-tuning with MSE Anchor
STAGE2_STEPS = 1000
STAGE2_LR = 1e-5
STAGE2_GRADIENT_ACCUMULATION = 4
LAMBDA_MSE = 1.0  # Strength of the anchor (Prevent cheating)

# General settings
SEED = 42
MIXED_PRECISION = True
MAX_GRAD_NORM = 1.0
MAX_CTX_TOKENS = 512
MAX_RESPONSE_TOKENS = 8

# Evaluation settings
EVAL_EVERY = 500
NUM_EVAL_SAMPLES = 100
MAX_NEW_TOKENS = 8

# Data generation
NUM_CACHE_PAIRS = 2000
DATA_BATCH_SIZE = 4
###########################

torch.manual_seed(SEED)
random.seed(SEED)


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


class SimpleDeepTranslator(nn.Module):
    def __init__(self, input_size, output_size, target_heads, target_head_dim):
        super().__init__()
        self.target_heads = target_heads
        self.target_head_dim = target_head_dim
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

    def forward(self, cache_tensor):
        batch, num_heads, seq_len, head_dim = cache_tensor.shape
        x = cache_tensor.permute(0, 2, 1, 3).contiguous().view(batch * seq_len, -1)
        y = self.net(x)
        y = y.view(batch, seq_len, self.target_heads, self.target_head_dim)
        return y.permute(0, 2, 1, 3).contiguous()


def get_model_dimensions(tokenizer, model, device, num_layers):
    dummy_input = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**dummy_input, use_cache=True)
    kv_cache = outputs.past_key_values
    dims = []
    if hasattr(kv_cache, 'key_cache'):
        for i in range(num_layers):
            k_shape = kv_cache.key_cache[i].shape
            v_shape = kv_cache.value_cache[i].shape
            dims.append({'k_heads': k_shape[1], 'k_head_dim': k_shape[3], 
                         'v_heads': v_shape[1], 'v_head_dim': v_shape[3]})
    else:
        for i in range(num_layers):
            dims.append({'k_heads': kv_cache[i][0].shape[1], 'k_head_dim': kv_cache[i][0].shape[3], 
                         'v_heads': kv_cache[i][1].shape[1], 'v_head_dim': kv_cache[i][1].shape[3]})
    return dims


def generate_kv_cache_pair(prompt, tokenizer_a, tokenizer_b, model_a, model_b, device, max_length):
    """Generate KV cache pair from both models (Used for Ground Truth in Stage 2)."""
    inputs_a = tokenizer_a([prompt], return_tensors="pt", padding="max_length", 
                           truncation=True, max_length=max_length).to(device)
    inputs_b = tokenizer_b([prompt], return_tensors="pt", padding="max_length", 
                           truncation=True, max_length=max_length).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            out_a = model_a(**inputs_a, use_cache=True)
            out_b = model_b(**inputs_b, use_cache=True)
    
    kv_a, kv_b = out_a.past_key_values, out_b.past_key_values
    
    if hasattr(kv_a, 'key_cache'):
        source_keys = [kv_a.key_cache[i].cpu() for i in range(NUM_LAYERS)]
        source_vals = [kv_a.value_cache[i].cpu() for i in range(NUM_LAYERS)]
    else:
        source_keys = [kv_a[i][0].cpu() for i in range(NUM_LAYERS)]
        source_vals = [kv_a[i][1].cpu() for i in range(NUM_LAYERS)]
    
    if hasattr(kv_b, 'key_cache'):
        target_keys = [kv_b.key_cache[i].cpu() for i in range(NUM_LAYERS)]
        target_vals = [kv_b.value_cache[i].cpu() for i in range(NUM_LAYERS)]
    else:
        target_keys = [kv_b[i][0].cpu() for i in range(NUM_LAYERS)]
        target_vals = [kv_b[i][1].cpu() for i in range(NUM_LAYERS)]
    
    return source_keys, source_vals, target_keys, target_vals


def generate_source_kv_cache(prompt, tokenizer, model, device, max_length):
    """Generate KV cache from source model only (Used for eval)."""
    inputs = tokenizer([prompt], return_tensors="pt", padding="max_length", 
                       truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    kv_cache = outputs.past_key_values
    
    if hasattr(kv_cache, 'key_cache'):
        keys = [kv_cache.key_cache[i] for i in range(NUM_LAYERS)]
        vals = [kv_cache.value_cache[i] for i in range(NUM_LAYERS)]
    else:
        keys = [kv_cache[i][0] for i in range(NUM_LAYERS)]
        vals = [kv_cache[i][1] for i in range(NUM_LAYERS)]
    return keys, vals


def translate_cache(source_keys, source_vals, translators_k, translators_v):
    """Translate and return as DynamicCache."""
    translated_cache = DynamicCache()
    for i in range(NUM_LAYERS):
        trans_k = translators_k[i](source_keys[i])
        trans_v = translators_v[i](source_vals[i])
        translated_cache.update(trans_k, trans_v, i)
    return translated_cache


def compute_ntp_loss(model_b, tokenizer_b, translated_cache, response_text, device):
    """Compute next-token prediction loss."""
    
    # FIX: Add BOS token to ensure length >= 2
    bos = tokenizer_b.bos_token if tokenizer_b.bos_token else " "
    full_text = f"{bos}{response_text}"

    response_ids = tokenizer_b(full_text, return_tensors="pt", padding=False, 
                               truncation=True, max_length=MAX_RESPONSE_TOKENS).input_ids.to(device)
    
    if response_ids.shape[1] < 2:
        return None
    
    cache_seq_len = translated_cache.get_seq_length()
    response_len = response_ids.shape[1]
    attention_mask = torch.ones(1, cache_seq_len + response_len, device=device)
    position_ids = torch.arange(cache_seq_len, cache_seq_len + response_len, device=device).unsqueeze(0)
    
    outputs = model_b(input_ids=response_ids, attention_mask=attention_mask, 
                      past_key_values=translated_cache, position_ids=position_ids, use_cache=False)
    
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = response_ids[:, 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))


def evaluate_translators(model_a, model_b, tokenizer_a, tokenizer_b, 
                         translators_k, translators_v, eval_data, device):
    """Evaluate translators on BoolQ."""
    print("\n" + "-"*60)
    print("Running Evaluation...")
    
    for i in range(NUM_LAYERS):
        translators_k[i].eval()
        translators_v[i].eval()
    
    correct = 0
    total = 0
    examples_shown = 0
    
    num_eval = min(NUM_EVAL_SAMPLES, len(eval_data))
    
    for idx in range(num_eval):
        ex = eval_data[idx]
        prompt = build_prompt(ex['passage'], ex['question'])
        ground_truth = ex['answer']
        
        try:
            source_keys, source_vals = generate_source_kv_cache(
                prompt, tokenizer_a, model_a, device, MAX_CTX_TOKENS
            )
            
            with torch.no_grad():
                with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
                    translated_cache = translate_cache(
                        source_keys, source_vals, translators_k, translators_v
                    )
            
            start_token = tokenizer_b(" ", return_tensors="pt").input_ids.to(device)
            cache_seq_len = translated_cache.get_seq_length()
            attention_mask = torch.ones(1, cache_seq_len + 1, device=device)
            
            with torch.no_grad():
                generated = model_b.generate(
                    input_ids=start_token,
                    attention_mask=attention_mask,
                    past_key_values=translated_cache,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer_b.eos_token_id,
                    cache_position=torch.arange(cache_seq_len, cache_seq_len + 1, device=device)
                )
            
            response = tokenizer_b.decode(generated[0], skip_special_tokens=True)
            prediction = parse_yes_no(response)
            
            if prediction is not None:
                total += 1
                if prediction == ground_truth:
                    correct += 1
            
            if examples_shown < 3:
                print(f"  Q: {ex['question'][:50]}...")
                print(f"  GT: {'Yes' if ground_truth else 'No'} | Pred: {prediction} | Response: {repr(response[:30])}")
                examples_shown += 1
        
        except Exception as e:
            print(f"  Eval failed: {e}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    print("-"*60)
    
    for i in range(NUM_LAYERS):
        translators_k[i].train()
        translators_v[i].train()
    
    return accuracy


def main():
    print(f"Device: {DEVICE}")
    print(f"Two-Stage Training for BoolQ (ANCHORED VERSION)")
    print(f"  Model A: {MODEL_A}")
    print(f"  Model B: {MODEL_B}")
    print(f"  Stage 1 (MSE): {STAGE1_STEPS} steps")
    print(f"  Stage 2 (NTP + MSE Anchor): {STAGE2_STEPS} steps, Lambda={LAMBDA_MSE}")
    print()
    
    # Load dataset
    print("Loading BoolQ dataset...")
    dataset = load_dataset("google/boolq")
    train_data = [dataset['train'][i] for i in range(len(dataset['train']))]
    eval_data = [dataset['validation'][i] for i in range(len(dataset['validation']))]
    random.shuffle(train_data)
    print(f"Train: {len(train_data)}, Validation: {len(eval_data)}")
    
    # Load models
    print("\n--- Loading Models ---")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    
    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None:
        tokenizer_b.pad_token = tokenizer_b.eos_token
    
    model_a.eval()
    model_b.eval()
    for param in model_a.parameters():
        param.requires_grad = False
    for param in model_b.parameters():
        param.requires_grad = False
    
    print(f"✅ Models loaded")
    
    # Get dimensions and create translators
    source_dims = get_model_dimensions(tokenizer_a, model_a, DEVICE, NUM_LAYERS)
    target_dims = get_model_dimensions(tokenizer_b, model_b, DEVICE, NUM_LAYERS)
    
    print("\n--- Creating Translators ---")
    translators_k = nn.ModuleList()
    translators_v = nn.ModuleList()
    
    for i in range(NUM_LAYERS):
        src, tgt = source_dims[i], target_dims[i]
        k_in = src['k_heads'] * src['k_head_dim']
        k_out = tgt['k_heads'] * tgt['k_head_dim']
        v_in = src['v_heads'] * src['v_head_dim']
        v_out = tgt['v_heads'] * tgt['v_head_dim']
        translators_k.append(SimpleDeepTranslator(k_in, k_out, tgt['k_heads'], tgt['k_head_dim']).to(DEVICE))
        translators_v.append(SimpleDeepTranslator(v_in, v_out, tgt['v_heads'], tgt['v_head_dim']).to(DEVICE))
    
    total_params = sum(p.numel() for p in translators_k.parameters()) + sum(p.numel() for p in translators_v.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # =====================
    # STAGE 1: MSE Training
    # =====================
    print("\n" + "="*80)
    print("STAGE 1: MSE PRETRAINING")
    print("="*80)
    
    print(f"\nGenerating {NUM_CACHE_PAIRS} cache pairs...")
    data_by_layer = [[[] for _ in range(NUM_LAYERS)] for _ in range(4)]
    
    data_idx = 0
    for batch_idx in trange(NUM_CACHE_PAIRS // DATA_BATCH_SIZE, desc="Generating data"):
        for _ in range(DATA_BATCH_SIZE):
            ex = train_data[data_idx % len(train_data)]
            data_idx += 1
            prompt = build_prompt(ex['passage'], ex['question'])
            
            sk, sv, tk, tv = generate_kv_cache_pair(
                prompt, tokenizer_a, tokenizer_b, model_a, model_b, DEVICE, MAX_CTX_TOKENS
            )
            for i in range(NUM_LAYERS):
                data_by_layer[0][i].append(sk[i])
                data_by_layer[1][i].append(sv[i])
                data_by_layer[2][i].append(tk[i])
                data_by_layer[3][i].append(tv[i])
    
    data_tensors = [[torch.cat(d) for d in layer_data] for layer_data in data_by_layer]
    num_samples = data_tensors[0][0].shape[0]
    print(f"Generated {num_samples} samples")
    
    # MSE training
    params = list(translators_k.parameters()) + list(translators_v.parameters())
    optimizer = optim.AdamW(params, lr=STAGE1_LR)
    loss_fn_mse = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=(MIXED_PRECISION and DEVICE == 'cuda'))
    
    indices = torch.randperm(num_samples)
    eval_history = []
    
    for step in trange(STAGE1_STEPS, desc="Stage 1 (MSE)"):
        for i in range(NUM_LAYERS):
            translators_k[i].train()
            translators_v[i].train()
        
        start_idx = (step * STAGE1_BATCH_SIZE) % num_samples
        end_idx = start_idx + STAGE1_BATCH_SIZE
        if end_idx > num_samples:
            batch_indices = torch.cat((indices[start_idx:], indices[:end_idx - num_samples]))
        else:
            batch_indices = indices[start_idx:end_idx]
        
        total_loss = 0
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
            for i in range(NUM_LAYERS):
                source_k = data_tensors[0][i][batch_indices].to(DEVICE)
                source_v = data_tensors[1][i][batch_indices].to(DEVICE)
                target_k = data_tensors[2][i][batch_indices].to(DEVICE)
                target_v = data_tensors[3][i][batch_indices].to(DEVICE)
                
                pred_k = translators_k[i](source_k)
                pred_v = translators_v[i](source_v)
                total_loss += loss_fn_mse(pred_k, target_k) + loss_fn_mse(pred_v, target_v)
        
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if (step + 1) % EVAL_EVERY == 0:
            avg_loss = total_loss.item() / (2 * NUM_LAYERS)
            print(f"\n[Stage 1, Step {step+1}] MSE Loss: {avg_loss:.6f}")
            accuracy = evaluate_translators(model_a, model_b, tokenizer_a, tokenizer_b, 
                                            translators_k, translators_v, eval_data, DEVICE)
            eval_history.append(('stage1', step + 1, avg_loss, accuracy))
    
    del data_tensors, data_by_layer
    torch.cuda.empty_cache()
    
    # =====================
    # STAGE 2: NTP + MSE Anchor
    # =====================
    print("\n" + "="*80)
    print("STAGE 2: NTP FINE-TUNING (With MSE Anchor)")
    print("="*80)
    
    optimizer = optim.AdamW(params, lr=STAGE2_LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE2_STEPS, eta_min=STAGE2_LR/10)
    scaler = torch.amp.GradScaler(enabled=(MIXED_PRECISION and DEVICE == 'cuda'))
    
    running_ntp = 0.0
    running_mse = 0.0
    successful_steps = 0
    data_idx = 0
    
    progress_bar = trange(STAGE2_STEPS, desc="Stage 2")
    
    for step in progress_bar:
        for i in range(NUM_LAYERS):
            translators_k[i].train()
            translators_v[i].train()
        
        optimizer.zero_grad()
        batch_ntp_loss = 0.0
        batch_mse_loss = 0.0
        batch_success = 0
        
        for accum_step in range(STAGE2_GRADIENT_ACCUMULATION):
            ex = train_data[data_idx % len(train_data)]
            data_idx += 1
            
            prompt = build_prompt(ex['passage'], ex['question'])
            answer = "Yes" if ex['answer'] else "No"
            response_text = f" {answer}"
            
            try:
                with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
                    # 1. Generate BOTH source and target caches (Ground Truth for Anchor)
                    # We need target keys/vals to act as the "Anchor"
                    sk, sv, tk, tv = generate_kv_cache_pair(
                        prompt, tokenizer_a, tokenizer_b, model_a, model_b, DEVICE, MAX_CTX_TOKENS
                    )
                    
                    # 2. Run Translator & Calculate MSE Anchor
                    # We compute MSE directly on tensors before packing into DynamicCache
                    mse_step_loss = 0.0
                    translated_cache = DynamicCache()
                    
                    for i in range(NUM_LAYERS):
                        s_k_gpu = sk[i].to(DEVICE)
                        s_v_gpu = sv[i].to(DEVICE)
                        t_k_gpu = tk[i].to(DEVICE)
                        t_v_gpu = tv[i].to(DEVICE)
                        
                        pred_k = translators_k[i](s_k_gpu)
                        pred_v = translators_v[i](s_v_gpu)
                        
                        mse_step_loss += loss_fn_mse(pred_k, t_k_gpu) + loss_fn_mse(pred_v, t_v_gpu)
                        translated_cache.update(pred_k, pred_v, i)
                    
                    mse_step_loss = mse_step_loss / (2 * NUM_LAYERS)
                    
                    # 3. Calculate NTP Loss
                    ntp_step_loss = compute_ntp_loss(model_b, tokenizer_b, translated_cache, response_text, DEVICE)
                    
                    if ntp_step_loss is not None:
                        # 4. Combined Loss
                        total_loss = ntp_step_loss + (LAMBDA_MSE * mse_step_loss)
                        
                        total_loss = total_loss / STAGE2_GRADIENT_ACCUMULATION
                        scaler.scale(total_loss).backward()
                        
                        batch_ntp_loss += ntp_step_loss.item()
                        batch_mse_loss += mse_step_loss.item()
                        batch_success += 1
                        
            except Exception as e:
                print(f"\nStep {step}, accum {accum_step} failed: {e}")
                continue
        
        if batch_success > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Normalize statistics
            curr_ntp = batch_ntp_loss / batch_success
            curr_mse = batch_mse_loss / batch_success
            
            running_ntp += curr_ntp
            running_mse += curr_mse
            successful_steps += 1
            
            # Update progress bar description
            progress_bar.set_description(f"Stage 2 (NTP={curr_ntp:.4f}, MSE={curr_mse:.4f})")
        else:
            scaler.update() # Skip step if all failed
        
        if (step + 1) % 100 == 0 and successful_steps > 0:
            avg_ntp = running_ntp / min(100, successful_steps)
            avg_mse = running_mse / min(100, successful_steps)
            print(f"\n[Stage 2, Step {step+1}] NTP: {avg_ntp:.4f} | MSE (Anchor): {avg_mse:.4f}")
            running_ntp = 0.0
            running_mse = 0.0
            successful_steps = 0
        
        if (step + 1) % EVAL_EVERY == 0:
            accuracy = evaluate_translators(model_a, model_b, tokenizer_a, tokenizer_b, 
                                            translators_k, translators_v, eval_data, DEVICE)
            eval_history.append(('stage2', step + 1, batch_ntp_loss, accuracy))
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    final_accuracy = evaluate_translators(model_a, model_b, tokenizer_a, tokenizer_b, 
                                          translators_k, translators_v, eval_data, DEVICE)
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING HISTORY")
    print("="*80)
    print(f"{'Stage':<8} {'Step':<8} {'Loss':<12} {'Accuracy':<12}")
    print("-" * 45)
    for stage, step, loss, acc in eval_history:
        print(f"{stage:<8} {step:<8} {loss:<12.4f} {acc:<12.4f}")
    print("="*80)
    
    # Save
    torch.save({
        'translators_k_state_dict': translators_k.state_dict(),
        'translators_v_state_dict': translators_v.state_dict(),
        'eval_history': eval_history,
        'final_accuracy': final_accuracy,
        'config': {
            'model_a': MODEL_A,
            'model_b': MODEL_B,
            'task': 'boolq',
            'stage1_steps': STAGE1_STEPS,
            'stage2_steps': STAGE2_STEPS,
            'lambda_mse': LAMBDA_MSE
        }
    }, SAVE_PATH)
    
    print(f"\n✅ Saved to {SAVE_PATH}")
    print(f"Final Accuracy: {final_accuracy:.4f}")


if __name__ == "__main__":
    main()
