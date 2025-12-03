import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from transformers.cache_utils import DynamicCache
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
    print("Usage: python train_two_stage.py <qwen|mistral> <num_facts>")
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
    MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
else:
    MODEL_A = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

NUM_LAYERS = 28 if 'Qwen' in MODEL_A else 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = f"kv_translators_twostage_{NUM_FACTS}facts.pth"

# Stage 1: MSE pretraining
STAGE1_STEPS = 500
STAGE1_LR = 1e-3
STAGE1_BATCH_SIZE = 32

# Stage 2: NTP fine-tuning
STAGE2_STEPS = 1000
STAGE2_LR = 1e-5  # Much lower LR for fine-tuning
STAGE2_GRADIENT_ACCUMULATION = 4

# General settings
SEED = 42
MIXED_PRECISION = True
MAX_GRAD_NORM = 1.0
MAX_CTX_TOKENS = 32 + (NUM_FACTS * 32)
MAX_RESPONSE_TOKENS = 32  # Longer response for better NTP signal

# Evaluation settings
EVAL_EVERY = 500
NUM_EVAL_SAMPLES = 50
MAX_NEW_TOKENS = 32

# Data generation for Stage 1
NUM_CACHE_PAIRS = 2000
DATA_BATCH_SIZE = 4
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
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def generate_fact():
    """Generate a single fact with expanded vocabulary."""
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
    """Generate synthetic QA with multiple facts in context."""
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
    
    return context, question, answer


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

    def forward(self, cache_tensor_a):
        batch, num_heads, seq_len, head_dim = cache_tensor_a.shape
        x = cache_tensor_a.permute(0, 2, 1, 3).contiguous().view(batch * seq_len, -1)
        y = self.net(x)
        y = y.view(batch, seq_len, self.target_heads, self.target_head_dim)
        return y.permute(0, 2, 1, 3).contiguous()


def generate_kv_cache_pair(prompt, tokenizer_a, tokenizer_b, model_a, model_b, device, max_length):
    """Generate KV cache pair from both models for the same prompt."""
    inputs_a = tokenizer_a(
        [prompt], return_tensors="pt", padding="max_length",
        truncation=True, max_length=max_length
    ).to(device)
    inputs_b = tokenizer_b(
        [prompt], return_tensors="pt", padding="max_length",
        truncation=True, max_length=max_length
    ).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            out_a = model_a(**inputs_a, use_cache=True)
            out_b = model_b(**inputs_b, use_cache=True)
    
    kv_a = out_a.past_key_values
    kv_b = out_b.past_key_values
    
    # Handle DynamicCache format
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


def generate_source_kv_cache(prompt, tokenizer_a, model_a, device, max_length):
    """Generate KV cache from source model only."""
    inputs = tokenizer_a(
        [prompt], return_tensors="pt", padding="max_length",
        truncation=True, max_length=max_length
    ).to(device)
    
    with torch.no_grad():
        outputs = model_a(**inputs, use_cache=True)
    
    kv_cache = outputs.past_key_values
    
    if hasattr(kv_cache, 'key_cache'):
        source_keys = [kv_cache.key_cache[i] for i in range(NUM_LAYERS)]
        source_vals = [kv_cache.value_cache[i] for i in range(NUM_LAYERS)]
    else:
        source_keys = [kv_cache[i][0] for i in range(NUM_LAYERS)]
        source_vals = [kv_cache[i][1] for i in range(NUM_LAYERS)]
    
    return source_keys, source_vals


def translate_cache_tuple(source_keys, source_vals, translators_k, translators_v):
    """Translate and return as tuple (for MSE training)."""
    translated_keys = []
    translated_vals = []
    
    for i in range(NUM_LAYERS):
        trans_k = translators_k[i](source_keys[i])
        trans_v = translators_v[i](source_vals[i])
        translated_keys.append(trans_k)
        translated_vals.append(trans_v)
    
    return translated_keys, translated_vals


def translate_cache_dynamic(source_keys, source_vals, translators_k, translators_v):
    """Translate and return as DynamicCache (for generation)."""
    translated_cache = DynamicCache()
    
    for i in range(NUM_LAYERS):
        trans_k = translators_k[i](source_keys[i])
        trans_v = translators_v[i](source_vals[i])
        translated_cache.update(trans_k, trans_v, i)
    
    return translated_cache


def compute_ntp_loss(model_b, tokenizer_b, translated_cache, response_text, device):
    """Compute next-token prediction loss using translated cache."""
    response_ids = tokenizer_b(
        response_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_RESPONSE_TOKENS
    ).input_ids.to(device)
    
    if response_ids.shape[1] < 2:
        return None  # Need at least 2 tokens for NTP loss
    
    cache_seq_len = translated_cache.get_seq_length()
    response_len = response_ids.shape[1]
    
    attention_mask = torch.ones(1, cache_seq_len + response_len, device=device)
    position_ids = torch.arange(cache_seq_len, cache_seq_len + response_len, device=device).unsqueeze(0)
    
    outputs = model_b(
        input_ids=response_ids,
        attention_mask=attention_mask,
        past_key_values=translated_cache,
        position_ids=position_ids,
        use_cache=False,
    )
    
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = response_ids[:, 1:].contiguous()
    
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    return loss


def evaluate_translators(model_a, model_b, tokenizer_a, tokenizer_b,
                         translators_k, translators_v, device, num_facts):
    """Evaluate translators on held-out synthetic examples."""
    print("\n" + "-"*60)
    print("Running Evaluation...")
    
    for i in range(NUM_LAYERS):
        translators_k[i].eval()
        translators_v[i].eval()
    
    total_f1 = 0
    examples_shown = 0
    
    for i in range(NUM_EVAL_SAMPLES):
        context, question, ground_truth_answer = generate_synthetic_qa(num_facts)
        context_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        
        try:
            source_keys, source_vals = generate_source_kv_cache(
                context_prompt, tokenizer_a, model_a, device, MAX_CTX_TOKENS
            )
            
            with torch.no_grad():
                with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
                    translated_cache = translate_cache_dynamic(
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
            
            response = tokenizer_b.decode(generated[0], skip_special_tokens=True).strip()
            cleaned_response = response.split('\n')[0].strip()
            for ending in ['.', '?', '!']:
                if ending in cleaned_response:
                    cleaned_response = cleaned_response.split(ending)[0].strip()
                    break
            
            f1 = calculate_f1_score(cleaned_response, ground_truth_answer)
            total_f1 += f1
            
            if examples_shown < 3:
                print(f"  Q: {question}")
                print(f"  GT: {ground_truth_answer} | Pred: {cleaned_response} | F1: {f1:.3f}")
                examples_shown += 1
        
        except Exception as e:
            print(f"  Eval {i+1} failed: {e}")
    
    avg_f1 = total_f1 / NUM_EVAL_SAMPLES
    print(f"  Average F1: {avg_f1:.4f}")
    print("-"*60)
    
    for i in range(NUM_LAYERS):
        translators_k[i].train()
        translators_v[i].train()
    
    return avg_f1


def get_model_dimensions(tokenizer, model, device, num_layers):
    """Get KV cache dimensions for a model."""
    dummy_input = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**dummy_input, use_cache=True)
    
    kv_cache = outputs.past_key_values
    dims = []
    
    if hasattr(kv_cache, 'key_cache'):
        for i in range(num_layers):
            k_shape = kv_cache.key_cache[i].shape
            v_shape = kv_cache.value_cache[i].shape
            dims.append({
                'k_heads': k_shape[1], 'k_head_dim': k_shape[3],
                'v_heads': v_shape[1], 'v_head_dim': v_shape[3]
            })
    else:
        for i in range(num_layers):
            k_shape = kv_cache[i][0].shape
            v_shape = kv_cache[i][1].shape
            dims.append({
                'k_heads': k_shape[1], 'k_head_dim': k_shape[3],
                'v_heads': v_shape[1], 'v_head_dim': v_shape[3]
            })
    
    return dims


def main():
    print(f"Device: {DEVICE}")
    print(f"Two-Stage Training: MSE → NTP")
    print(f"  Facts per context: {NUM_FACTS}")
    print(f"  Stage 1 (MSE): {STAGE1_STEPS} steps, LR={STAGE1_LR}")
    print(f"  Stage 2 (NTP): {STAGE2_STEPS} steps, LR={STAGE2_LR}")
    print()
    
    # Load models
    print("--- Loading Models ---")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)
    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL_A, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL_B, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    
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
    
    print(f"✅ Models loaded: {MODEL_A} → {MODEL_B}")
    
    # Get dimensions
    source_dims = get_model_dimensions(tokenizer_a, model_a, DEVICE, NUM_LAYERS)
    target_dims = get_model_dimensions(tokenizer_b, model_b, DEVICE, NUM_LAYERS)
    
    # Create translators
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
    
    # Generate training data for MSE
    print(f"\nGenerating {NUM_CACHE_PAIRS} cache pairs...")
    data_by_layer = [[[] for _ in range(NUM_LAYERS)] for _ in range(4)]
    
    for _ in trange(NUM_CACHE_PAIRS // DATA_BATCH_SIZE, desc="Generating data"):
        prompts = []
        for _ in range(DATA_BATCH_SIZE):
            context, question, _ = generate_synthetic_qa(NUM_FACTS)
            prompts.append(f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        
        for prompt in prompts:
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
    loss_fn = nn.MSELoss()
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
                
                total_loss += loss_fn(pred_k, target_k) + loss_fn(pred_v, target_v)
        
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if (step + 1) % EVAL_EVERY == 0:
            avg_loss = total_loss.item() / (2 * NUM_LAYERS)
            print(f"\n[Stage 1, Step {step+1}] MSE Loss: {avg_loss:.6f}")
            avg_f1 = evaluate_translators(
                model_a, model_b, tokenizer_a, tokenizer_b,
                translators_k, translators_v, DEVICE, NUM_FACTS
            )
            eval_history.append(('stage1', step + 1, avg_loss, avg_f1))
    
    # Free MSE training data
    del data_tensors, data_by_layer
    torch.cuda.empty_cache()
    
    # =====================
    # STAGE 2: NTP Fine-tuning
    # =====================
    print("\n" + "="*80)
    print("STAGE 2: NTP FINE-TUNING")
    print("="*80)
    
    # Lower learning rate for fine-tuning
    optimizer = optim.AdamW(params, lr=STAGE2_LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE2_STEPS, eta_min=STAGE2_LR/10)
    scaler = torch.amp.GradScaler(enabled=(MIXED_PRECISION and DEVICE == 'cuda'))
    
    running_loss = 0.0
    successful_steps = 0
    
    for step in trange(STAGE2_STEPS, desc="Stage 2 (NTP)"):
        for i in range(NUM_LAYERS):
            translators_k[i].train()
            translators_v[i].train()
        
        optimizer.zero_grad()
        batch_loss = 0.0
        batch_success = 0
        
        for accum_step in range(STAGE2_GRADIENT_ACCUMULATION):
            context, question, answer = generate_synthetic_qa(NUM_FACTS)
            context_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            # Use longer response with some context for better gradient signal
            response_text = f" {answer}."
            
            try:
                with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
                    source_keys, source_vals = generate_source_kv_cache(
                        context_prompt, tokenizer_a, model_a, DEVICE, MAX_CTX_TOKENS
                    )
                    
                    translated_cache = translate_cache_dynamic(
                        source_keys, source_vals, translators_k, translators_v
                    )
                    
                    loss = compute_ntp_loss(model_b, tokenizer_b, translated_cache, response_text, DEVICE)
                    
                    if loss is not None:
                        loss = loss / STAGE2_GRADIENT_ACCUMULATION
                        scaler.scale(loss).backward()
                        batch_loss += loss.item()
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
            
            running_loss += batch_loss * (STAGE2_GRADIENT_ACCUMULATION / batch_success)
            successful_steps += 1
        else:
            scaler.update()
        
        if (step + 1) % 100 == 0 and successful_steps > 0:
            avg_loss = running_loss / min(100, successful_steps)
            print(f"\n[Stage 2, Step {step+1}] NTP Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
            running_loss = 0.0
            successful_steps = 0
        
        if (step + 1) % EVAL_EVERY == 0:
            avg_f1 = evaluate_translators(
                model_a, model_b, tokenizer_a, tokenizer_b,
                translators_k, translators_v, DEVICE, NUM_FACTS
            )
            eval_history.append(('stage2', step + 1, batch_loss * STAGE2_GRADIENT_ACCUMULATION, avg_f1))
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    final_f1 = evaluate_translators(
        model_a, model_b, tokenizer_a, tokenizer_b,
        translators_k, translators_v, DEVICE, NUM_FACTS
    )
    
    # Print history
    print("\n" + "="*80)
    print("TRAINING HISTORY")
    print("="*80)
    print(f"{'Stage':<8} {'Step':<8} {'Loss':<12} {'F1 Score':<12}")
    print("-" * 45)
    for stage, step, loss, f1 in eval_history:
        print(f"{stage:<8} {step:<8} {loss:<12.4f} {f1:<12.4f}")
    print("="*80)
    
    # Save
    torch.save({
        'translators_k_state_dict': translators_k.state_dict(),
        'translators_v_state_dict': translators_v.state_dict(),
        'eval_history': eval_history,
        'final_f1': final_f1,
        'num_facts': NUM_FACTS,
        'config': {
            'model_a': MODEL_A,
            'model_b': MODEL_B,
            'stage1_steps': STAGE1_STEPS,
            'stage2_steps': STAGE2_STEPS,
        }
    }, SAVE_PATH)
    print(f"\n✅ Saved to {SAVE_PATH}")
    print(f"Final F1: {final_f1:.4f}")


if __name__ == "__main__":
    main()
