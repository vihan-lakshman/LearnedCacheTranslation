#!/usr/bin/env python3
"""
Hybrid KV Cache Translation on SQuAD

Uses the hybrid approach that worked well on synthetic data:
- Translate early layers (0 to cutoff-1) from Model A
- Use Model B's native cache for late layers (cutoff to num_layers-1)

This is the real test - can the hybrid approach work on a real benchmark?

Usage:
    python train_squad_hybrid.py --cutoff 8 --steps 2000
    python train_squad_hybrid.py --cutoff 16 --steps 2000
"""

import argparse
import random
import math
from typing import List, Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, logging, DynamicCache
from datasets import load_dataset
from tqdm import tqdm, trange
import re
import string

logging.set_verbosity_error()


# =============================================================================
# Configuration
# =============================================================================

MODEL_A = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_B = "meta-llama/Llama-3.1-8B-Instruct"


# =============================================================================
# SQuAD Dataset
# =============================================================================

class SQuADDataset:
    """Loads and processes SQuAD for extractive QA."""
    
    def __init__(self, max_context_chars: int = 1000, seed: int = 42):
        self.max_context_chars = max_context_chars
        self.seed = seed
        self.train_data = []
        self.test_data = []
    
    def load(self, train_size: int = 5000, test_size: int = 500):
        """Load SQuAD dataset."""
        print("Loading SQuAD dataset...")
        dataset = load_dataset("squad", split="train")
        
        # Shuffle with seed
        dataset = dataset.shuffle(seed=self.seed)
        
        # Process examples
        all_examples = []
        for item in dataset:
            context = item['context']
            question = item['question']
            answers = item['answers']['text']
            
            if not answers:
                continue
            
            answer = answers[0]  # Take first answer
            
            # Skip if context too long
            if len(context) > self.max_context_chars:
                # Try to find answer position and truncate around it
                ans_pos = context.find(answer)
                if ans_pos == -1:
                    continue
                
                # Keep context around the answer
                start = max(0, ans_pos - self.max_context_chars // 2)
                end = min(len(context), start + self.max_context_chars)
                context = context[start:end]
                
                # Verify answer still in context
                if answer not in context:
                    continue
            
            all_examples.append((context, question, answer))
            
            if len(all_examples) >= train_size + test_size:
                break
        
        # Split into train/test
        self.train_data = all_examples[:train_size]
        self.test_data = all_examples[train_size:train_size + test_size]
        
        print(f"Loaded {len(self.train_data)} train, {len(self.test_data)} test examples")
        
        # Show sample
        if self.train_data:
            ctx, q, a = self.train_data[0]
            print(f"\nSample:")
            print(f"  Context: {ctx[:100]}...")
            print(f"  Question: {q}")
            print(f"  Answer: {a}")
    
    def get_train_batch(self, batch_size: int, rng: random.Random) -> List[Tuple[str, str, str]]:
        samples = rng.choices(self.train_data, k=batch_size)
        return samples
    
    def get_test_data(self) -> List[Tuple[str, str, str]]:
        return self.test_data


# =============================================================================
# Translator
# =============================================================================

class NormalizedTranslator(nn.Module):
    def __init__(self, input_size, output_size, target_heads, target_head_dim):
        super().__init__()
        self.target_heads = target_heads
        self.target_head_dim = target_head_dim
        
        self.register_buffer('source_mean', torch.zeros(input_size))
        self.register_buffer('source_std', torch.ones(input_size))
        self.register_buffer('target_mean', torch.zeros(output_size))
        self.register_buffer('target_std', torch.ones(output_size))
        
        hidden = max(input_size, output_size) * 2
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, output_size),
        )
    
    def set_statistics(self, source_mean, source_std, target_mean, target_std):
        self.source_mean = source_mean
        self.source_std = source_std.clamp(min=1e-6)
        self.target_mean = target_mean
        self.target_std = target_std.clamp(min=1e-6)
    
    def forward(self, x):
        batch, heads, seq_len, head_dim = x.shape
        original_dtype = x.dtype
        
        x_flat = x.float().permute(0, 2, 1, 3).contiguous().view(batch * seq_len, -1)
        x_norm = (x_flat - self.source_mean) / self.source_std
        y_norm = self.net(x_norm)
        y = y_norm * self.target_std + self.target_mean
        y = y.view(batch, seq_len, self.target_heads, self.target_head_dim)
        
        return y.permute(0, 2, 1, 3).contiguous().to(original_dtype)


# =============================================================================
# Utilities
# =============================================================================

def normalize_text(s):
    """Normalize text for evaluation."""
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def calculate_f1(prediction, ground_truth):
    """Calculate F1 score."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    
    if not gt_tokens and not pred_tokens:
        return 1.0
    if not gt_tokens or not pred_tokens:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def calculate_em(prediction, ground_truth):
    """Calculate exact match."""
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def create_cache(keys, vals):
    cache = DynamicCache()
    for i in range(len(keys)):
        cache.update(keys[i], vals[i], i)
    return cache


def generate_with_cache(model, tokenizer, cache, max_tokens, device):
    """Generate text using a pre-filled KV cache."""
    current_ids = tokenizer(" ", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if current_ids.numel() == 0:
        current_ids = torch.tensor([[220]], device=device)
    
    generated = []
    cache_len = cache.get_seq_length()
    
    for _ in range(max_tokens):
        attn_mask = torch.ones(1, cache_len + len(generated) + current_ids.shape[1], device=device)
        
        with torch.no_grad():
            out = model(input_ids=current_ids, attention_mask=attn_mask,
                       past_key_values=cache, use_cache=True)
        
        next_logits = out.logits[:, -1, :].clone()
        if generated:
            for tok in set(generated[-10:]):
                next_logits[0, tok] /= 1.2
        
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        
        if next_id.item() == tokenizer.eos_token_id:
            break
        if len(generated) >= 3 and generated[-1] == generated[-2] == generated[-3]:
            break
        
        generated.append(next_id.item())
        current_ids = next_id
        cache = out.past_key_values
    
    return tokenizer.decode(generated, skip_special_tokens=True) if generated else ""


# =============================================================================
# Hybrid SQuAD Trainer
# =============================================================================

class HybridSQuADTrainer:
    """Hybrid approach for SQuAD: translate early layers, native late layers."""
    
    def __init__(
        self,
        cutoff_layer: int = 8,
        max_ctx_tokens: int = 512,
        train_steps: int = 2000,
        batch_size: int = 4,
        learning_rate: float = 3e-4,
        eval_every: int = 200,
        seed: int = 42,
    ):
        self.cutoff_layer = cutoff_layer
        self.max_ctx_tokens = max_ctx_tokens
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.eval_every = eval_every
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.eval_history = []
    
    def setup(self):
        print(f"\n{'='*70}")
        print(f"Hybrid SQuAD Training (cutoff={self.cutoff_layer})")
        print(f"{'='*70}")
        
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        
        # Load models
        print("\n1. Loading models...")
        self.tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A)
        self.tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B)
        
        if self.tokenizer_a.pad_token is None:
            self.tokenizer_a.pad_token = self.tokenizer_a.eos_token
        if self.tokenizer_b.pad_token is None:
            self.tokenizer_b.pad_token = self.tokenizer_b.eos_token
        
        self.model_a = AutoModelForCausalLM.from_pretrained(
            MODEL_A, torch_dtype=torch.float16).to(self.device)
        self.model_b = AutoModelForCausalLM.from_pretrained(
            MODEL_B, torch_dtype=torch.float16).to(self.device)
        
        self.model_a.eval()
        self.model_b.eval()
        for p in self.model_a.parameters():
            p.requires_grad = False
        for p in self.model_b.parameters():
            p.requires_grad = False
        
        # Get layer counts
        with torch.no_grad():
            sample = self.tokenizer_a(["test"], return_tensors="pt", max_length=32,
                                      padding="max_length", truncation=True).to(self.device)
            out_a = self.model_a(**sample, use_cache=True)
            self.num_layers_a = len(out_a.past_key_values)
            
            sample = self.tokenizer_b(["test"], return_tensors="pt", max_length=32,
                                      padding="max_length", truncation=True).to(self.device)
            out_b = self.model_b(**sample, use_cache=True)
            self.num_layers_b = len(out_b.past_key_values)
        
        print(f"   Model A: {self.num_layers_a} layers")
        print(f"   Model B: {self.num_layers_b} layers")
        
        # Validate cutoff
        if self.cutoff_layer > self.num_layers_b:
            self.cutoff_layer = self.num_layers_b
        
        # Create interpolated layer mapping
        self.layer_mapping = []
        for j in range(self.cutoff_layer):
            src_idx = round(j * (self.num_layers_a - 1) / (self.num_layers_b - 1))
            self.layer_mapping.append(src_idx)
        
        print(f"\n   HYBRID CONFIGURATION:")
        print(f"   - Translated layers: 0 to {self.cutoff_layer - 1} ({self.cutoff_layer} layers, {100*self.cutoff_layer/self.num_layers_b:.0f}%)")
        print(f"   - Native Model B layers: {self.cutoff_layer} to {self.num_layers_b - 1} ({self.num_layers_b - self.cutoff_layer} layers, {100*(self.num_layers_b-self.cutoff_layer)/self.num_layers_b:.0f}%)")
        
        # Load SQuAD dataset
        print("\n2. Loading SQuAD dataset...")
        self.dataset = SQuADDataset(max_context_chars=1500, seed=self.seed)
        self.dataset.load(train_size=5000, test_size=500)
        
        # Create translators for early layers only
        print("\n3. Creating translators...")
        with torch.no_grad():
            sample_a = self.tokenizer_a(["test"], return_tensors="pt", max_length=self.max_ctx_tokens,
                                        padding="max_length", truncation=True).to(self.device)
            sample_b = self.tokenizer_b(["test"], return_tensors="pt", max_length=self.max_ctx_tokens,
                                        padding="max_length", truncation=True).to(self.device)
            out_a = self.model_a(**sample_a, use_cache=True)
            out_b = self.model_b(**sample_b, use_cache=True)
        
        self.translators_k = nn.ModuleList()
        self.translators_v = nn.ModuleList()
        
        for target_idx in range(self.cutoff_layer):
            source_idx = self.layer_mapping[target_idx]
            sk, sv = out_a.past_key_values[source_idx]
            tk, tv = out_b.past_key_values[target_idx]
            
            k_in = sk.shape[1] * sk.shape[3]
            k_out = tk.shape[1] * tk.shape[3]
            
            self.translators_k.append(
                NormalizedTranslator(k_in, k_out, tk.shape[1], tk.shape[3]).to(self.device))
            self.translators_v.append(
                NormalizedTranslator(k_in, k_out, tv.shape[1], tv.shape[3]).to(self.device))
        
        # Calibrate on SQuAD data
        print("\n4. Calibrating on SQuAD...")
        self._calibrate()
        
        total_params = sum(p.numel() for p in self.translators_k.parameters()) + \
                       sum(p.numel() for p in self.translators_v.parameters())
        print(f"\n   Translator parameters: {total_params:,}")
    
    def _calibrate(self):
        """Calibrate normalization statistics on SQuAD data."""
        all_src_k = [[] for _ in range(self.cutoff_layer)]
        all_src_v = [[] for _ in range(self.cutoff_layer)]
        all_tgt_k = [[] for _ in range(self.cutoff_layer)]
        all_tgt_v = [[] for _ in range(self.cutoff_layer)]
        
        calib_data = self.dataset.train_data[:100]
        
        for ctx, q, _ in tqdm(calib_data, desc="Calibrating"):
            prompt = f"Context: {ctx}\n\nQuestion: {q}\n\nAnswer:"
            
            inputs_a = self.tokenizer_a(prompt, return_tensors="pt", padding="max_length",
                                        truncation=True, max_length=self.max_ctx_tokens).to(self.device)
            inputs_b = self.tokenizer_b(prompt, return_tensors="pt", padding="max_length",
                                        truncation=True, max_length=self.max_ctx_tokens).to(self.device)
            
            with torch.no_grad():
                out_a = self.model_a(**inputs_a, use_cache=True)
                out_b = self.model_b(**inputs_b, use_cache=True)
            
            for i in range(self.cutoff_layer):
                src_idx = self.layer_mapping[i]
                sk = out_a.past_key_values[src_idx][0]
                sv = out_a.past_key_values[src_idx][1]
                tk = out_b.past_key_values[i][0]
                tv = out_b.past_key_values[i][1]
                
                all_src_k[i].append(sk.permute(0,2,1,3).reshape(-1, sk.shape[1]*sk.shape[3]).float().cpu())
                all_src_v[i].append(sv.permute(0,2,1,3).reshape(-1, sv.shape[1]*sv.shape[3]).float().cpu())
                all_tgt_k[i].append(tk.permute(0,2,1,3).reshape(-1, tk.shape[1]*tk.shape[3]).float().cpu())
                all_tgt_v[i].append(tv.permute(0,2,1,3).reshape(-1, tv.shape[1]*tv.shape[3]).float().cpu())
        
        for i in range(self.cutoff_layer):
            sk = torch.cat(all_src_k[i], 0)
            sv = torch.cat(all_src_v[i], 0)
            tk = torch.cat(all_tgt_k[i], 0)
            tv = torch.cat(all_tgt_v[i], 0)
            
            self.translators_k[i].set_statistics(
                sk.mean(0).to(self.device), sk.std(0).to(self.device),
                tk.mean(0).to(self.device), tk.std(0).to(self.device))
            self.translators_v[i].set_statistics(
                sv.mean(0).to(self.device), sv.std(0).to(self.device),
                tv.mean(0).to(self.device), tv.std(0).to(self.device))
    
    def _build_hybrid_cache(self, prompt: str) -> DynamicCache:
        """Build hybrid cache: translated early + native late."""
        inputs_a = self.tokenizer_a(prompt, return_tensors="pt", padding="max_length",
                                    truncation=True, max_length=self.max_ctx_tokens).to(self.device)
        
        with torch.no_grad():
            out_a = self.model_a(**inputs_a, use_cache=True)
        
        inputs_b = self.tokenizer_b(prompt, return_tensors="pt", padding="max_length",
                                    truncation=True, max_length=self.max_ctx_tokens).to(self.device)
        
        with torch.no_grad():
            out_b = self.model_b(**inputs_b, use_cache=True)
        
        all_keys = []
        all_vals = []
        
        # Early layers: translated
        for i in range(self.cutoff_layer):
            src_idx = self.layer_mapping[i]
            src_k = out_a.past_key_values[src_idx][0]
            src_v = out_a.past_key_values[src_idx][1]
            
            with torch.no_grad():
                trans_k = self.translators_k[i](src_k)
                trans_v = self.translators_v[i](src_v)
            
            all_keys.append(trans_k)
            all_vals.append(trans_v)
        
        # Late layers: native Model B
        for i in range(self.cutoff_layer, self.num_layers_b):
            all_keys.append(out_b.past_key_values[i][0])
            all_vals.append(out_b.past_key_values[i][1])
        
        return create_cache(all_keys, all_vals)
    
    def _compute_loss(self, prompts: List[str], answers: List[str]) -> torch.Tensor:
        """Compute NTP loss using hybrid cache."""
        inputs_a = self.tokenizer_a(prompts, return_tensors="pt", padding="max_length",
                                    truncation=True, max_length=self.max_ctx_tokens).to(self.device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                out_a = self.model_a(**inputs_a, use_cache=True)
        
        inputs_b = self.tokenizer_b(prompts, return_tensors="pt", padding="max_length",
                                    truncation=True, max_length=self.max_ctx_tokens).to(self.device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                out_b = self.model_b(**inputs_b, use_cache=True)
        
        all_keys = []
        all_vals = []
        
        # Early layers: translated (with gradients)
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
            for i in range(self.cutoff_layer):
                src_idx = self.layer_mapping[i]
                src_k = out_a.past_key_values[src_idx][0]
                src_v = out_a.past_key_values[src_idx][1]
                
                trans_k = self.translators_k[i](src_k)
                trans_v = self.translators_v[i](src_v)
                
                all_keys.append(trans_k)
                all_vals.append(trans_v)
        
        # Late layers: native (no gradients)
        for i in range(self.cutoff_layer, self.num_layers_b):
            all_keys.append(out_b.past_key_values[i][0])
            all_vals.append(out_b.past_key_values[i][1])
        
        cache = create_cache(all_keys, all_vals)
        
        ans_inputs = self.tokenizer_b(answers, return_tensors="pt", padding=True,
                                      truncation=True, max_length=64).to(self.device)
        ans_ids = ans_inputs.input_ids
        ctx_len = cache.get_seq_length()
        
        attn_mask = torch.ones(len(prompts), ctx_len + ans_ids.shape[1], device=self.device)
        pos_ids = torch.arange(ctx_len, ctx_len + ans_ids.shape[1], device=self.device).unsqueeze(0).expand(len(prompts), -1)
        
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
            out = self.model_b(input_ids=ans_ids, attention_mask=attn_mask,
                              position_ids=pos_ids, past_key_values=cache, use_cache=False)
        
        logits = out.logits[:, :-1, :].contiguous()
        labels = ans_ids[:, 1:].contiguous()
        
        return nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    def evaluate(self, num_samples: int = None) -> Tuple[float, float]:
        """Evaluate on SQuAD test set."""
        self.translators_k.eval()
        self.translators_v.eval()
        
        test_data = self.dataset.get_test_data()
        if num_samples:
            test_data = test_data[:num_samples]
        
        total_f1 = 0
        total_em = 0
        
        for i, (ctx, q, gt) in enumerate(tqdm(test_data, desc="Evaluating")):
            prompt = f"Context: {ctx}\n\nQuestion: {q}\n\nAnswer:"
            
            try:
                cache = self._build_hybrid_cache(prompt)
                response = generate_with_cache(self.model_b, self.tokenizer_b, cache, 50, self.device)
                
                # Clean response
                cleaned = response.strip()
                if '\n' in cleaned:
                    cleaned = cleaned.split('\n')[0]
                for end in ['.', '!', '?']:
                    if end in cleaned:
                        cleaned = cleaned.split(end)[0]
                cleaned = cleaned.strip()
                
                f1 = calculate_f1(cleaned, gt)
                em = calculate_em(cleaned, gt)
                
                total_f1 += f1
                total_em += em
                
                if i < 3:
                    print(f"\n[{i+1}] Q: {q[:60]}...")
                    print(f"     GT: {gt}")
                    print(f"     Pred: {cleaned[:60]}")
                    print(f"     F1: {f1:.2f}, EM: {em:.0f}")
                
            except Exception as e:
                if i < 3:
                    print(f"\n[{i+1}] Error: {e}")
        
        avg_f1 = total_f1 / len(test_data)
        avg_em = total_em / len(test_data)
        
        print(f"\n   Test F1: {avg_f1:.4f}, EM: {avg_em:.4f}")
        return avg_f1, avg_em
    
    def train(self):
        print(f"\n{'='*70}")
        print(f"Training Hybrid on SQuAD (cutoff={self.cutoff_layer})")
        print(f"{'='*70}")
        
        params = list(self.translators_k.parameters()) + list(self.translators_v.parameters())
        optimizer = optim.AdamW(params, lr=self.learning_rate, weight_decay=0.01)
        
        def lr_lambda(step):
            warmup = 200
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / (self.train_steps - warmup)
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = torch.amp.GradScaler(enabled=self.device == "cuda")
        
        train_rng = random.Random(self.seed + 1000)
        running_loss = 0
        best_f1 = 0
        
        for step in trange(self.train_steps, desc="Training"):
            self.translators_k.train()
            self.translators_v.train()
            
            batch = self.dataset.get_train_batch(self.batch_size, train_rng)
            prompts = [f"Context: {ctx}\n\nQuestion: {q}\n\nAnswer:" for ctx, q, _ in batch]
            answers = [" " + a for _, _, a in batch]
            
            optimizer.zero_grad()
            
            try:
                loss = self._compute_loss(prompts, answers)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item()
            except Exception as e:
                print(f"\nStep {step+1} failed: {e}")
                continue
            
            scheduler.step()
            
            if (step + 1) % 100 == 0:
                print(f"\n[{step+1}] Loss: {running_loss/100:.4f}, LR: {scheduler.get_last_lr()[0]:.5f}")
                running_loss = 0
            
            if (step + 1) % self.eval_every == 0:
                torch.cuda.empty_cache()
                f1, em = self.evaluate(num_samples=100)
                self.eval_history.append((step + 1, f1, em))
                
                if f1 > best_f1:
                    best_f1 = f1
                    self.save_checkpoint(f"squad_hybrid_cutoff{self.cutoff_layer}_best.pth")
                
                torch.cuda.empty_cache()
        
        print(f"\n{'='*70}")
        print(f"Training complete! Best F1: {best_f1:.4f}")
        print(f"{'='*70}")
        
        return best_f1
    
    def save_checkpoint(self, path: str):
        torch.save({
            'translators_k': self.translators_k.state_dict(),
            'translators_v': self.translators_v.state_dict(),
            'layer_mapping': self.layer_mapping,
            'cutoff_layer': self.cutoff_layer,
            'eval_history': self.eval_history,
        }, path)
        print(f"   Saved: {path}")
    
    def run_baseline(self) -> Tuple[float, float]:
        """Run baseline: Model B native on SQuAD."""
        print(f"\n{'='*70}")
        print("Baseline (Model B native)")
        print(f"{'='*70}")
        
        test_data = self.dataset.get_test_data()[:100]
        
        total_f1 = 0
        total_em = 0
        
        for i, (ctx, q, gt) in enumerate(tqdm(test_data, desc="Baseline")):
            prompt = f"Context: {ctx}\n\nQuestion: {q}\n\nAnswer:"
            inputs = self.tokenizer_b(prompt, return_tensors="pt", 
                                      truncation=True, max_length=self.max_ctx_tokens).to(self.device)
            
            with torch.no_grad():
                outputs = self.model_b.generate(
                    **inputs, max_new_tokens=50, do_sample=False,
                    pad_token_id=self.tokenizer_b.eos_token_id
                )
            
            response = self.tokenizer_b.decode(outputs[0][inputs.input_ids.shape[1]:], 
                                               skip_special_tokens=True)
            
            cleaned = response.strip()
            if '\n' in cleaned:
                cleaned = cleaned.split('\n')[0]
            for end in ['.', '!', '?']:
                if end in cleaned:
                    cleaned = cleaned.split(end)[0]
            cleaned = cleaned.strip()
            
            f1 = calculate_f1(cleaned, gt)
            em = calculate_em(cleaned, gt)
            
            total_f1 += f1
            total_em += em
            
            if i < 3:
                print(f"\n[{i+1}] Q: {q[:60]}...")
                print(f"     GT: {gt}")
                print(f"     Pred: {cleaned[:60]}")
                print(f"     F1: {f1:.2f}, EM: {em:.0f}")
        
        avg_f1 = total_f1 / len(test_data)
        avg_em = total_em / len(test_data)
        
        print(f"\nBaseline F1: {avg_f1:.4f}, EM: {avg_em:.4f}")
        return avg_f1, avg_em


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=8, help="Layer cutoff")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    trainer = HybridSQuADTrainer(
        cutoff_layer=args.cutoff,
        train_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_every=args.eval_every,
        seed=args.seed,
    )
    
    trainer.setup()
    
    # Run baseline
    baseline_f1, baseline_em = trainer.run_baseline()
    
    # Train
    best_f1 = trainer.train()
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")
    final_f1, final_em = trainer.evaluate()
    
    # Summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Cutoff: {args.cutoff} (translate layers 0-{args.cutoff-1})")
    print(f"  Native layers: {args.cutoff}-31")
    print(f"\nPerformance:")
    print(f"  Baseline F1: {baseline_f1:.4f}, EM: {baseline_em:.4f}")
    print(f"  Best Hybrid F1: {best_f1:.4f}")
    print(f"  Final Hybrid F1: {final_f1:.4f}, EM: {final_em:.4f}")
    print(f"  Recovery: {final_f1/baseline_f1*100:.1f}%")


if __name__ == "__main__":
    main()
