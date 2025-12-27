#!/usr/bin/env python3
"""
Hybrid KV Cache Translation - Version 5 (CORRECT Implementation)

This implements TRUE hybrid KV cache translation with causal conditioning:
1. Model A generates KV cache for prompt
2. Translate first K layers to Model B format (with gradients)
3. Build HybridCache: translated early + empty late
4. Run Model B forward with this cache:
   - Early layers: attention uses translated KV (gradients flow!)
   - Late layers: compute KV from hidden states affected by translated attention
5. Forward on answer tokens, NTP loss
6. Backprop to update translators

CRITICAL: Do NOT use GradScaler - it causes NaN gradients with HybridCache!
Autocast alone works fine for memory efficiency.

Usage:
    python train_squad_hybrid_v5.py --cutoff 12 --steps 2000 --batch 2 --max_len 384
"""

import argparse
import random
import math
from typing import List, Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, logging, DynamicCache
from datasets import load_dataset
from tqdm import tqdm, trange
import re
import string

logging.set_verbosity_error()

MODEL_A = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_B = "meta-llama/Llama-3.1-8B-Instruct"


def create_cache(keys, vals):
    """Helper to create DynamicCache from lists of keys and values."""
    cache = DynamicCache()
    for i in range(len(keys)):
        cache.update(keys[i], vals[i], i)
    return cache


class HybridCache(DynamicCache):
    """
    Cache that uses translated KV for early layers during prefill,
    then concatenates normally during generation.
    """
    
    def __init__(self, translated_kv: List[Tuple[torch.Tensor, torch.Tensor]], cutoff: int):
        super().__init__()
        self.translated_kv = translated_kv
        self.cutoff = cutoff
        self._prefill_done = [False] * cutoff
        
        # Pre-populate early layers with translated KV using the proper API
        for i in range(cutoff):
            k, v = translated_kv[i]
            super().update(k, v, i)
    
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx < self.cutoff:
            if not self._prefill_done[layer_idx]:
                # First call (prefill): return translated KV, ignore computed
                self._prefill_done[layer_idx] = True
                # Return the already-stored translated KV
                return self.translated_kv[layer_idx]
            else:
                # Subsequent calls (generation): use normal update (concatenates)
                return super().update(key_states, value_states, layer_idx, cache_kwargs)
        else:
            # Late layer: normal behavior
            return super().update(key_states, value_states, layer_idx, cache_kwargs)


# =============================================================================
# Simple KV Translator
# =============================================================================

class Translator(nn.Module):
    """Translates KV from Model A to Model B format (matches V3)."""
    
    def __init__(self, src_heads, src_dim, tgt_heads, tgt_dim):
        super().__init__()
        in_dim = src_heads * src_dim
        out_dim = tgt_heads * tgt_dim
        hidden = max(in_dim, out_dim) * 2  # V3 uses max, not min
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim)
        )
        self.tgt_heads = tgt_heads
        self.tgt_dim = tgt_dim
        
        # Normalization stats
        self.register_buffer('src_mean', torch.zeros(in_dim))
        self.register_buffer('src_std', torch.ones(in_dim))
        self.register_buffer('tgt_mean', torch.zeros(out_dim))
        self.register_buffer('tgt_std', torch.ones(out_dim))
    
    def set_stats(self, src_mean, src_std, tgt_mean, tgt_std):
        self.src_mean = src_mean
        self.src_std = src_std.clamp(min=1e-6)
        self.tgt_mean = tgt_mean
        self.tgt_std = tgt_std.clamp(min=1e-6)
    
    def forward(self, x):
        # x: [batch, heads, seq, dim]
        batch, heads, seq_len, head_dim = x.shape
        original_dtype = x.dtype
        
        x_flat = x.float().permute(0, 2, 1, 3).contiguous().view(batch * seq_len, -1)
        x_norm = (x_flat - self.src_mean) / self.src_std
        y_norm = self.net(x_norm)
        y = y_norm * self.tgt_std + self.tgt_mean
        y = y.view(batch, seq_len, self.tgt_heads, self.tgt_dim)
        
        return y.permute(0, 2, 1, 3).contiguous().to(original_dtype)


# =============================================================================
# Metrics
# =============================================================================

def normalize(s):
    s = s.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())

def f1_score(pred, gold):
    pred_tok, gold_tok = normalize(pred).split(), normalize(gold).split()
    if not gold_tok: return float(not pred_tok)
    if not pred_tok: return 0.0
    common = sum((Counter(pred_tok) & Counter(gold_tok)).values())
    if common == 0: return 0.0
    p, r = common / len(pred_tok), common / len(gold_tok)
    return 2 * p * r / (p + r)

def em_score(pred, gold):
    return float(normalize(pred) == normalize(gold))


# =============================================================================
# Trainer
# =============================================================================

class HybridTrainer:
    def __init__(self, cutoff=12, max_len=384, steps=2000, batch=2, lr=3e-4, eval_every=400):
        self.cutoff = cutoff
        self.max_len = max_len
        self.steps = steps
        self.batch = batch
        self.lr = lr
        self.eval_every = eval_every
        self.device = "cuda"
    
    def setup(self):
        print(f"\n{'='*60}")
        print(f"Hybrid Training V5 - CORRECT Implementation (cutoff={self.cutoff})")
        print(f"{'='*60}")
        
        # Set seeds like V3
        torch.manual_seed(42)
        random.seed(42)
        
        # Tokenizers (match V3)
        self.tok_a = AutoTokenizer.from_pretrained(MODEL_A)
        self.tok_b = AutoTokenizer.from_pretrained(MODEL_B)
        if self.tok_a.pad_token is None:
            self.tok_a.pad_token = self.tok_a.eos_token
        if self.tok_b.pad_token is None:
            self.tok_b.pad_token = self.tok_b.eos_token
        
        # Models - match V3's loading pattern
        print("Loading models...")
        self.model_a = AutoModelForCausalLM.from_pretrained(
            MODEL_A, torch_dtype=torch.float16).to(self.device)
        self.model_b = AutoModelForCausalLM.from_pretrained(
            MODEL_B, torch_dtype=torch.float16).to(self.device)
        self.model_a.eval()
        self.model_b.eval()
        
        # Freeze model parameters (critical for memory!)
        for p in self.model_a.parameters():
            p.requires_grad = False
        for p in self.model_b.parameters():
            p.requires_grad = False
        
        # Enable gradient checkpointing to save memory during backprop
        self.model_b.gradient_checkpointing_enable()
        
        # Dimensions
        cfg_a, cfg_b = self.model_a.config, self.model_b.config
        self.src_heads = cfg_a.num_key_value_heads
        self.src_dim = cfg_a.hidden_size // cfg_a.num_attention_heads
        self.tgt_heads = cfg_b.num_key_value_heads
        self.tgt_dim = cfg_b.hidden_size // cfg_b.num_attention_heads
        self.num_layers_b = cfg_b.num_hidden_layers
        
        print(f"Model B has {self.num_layers_b} layers")
        
        # Layer mapping: Model A (28 layers) -> Model B (32 layers)
        # Match V3's formula exactly
        num_layers_a = cfg_a.num_hidden_layers
        self.layer_map = [round(i * (num_layers_a - 1) / (self.num_layers_b - 1)) 
                         for i in range(self.cutoff)]
        print(f"Layer mapping: B[0:{self.cutoff}] <- A{self.layer_map}")
        
        # Translators
        print("Creating translators...")
        self.trans_k = nn.ModuleList([
            Translator(self.src_heads, self.src_dim, self.tgt_heads, self.tgt_dim)
            for _ in range(self.cutoff)
        ]).to(self.device)
        self.trans_v = nn.ModuleList([
            Translator(self.src_heads, self.src_dim, self.tgt_heads, self.tgt_dim)
            for _ in range(self.cutoff)
        ]).to(self.device)
        
        self.opt = optim.AdamW(
            list(self.trans_k.parameters()) + list(self.trans_v.parameters()), 
            lr=self.lr, weight_decay=0.01
        )
        # NOTE: We do NOT use GradScaler - it causes NaN gradients with HybridCache
        # Autocast alone works fine for memory efficiency
        
        # Data
        print("Loading SQuAD...")
        ds = load_dataset("squad", split="train").shuffle(seed=42)
        self.train_data, self.test_data = [], []
        for item in ds:
            if not item['answers']['text']: continue
            ctx = item['context'][:800]  # Fits in max_len=384
            self.train_data.append({'ctx': ctx, 'q': item['question'], 'a': item['answers']['text'][0]})
            if len(self.train_data) >= 5500: break
        self.test_data = self.train_data[5000:]
        self.train_data = self.train_data[:5000]
        print(f"Train: {len(self.train_data)}, Test: {len(self.test_data)}")
        
        # Compute stats
        self._compute_stats()
    
    def _compute_stats(self):
        """Compute normalization statistics (matches V3's _calibrate)."""
        print("Calibrating...")
        
        src_k = [[] for _ in range(self.cutoff)]
        src_v = [[] for _ in range(self.cutoff)]
        tgt_k = [[] for _ in range(self.cutoff)]
        tgt_v = [[] for _ in range(self.cutoff)]
        
        calib_data = self.train_data[:100]
        
        for ex in tqdm(calib_data, desc="Calibrating"):
            prompt = self._prompt(ex)
            
            inp_a = self.tok_a(prompt, return_tensors="pt", padding="max_length",
                              truncation=True, max_length=self.max_len).to(self.device)
            inp_b = self.tok_b(prompt, return_tensors="pt", padding="max_length",
                              truncation=True, max_length=self.max_len).to(self.device)
            
            with torch.no_grad():
                out_a = self.model_a(**inp_a, use_cache=True)
                out_b = self.model_b(**inp_b, use_cache=True)
            
            for i in range(self.cutoff):
                src_idx = self.layer_map[i]
                sk = out_a.past_key_values[src_idx][0]
                sv = out_a.past_key_values[src_idx][1]
                tk = out_b.past_key_values[i][0]
                tv = out_b.past_key_values[i][1]
                
                src_k[i].append(sk.permute(0,2,1,3).reshape(-1, sk.shape[1]*sk.shape[3]).float().cpu())
                src_v[i].append(sv.permute(0,2,1,3).reshape(-1, sv.shape[1]*sv.shape[3]).float().cpu())
                tgt_k[i].append(tk.permute(0,2,1,3).reshape(-1, tk.shape[1]*tk.shape[3]).float().cpu())
                tgt_v[i].append(tv.permute(0,2,1,3).reshape(-1, tv.shape[1]*tv.shape[3]).float().cpu())
        
        for i in range(self.cutoff):
            sk = torch.cat(src_k[i], 0)
            sv = torch.cat(src_v[i], 0)
            tk = torch.cat(tgt_k[i], 0)
            tv = torch.cat(tgt_v[i], 0)
            
            self.trans_k[i].set_stats(
                sk.mean(0).to(self.device), sk.std(0).to(self.device),
                tk.mean(0).to(self.device), tk.std(0).to(self.device))
            self.trans_v[i].set_stats(
                sv.mean(0).to(self.device), sv.std(0).to(self.device),
                tv.mean(0).to(self.device), tv.std(0).to(self.device))
    
    def _prompt(self, ex):
        return f"Context: {ex['ctx']}\n\nQuestion: {ex['q']}\n\nAnswer:"
    
    def _get_translated_kv(self, prompts: List[str]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get Model A's KV and translate early layers (matches V3)."""
        inp = self.tok_a(prompts, return_tensors="pt", padding="max_length", 
                        truncation=True, max_length=self.max_len).to(self.device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                out_a = self.model_a(**inp, use_cache=True)
        
        translated = []
        for i in range(self.cutoff):
            src_k = out_a.past_key_values[self.layer_map[i]][0]
            src_v = out_a.past_key_values[self.layer_map[i]][1]
            trans_k = self.trans_k[i](src_k)
            trans_v = self.trans_v[i](src_v)
            translated.append((trans_k, trans_v))
        
        return translated
    
    def _forward_hybrid(self, prompts: List[str], translated_kv: List[Tuple]) -> Tuple[None, DynamicCache]:
        """
        CORRECT hybrid forward:
        - Build HybridCache with translated KV for early layers
        - Run Model B forward (late layers compute from affected hidden states)
        """
        inp = self.tok_b(prompts, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=self.max_len).to(self.device)
        
        cache = HybridCache(translated_kv, self.cutoff)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                out_b = self.model_b(
                    input_ids=inp.input_ids,
                    attention_mask=inp.attention_mask,
                    past_key_values=cache,
                    use_cache=True
                )
        
        return None, out_b.past_key_values
    
    def _compute_loss(self, prompts: List[str], answers: List[str]) -> torch.Tensor:
        """
        CORRECT training with gradient checkpointing:
        1. Translate early KV (with gradients)
        2. Build HybridCache with translated early KV
        3. Run Model B prefill (WITH gradients, using checkpointing to save memory)
           - Early layers: attention uses translated KV from cache (gradients flow!)
           - Late layers: compute KV from hidden states affected by translated attention
        4. Forward on answer tokens
        5. NTP loss -> gradients flow through everything -> translators
        """
        # Step 1: Get translated KV with gradients
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
            translated_kv = self._get_translated_kv(prompts)
        
        # Step 2: Build hybrid cache
        cache = HybridCache(translated_kv, self.cutoff)
        
        # Step 3: Prefill WITH gradients (checkpointing enabled in setup)
        inp = self.tok_b(prompts, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=self.max_len).to(self.device)
        
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
            out_prefill = self.model_b(
                input_ids=inp.input_ids,
                attention_mask=inp.attention_mask,
                past_key_values=cache,
                use_cache=True
            )
        cache = out_prefill.past_key_values
        
        # Step 4: Forward on answer tokens
        ans_inputs = self.tok_b(
            answers, return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(self.device)
        ans_ids = ans_inputs.input_ids
        
        ctx_len = cache.get_seq_length()
        attn_mask = torch.ones(len(prompts), ctx_len + ans_ids.shape[1], device=self.device)
        pos_ids = torch.arange(ctx_len, ctx_len + ans_ids.shape[1], device=self.device)
        pos_ids = pos_ids.unsqueeze(0).expand(len(prompts), -1)
        
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
            out = self.model_b(
                input_ids=ans_ids,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_values=cache,
                use_cache=False
            )
        
        # Step 5: NTP loss
        logits = out.logits[:, :-1, :].contiguous()
        labels = ans_ids[:, 1:].contiguous()
        
        return nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    def _generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate using CORRECT hybrid cache."""
        with torch.no_grad():
            # Get translated KV
            translated_kv = self._get_translated_kv([prompt])
            
            # Run hybrid prefill
            inp = self.tok_b(prompt, return_tensors="pt", padding="max_length",
                            truncation=True, max_length=self.max_len).to(self.device)
            
            cache = HybridCache(translated_kv, self.cutoff)
            
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                out = self.model_b(
                    input_ids=inp.input_ids,
                    attention_mask=inp.attention_mask,
                    past_key_values=cache,
                    use_cache=True
                )
            cache = out.past_key_values
        
        # Generate token by token
        curr = self.tok_b(" ", return_tensors="pt", add_special_tokens=False).input_ids
        if curr.numel() == 0:
            curr = torch.tensor([[220]])  # space token
        curr = curr.to(self.device)
        
        generated = []
        cache_len = cache.get_seq_length()
        
        for _ in range(max_tokens):
            attn = torch.ones(1, cache_len + len(generated) + curr.shape[1], device=self.device)
            
            with torch.no_grad():
                out = self.model_b(
                    input_ids=curr,
                    attention_mask=attn,
                    past_key_values=cache,
                    use_cache=True
                )
            
            next_logits = out.logits[:, -1, :].clone()
            
            # Repetition penalty
            if generated:
                for tok in set(generated[-10:]):
                    next_logits[0, tok] /= 1.2
            
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            
            if next_id.item() == self.tok_b.eos_token_id:
                break
            if len(generated) >= 3 and generated[-1] == generated[-2] == generated[-3]:
                break
            
            generated.append(next_id.item())
            curr = next_id
            cache = out.past_key_values
        
        return self.tok_b.decode(generated, skip_special_tokens=True) if generated else ""
    
    def evaluate(self, n=50) -> dict:
        """Evaluate hybrid approach."""
        self.trans_k.eval()
        self.trans_v.eval()
        
        samples = random.sample(self.test_data, min(n, len(self.test_data)))
        f1s, ems = [], []
        
        for ex in tqdm(samples, desc="Eval", leave=False):
            pred = self._generate(self._prompt(ex)).strip().split('\n')[0]
            f1s.append(f1_score(pred, ex['a']))
            ems.append(em_score(pred, ex['a']))
        
        self.trans_k.train()
        self.trans_v.train()
        
        return {'f1': sum(f1s)/len(f1s), 'em': sum(ems)/len(ems)}
    
    def baseline(self, n=50) -> dict:
        """Native Model B baseline (matches V3)."""
        print("Computing baseline...")
        samples = random.sample(self.test_data, min(n, len(self.test_data)))
        f1s, ems = [], []
        
        for ex in tqdm(samples, desc="Baseline"):
            prompt = self._prompt(ex)
            inp = self.tok_b(prompt, return_tensors="pt", truncation=True, 
                            max_length=self.max_len).to(self.device)
            
            with torch.no_grad():
                out = self.model_b.generate(**inp, max_new_tokens=50, do_sample=False,
                                            pad_token_id=self.tok_b.eos_token_id)
            
            pred = self.tok_b.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True)
            pred = pred.strip().split('\n')[0]
            f1s.append(f1_score(pred, ex['a']))
            ems.append(em_score(pred, ex['a']))
        
        return {'f1': sum(f1s)/len(f1s), 'em': sum(ems)/len(ems)}
    
    def train(self):
        """Main training loop (matches V3)."""
        print(f"\n{'='*60}")
        print(f"Hybrid Training V5 CORRECT on SQuAD (cutoff={self.cutoff})")
        print(f"{'='*60}")
        
        params = list(self.trans_k.parameters()) + list(self.trans_v.parameters())
        
        def lr_lambda(step):
            warmup = 200
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / (self.steps - warmup)
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
        
        scheduler = optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        
        history = []
        running_loss = 0
        
        for step in trange(self.steps, desc="Training"):
            self.trans_k.train()
            self.trans_v.train()
            
            batch = random.sample(self.train_data, self.batch)
            prompts = [self._prompt(ex) for ex in batch]
            answers = [" " + ex['a'] for ex in batch]
            
            self.opt.zero_grad()
            
            try:
                loss = self._compute_loss(prompts, answers)
                
                # Regular backward without scaler (scaler causes NaN gradients)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                self.opt.step()
                
                running_loss += loss.item()
            except Exception as e:
                print(f"\nStep {step+1} failed: {e}")
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue
            
            if (step + 1) % 50 == 0:
                torch.cuda.empty_cache()
            
            scheduler.step()
            
            if (step + 1) % 100 == 0:
                print(f"\n[{step+1}] Loss: {running_loss/100:.4f}, LR: {scheduler.get_last_lr()[0]:.5f}")
                running_loss = 0
            
            if (step + 1) % self.eval_every == 0:
                torch.cuda.empty_cache()
                metrics = self.evaluate()
                history.append({'step': step+1, **metrics})
                print(f"  F1={metrics['f1']:.4f}, EM={metrics['em']:.4f}")
                torch.cuda.empty_cache()
        
        return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff', type=int, default=12)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eval_every', type=int, default=400)
    parser.add_argument('--max_len', type=int, default=384)
    args = parser.parse_args()
    
    trainer = HybridTrainer(
        cutoff=args.cutoff, steps=args.steps, batch=args.batch,
        lr=args.lr, eval_every=args.eval_every, max_len=args.max_len
    )
    trainer.setup()
    
    base = trainer.baseline()
    print(f"\nBaseline: F1={base['f1']:.4f}, EM={base['em']:.4f}")
    
    history = trainer.train()
    
    print(f"\n{'='*60}")
    print(f"RESULTS (V5 CORRECT, cutoff={args.cutoff})")
    print(f"{'='*60}")
    print(f"Baseline F1: {base['f1']:.4f}")
    if history:
        best = max(history, key=lambda x: x['f1'])
        print(f"Best F1: {best['f1']:.4f} (step {best['step']})")
        print(f"Final F1: {history[-1]['f1']:.4f}")
        print(f"Recovery: {100*history[-1]['f1']/base['f1']:.1f}%")


if __name__ == "__main__":
    main()
