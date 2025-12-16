#!/usr/bin/env python3
"""
KV Cache Translation Ablation Study

Systematically compares:
1. Model families: Qwen, Llama
2. Context complexity: 1, 2, 3, 4 facts, or mixed
3. Loss functions: MSE, NTP, or Curriculum (MSE→NTP)
4. Normalization: enabled or disabled

Usage:
    python ablation_study.py --model qwen --facts 2 --loss ntp --normalize
    python ablation_study.py --model llama --facts 4 --loss ntp --normalize
    python ablation_study.py --model mistral --facts mixed --loss curriculum --no-normalize

All combinations for a model family:
    python ablation_study.py --model llama --run-all
"""

import argparse
import random
import math
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, logging, DynamicCache
from tqdm import tqdm, trange
import re
import string
from collections import Counter

logging.set_verbosity_error()


# =============================================================================
# Model Configurations
# =============================================================================

MODEL_CONFIGS = {
    "qwen": {
        "model_a": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_b": "Qwen/Qwen2.5-7B-Instruct",
        "num_layers": 28,
        "description": "Qwen2.5 1.5B → 7B",
    },
    "qwen-small": {
        "model_a": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_b": "Qwen/Qwen2.5-1.5B-Instruct", 
        "num_layers": 28,
        "description": "Qwen2.5 0.5B → 1.5B",
    },
    "llama": {
        "model_a": "meta-llama/Llama-3.2-1B-Instruct",
        "model_b": "meta-llama/Llama-3.2-3B-Instruct",
        "num_layers": 16,  # Llama-3.2-1B has 16 layers
        "description": "Llama-3.2 1B → 3B",
    },
    "llama-large": {
        "model_a": "meta-llama/Llama-3.2-3B-Instruct",
        "model_b": "meta-llama/Llama-3.1-8B-Instruct",
        "num_layers": 28,  # Llama-3.2-3B has 28 layers
        "description": "Llama-3.2 3B → Llama-3.1 8B",
    },
    "mistral": {
        "model_a": "mistralai/Mistral-7B-Instruct-v0.2",
        "model_b": "mistralai/Mistral-7B-Instruct-v0.3",
        "num_layers": 32,
        "description": "Mistral-7B v0.2 → v0.3 (same size, different versions)",
    },
    # Cross-family (experimental)
    "cross-qwen-llama": {
        "model_a": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_b": "meta-llama/Llama-3.2-3B-Instruct",
        "num_layers": 28,  # Use smaller of the two
        "description": "Qwen2.5 1.5B → Llama-3.2 3B (cross-family)",
    },
}


# Layer alignment strategies
LAYER_ALIGN_TERMINAL = "terminal"  # Align from end (deeper layers)
LAYER_ALIGN_INTERP = "interpolate"  # Map all source to all target via interpolation


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    # Model settings
    model_family: str = "qwen"
    model_a: str = ""
    model_b: str = ""
    num_layers: int = 28
    
    # Data settings
    num_facts: str = "2"  # "1", "2", "3", "4", or "mixed"
    max_ctx_tokens: int = 160  # Enough for 4 facts
    train_samples: int = 10000  # Pre-generated training samples
    test_samples: int = 500     # Held-out test samples
    
    # Training settings
    train_steps: int = 4000
    batch_size: int = 8
    learning_rate: float = 1e-3
    warmup_steps: int = 200
    
    # Loss settings: "mse", "ntp", "curriculum"
    loss_type: str = "ntp"
    
    # Curriculum settings (only used if loss_type == "curriculum")
    mse_weight_start: float = 1.0
    mse_weight_end: float = 0.1
    ntp_weight_start: float = 0.1
    ntp_weight_end: float = 1.0
    
    # Normalization
    use_normalization: bool = True
    calibration_samples: int = 100
    
    # Layer alignment strategy: "terminal" or "interpolate"
    layer_alignment: str = "terminal"
    
    # Evaluation
    eval_every: int = 500
    max_new_tokens: int = 32
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 3  # Stop if no improvement for this many evals
    
    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    def __post_init__(self):
        """Set model paths based on model_family if not already set."""
        if not self.model_a or not self.model_b:
            if self.model_family in MODEL_CONFIGS:
                config = MODEL_CONFIGS[self.model_family]
                self.model_a = config["model_a"]
                self.model_b = config["model_b"]
                self.num_layers = config["num_layers"]
            else:
                raise ValueError(f"Unknown model family: {self.model_family}. "
                               f"Available: {list(MODEL_CONFIGS.keys())}")
    
    def get_name(self) -> str:
        """Generate experiment name from config."""
        norm_str = "norm" if self.use_normalization else "nonorm"
        return f"{self.model_family}_facts_{self.num_facts}_loss_{self.loss_type}_{norm_str}"


# =============================================================================
# Data Generation with Train/Test Split
# =============================================================================

class SyntheticQADataset:
    """
    Generates synthetic QA data with guaranteed train/test separation.
    
    Uses different random seeds for train vs test to ensure no leakage.
    Also tracks which fact combinations have been used.
    """
    
    SUBJECTS = [
        "Dr. Evans", "The pilot", "The scientist", "Agent Helix", "The cartographer",
        "Captain Rivera", "Professor Kim", "Detective Chen", "Colonel Hayes", "Dr. Martinez",
        "Ambassador Lee", "Inspector Walsh", "Commander Singh", "Dr. Patel", "Agent Morrison",
        "Professor Anderson", "Captain Brooks", "Detective Quinn", "Colonel Foster", "Dr. Zhang",
        "The engineer", "The diplomat", "The analyst", "The curator", "The technician",
    ]
    
    ACTIONS = [
        "placed", "hid", "secured", "delivered", "found",
        "stored", "moved", "retrieved", "archived", "discovered",
        "transferred", "collected", "obtained", "examined", "acquired",
    ]
    
    OBJECTS = [
        "the blue folder", "the silver key", "the encrypted drive", "the heavy package", "the coded map",
        "the sealed envelope", "the metal case", "the classified document", "the ancient artifact", "the prototype device",
        "the red binder", "the brass medallion", "the backup drive", "the wooden crate", "the detailed blueprint",
    ]
    
    LOCATIONS = [
        "in the library", "in the hangar", "in the laboratory", "at the north gate", "behind the console",
        "in the archive room", "at the main entrance", "in the storage facility", "near the fountain", "in the vault",
        "in the control room", "at the south entrance", "in the research wing", "near the statue", "in the safe",
    ]
    
    def __init__(self, seed: int = 42):
        self.base_seed = seed
        self.train_data: List[Tuple[str, str, str, int]] = []  # (context, question, answer, num_facts)
        self.test_data: List[Tuple[str, str, str, int]] = []
    
    def _generate_fact(self, rng: random.Random) -> dict:
        """Generate a single fact using the provided RNG."""
        return {
            'subject': rng.choice(self.SUBJECTS),
            'action': rng.choice(self.ACTIONS),
            'object': rng.choice(self.OBJECTS),
            'location': rng.choice(self.LOCATIONS),
        }
    
    def _fact_to_text(self, fact: dict) -> str:
        return f"{fact['subject']} {fact['action']} {fact['object']} {fact['location']}."
    
    def _generate_qa(self, rng: random.Random, num_facts: int) -> Tuple[str, str, str]:
        """Generate a single QA pair."""
        facts = [self._generate_fact(rng) for _ in range(num_facts)]
        context = " ".join([self._fact_to_text(f) for f in facts])
        target = rng.choice(facts)
        
        q_type = rng.choice(["who", "what", "where"])
        if q_type == "who":
            question = f"Who {target['action']} {target['object']}?"
            answer = target['subject']
        elif q_type == "what":
            question = f"What did {target['subject']} {target['action']}?"
            answer = target['object']
        else:
            question = f"Where was {target['object']} {target['action']}?"
            answer = target['location']
        
        return context, question, answer
    
    def generate_datasets(
        self, 
        num_facts_config: str,
        train_size: int, 
        test_size: int
    ):
        """
        Generate train and test datasets with no overlap.
        
        Args:
            num_facts_config: "1", "2", "3", "4", or "mixed"
            train_size: Number of training samples
            test_size: Number of test samples
        """
        # Use different seeds for train and test
        train_rng = random.Random(self.base_seed)
        test_rng = random.Random(self.base_seed + 10000)  # Different seed for test
        
        def get_num_facts(rng: random.Random) -> int:
            if num_facts_config == "mixed":
                return rng.choice([1, 2, 3, 4])
            return int(num_facts_config)
        
        # Generate training data
        self.train_data = []
        for _ in range(train_size):
            nf = get_num_facts(train_rng)
            ctx, q, a = self._generate_qa(train_rng, nf)
            self.train_data.append((ctx, q, a, nf))
        
        # Generate test data (with different seed)
        self.test_data = []
        for _ in range(test_size):
            nf = get_num_facts(test_rng)
            ctx, q, a = self._generate_qa(test_rng, nf)
            self.test_data.append((ctx, q, a, nf))
        
        print(f"Generated {len(self.train_data)} train, {len(self.test_data)} test samples")
        
        # Verify no overlap (sample check)
        train_contexts = set(d[0] for d in self.train_data[:1000])
        test_contexts = set(d[0] for d in self.test_data)
        overlap = train_contexts & test_contexts
        if overlap:
            print(f"WARNING: {len(overlap)} overlapping contexts found!")
        else:
            print("✓ No train/test overlap detected")
    
    def get_train_batch(self, batch_size: int, rng: random.Random) -> List[Tuple[str, str, str]]:
        """Get a random batch from training data."""
        samples = rng.choices(self.train_data, k=batch_size)
        return [(ctx, q, a) for ctx, q, a, _ in samples]
    
    def get_test_data(self) -> List[Tuple[str, str, str, int]]:
        """Get all test data."""
        return self.test_data


# =============================================================================
# Translator Architectures
# =============================================================================

class BaseTranslator(nn.Module):
    """Base translator without normalization."""
    
    def __init__(self, input_size: int, output_size: int, target_heads: int, target_head_dim: int):
        super().__init__()
        self.target_heads = target_heads
        self.target_head_dim = target_head_dim
        
        hidden = max(input_size, output_size) * 2
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_size),
        )
    
    def forward(self, source_cache: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, head_dim = source_cache.shape
        original_dtype = source_cache.dtype
        
        x = source_cache.float().permute(0, 2, 1, 3).contiguous().view(batch * seq_len, -1)
        y = self.net(x)
        y = y.view(batch, seq_len, self.target_heads, self.target_head_dim)
        
        return y.permute(0, 2, 1, 3).contiguous().to(original_dtype)


class NormalizedTranslator(nn.Module):
    """Translator with statistical normalization."""
    
    def __init__(self, input_size: int, output_size: int, target_heads: int, target_head_dim: int):
        super().__init__()
        self.target_heads = target_heads
        self.target_head_dim = target_head_dim
        
        # Normalization parameters (set during calibration)
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
    
    def set_statistics(
        self, 
        source_mean: torch.Tensor, 
        source_std: torch.Tensor,
        target_mean: torch.Tensor, 
        target_std: torch.Tensor
    ):
        """Set normalization statistics from calibration."""
        self.source_mean = source_mean
        self.source_std = source_std.clamp(min=1e-6)
        self.target_mean = target_mean
        self.target_std = target_std.clamp(min=1e-6)
    
    def forward(self, source_cache: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, head_dim = source_cache.shape
        original_dtype = source_cache.dtype
        
        x = source_cache.float().permute(0, 2, 1, 3).contiguous().view(batch * seq_len, -1)
        
        # Normalize → Translate → Denormalize
        x_norm = (x - self.source_mean) / self.source_std
        y_norm = self.net(x_norm)
        y = y_norm * self.target_std + self.target_mean
        
        y = y.view(batch, seq_len, self.target_heads, self.target_head_dim)
        return y.permute(0, 2, 1, 3).contiguous().to(original_dtype)


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_text(s: str) -> str:
    """Normalize text for F1 calculation."""
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def calculate_f1(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth."""
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


def create_cache(keys: List[torch.Tensor], vals: List[torch.Tensor]) -> DynamicCache:
    """Create DynamicCache from key/value lists."""
    cache = DynamicCache()
    for i in range(len(keys)):
        cache.update(keys[i], vals[i], i)
    return cache


def generate_with_cache(
    model, 
    tokenizer, 
    cache: DynamicCache, 
    max_tokens: int, 
    device: str
) -> str:
    """Generate text using a pre-filled KV cache."""
    current_ids = tokenizer(" ", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if current_ids.numel() == 0:
        current_ids = torch.tensor([[220]], device=device)
    
    generated = []
    cache_len = cache.get_seq_length()
    repetition_penalty = 1.2
    
    for _ in range(max_tokens):
        attn_mask = torch.ones(1, cache_len + len(generated) + current_ids.shape[1], device=device)
        
        with torch.no_grad():
            out = model(
                input_ids=current_ids,
                attention_mask=attn_mask,
                past_key_values=cache,
                use_cache=True
            )
        
        next_logits = out.logits[:, -1, :].clone()
        
        # Apply repetition penalty
        if generated:
            for token_id in set(generated[-10:]):
                next_logits[0, token_id] /= repetition_penalty
        
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
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """Runs a single ablation experiment."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = config.device
        
        # Will be initialized in setup()
        self.tokenizer_a = None
        self.tokenizer_b = None
        self.model_a = None
        self.model_b = None
        self.translators_k = None
        self.translators_v = None
        self.dataset = None
        self.stats = None
        
        # Results
        self.eval_history: List[Tuple[int, float]] = []
        self.train_losses: List[float] = []
    
    def setup(self):
        """Initialize models, tokenizers, and data."""
        print(f"\n{'='*80}")
        print(f"Setting up experiment: {self.config.get_name()}")
        print(f"{'='*80}")
        
        # Set seed
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        
        # Load models
        print("\n1. Loading models...")
        self.tokenizer_a = AutoTokenizer.from_pretrained(self.config.model_a, trust_remote_code=True)
        self.tokenizer_b = AutoTokenizer.from_pretrained(self.config.model_b, trust_remote_code=True)
        
        if self.tokenizer_a.pad_token is None:
            self.tokenizer_a.pad_token = self.tokenizer_a.eos_token
        if self.tokenizer_b.pad_token is None:
            self.tokenizer_b.pad_token = self.tokenizer_b.eos_token
        
        self.model_a = AutoModelForCausalLM.from_pretrained(
            self.config.model_a, torch_dtype=torch.float16, trust_remote_code=True
        ).to(self.device)
        self.model_b = AutoModelForCausalLM.from_pretrained(
            self.config.model_b, torch_dtype=torch.float16, trust_remote_code=True
        ).to(self.device)
        
        self.model_a.eval()
        self.model_b.eval()
        for p in self.model_a.parameters():
            p.requires_grad = False
        for p in self.model_b.parameters():
            p.requires_grad = False
        
        # Detect actual layer counts
        sample_a = self.tokenizer_a(["test"], return_tensors="pt", max_length=32,
                                     padding="max_length", truncation=True).to(self.device)
        sample_b = self.tokenizer_b(["test"], return_tensors="pt", max_length=32,
                                     padding="max_length", truncation=True).to(self.device)
        with torch.no_grad():
            out_a = self.model_a(**sample_a, use_cache=True)
            out_b = self.model_b(**sample_b, use_cache=True)
        
        self.num_layers_a = len(out_a.past_key_values)
        self.num_layers_b = len(out_b.past_key_values)
        
        print(f"   Model A ({self.config.model_a}): {self.num_layers_a} layers")
        print(f"   Model B ({self.config.model_b}): {self.num_layers_b} layers")
        
        # Create layer mapping based on alignment strategy
        if self.config.layer_alignment == "interpolate":
            # Map ALL source layers to ALL target layers via interpolation
            # Each target layer j gets input from source layer round(j * (num_a-1) / (num_b-1))
            self.layer_mapping = []  # layer_mapping[target_idx] = source_idx
            for j in range(self.num_layers_b):
                if self.num_layers_b == 1:
                    src_idx = 0
                else:
                    src_idx = round(j * (self.num_layers_a - 1) / (self.num_layers_b - 1))
                self.layer_mapping.append(src_idx)
            
            self.num_translate_layers = self.num_layers_b  # We translate ALL target layers
            self.layer_offset_b = 0  # No offset, we cover everything
            self.use_interpolation = True
            
            print(f"   Using INTERPOLATE alignment: mapping {self.num_layers_a} → {self.num_layers_b} layers")
            print(f"   Layer mapping (target → source): {list(enumerate(self.layer_mapping))[:5]}...")
        else:
            # Terminal alignment (default): align from the end
            self.num_translate_layers = min(self.num_layers_a, self.num_layers_b)
            self.use_interpolation = False
            self.layer_mapping = None
            
            if self.num_layers_a != self.num_layers_b:
                print(f"   Using TERMINAL alignment with {self.num_translate_layers} layers")
                self.layer_offset_b = self.num_layers_b - self.num_translate_layers
                print(f"   Mapping: A[0:{self.num_translate_layers}] → B[{self.layer_offset_b}:{self.num_layers_b}]")
            else:
                self.layer_offset_b = 0
                print(f"   Perfect layer match: 1:1 mapping")
        
        del out_a, out_b
        
        # Generate data
        print("\n2. Generating datasets...")
        self.dataset = SyntheticQADataset(seed=self.config.seed)
        self.dataset.generate_datasets(
            self.config.num_facts,
            self.config.train_samples,
            self.config.test_samples
        )
        
        # Get dimensions for translator creation
        print("\n3. Creating translators...")
        sample_a = self.tokenizer_a(["test"], return_tensors="pt", max_length=self.config.max_ctx_tokens,
                                     padding="max_length", truncation=True).to(self.device)
        sample_b = self.tokenizer_b(["test"], return_tensors="pt", max_length=self.config.max_ctx_tokens,
                                     padding="max_length", truncation=True).to(self.device)
        with torch.no_grad():
            out_a = self.model_a(**sample_a, use_cache=True)
            out_b = self.model_b(**sample_b, use_cache=True)
        
        # Create translators
        TranslatorClass = NormalizedTranslator if self.config.use_normalization else BaseTranslator
        
        self.translators_k = nn.ModuleList()
        self.translators_v = nn.ModuleList()
        
        if self.use_interpolation:
            # Create one translator per TARGET layer
            # Each uses the source layer from layer_mapping
            for target_idx in range(self.num_layers_b):
                source_idx = self.layer_mapping[target_idx]
                sk, sv = out_a.past_key_values[source_idx]
                tk, tv = out_b.past_key_values[target_idx]
                
                k_in = sk.shape[1] * sk.shape[3]
                k_out = tk.shape[1] * tk.shape[3]
                v_in = sv.shape[1] * sv.shape[3]
                v_out = tv.shape[1] * tv.shape[3]
                
                self.translators_k.append(
                    TranslatorClass(k_in, k_out, tk.shape[1], tk.shape[3]).to(self.device)
                )
                self.translators_v.append(
                    TranslatorClass(v_in, v_out, tv.shape[1], tv.shape[3]).to(self.device)
                )
        else:
            # Terminal alignment: one translator per matched layer pair
            for i in range(self.num_translate_layers):
                # Source layer i maps to target layer (layer_offset_b + i)
                sk, sv = out_a.past_key_values[i]
                tk, tv = out_b.past_key_values[self.layer_offset_b + i]
                
                k_in = sk.shape[1] * sk.shape[3]
                k_out = tk.shape[1] * tk.shape[3]
                v_in = sv.shape[1] * sv.shape[3]
                v_out = tv.shape[1] * tv.shape[3]
                
                self.translators_k.append(
                    TranslatorClass(k_in, k_out, tk.shape[1], tk.shape[3]).to(self.device)
                )
                self.translators_v.append(
                    TranslatorClass(v_in, v_out, tv.shape[1], tv.shape[3]).to(self.device)
                )
        
        del out_a, out_b
        
        # Calibrate normalization if enabled
        if self.config.use_normalization:
            print("\n4. Calibrating normalization statistics...")
            self._calibrate_statistics()
        
        total_params = sum(p.numel() for p in self.translators_k.parameters()) + \
                       sum(p.numel() for p in self.translators_v.parameters())
        print(f"\n   Total translator parameters: {total_params:,}")
    
    def _calibrate_statistics(self):
        """Collect statistics for normalization."""
        all_source_k = [[] for _ in range(self.num_translate_layers)]
        all_source_v = [[] for _ in range(self.num_translate_layers)]
        all_target_k = [[] for _ in range(self.num_translate_layers)]
        all_target_v = [[] for _ in range(self.num_translate_layers)]
        
        # Use training data for calibration
        calib_samples = self.dataset.train_data[:self.config.calibration_samples]
        
        for ctx, q, _, _ in tqdm(calib_samples, desc="Calibrating"):
            prompt = f"Context:\n{ctx}\n\nQuestion: {q} Answer:"
            
            inputs_a = self.tokenizer_a(prompt, return_tensors="pt", padding="max_length",
                                        truncation=True, max_length=self.config.max_ctx_tokens).to(self.device)
            inputs_b = self.tokenizer_b(prompt, return_tensors="pt", padding="max_length",
                                        truncation=True, max_length=self.config.max_ctx_tokens).to(self.device)
            
            with torch.no_grad():
                out_a = self.model_a(**inputs_a, use_cache=True)
                out_b = self.model_b(**inputs_b, use_cache=True)
            
            for i in range(self.num_translate_layers):
                # Get source and target layer indices based on alignment strategy
                if self.use_interpolation:
                    source_idx = self.layer_mapping[i]
                    target_idx = i
                else:
                    source_idx = i
                    target_idx = self.layer_offset_b + i
                
                sk = out_a.past_key_values[source_idx][0]
                sv = out_a.past_key_values[source_idx][1]
                tk = out_b.past_key_values[target_idx][0]
                tv = out_b.past_key_values[target_idx][1]
                
                sk_flat = sk.permute(0, 2, 1, 3).reshape(-1, sk.shape[1] * sk.shape[3])
                sv_flat = sv.permute(0, 2, 1, 3).reshape(-1, sv.shape[1] * sv.shape[3])
                tk_flat = tk.permute(0, 2, 1, 3).reshape(-1, tk.shape[1] * tk.shape[3])
                tv_flat = tv.permute(0, 2, 1, 3).reshape(-1, tv.shape[1] * tv.shape[3])
                
                all_source_k[i].append(sk_flat.float().cpu())
                all_source_v[i].append(sv_flat.float().cpu())
                all_target_k[i].append(tk_flat.float().cpu())
                all_target_v[i].append(tv_flat.float().cpu())
        
        # Compute and set statistics
        for i in range(self.num_translate_layers):
            sk_all = torch.cat(all_source_k[i], dim=0)
            sv_all = torch.cat(all_source_v[i], dim=0)
            tk_all = torch.cat(all_target_k[i], dim=0)
            tv_all = torch.cat(all_target_v[i], dim=0)
            
            self.translators_k[i].set_statistics(
                sk_all.mean(dim=0).to(self.device),
                sk_all.std(dim=0).to(self.device),
                tk_all.mean(dim=0).to(self.device),
                tk_all.std(dim=0).to(self.device)
            )
            self.translators_v[i].set_statistics(
                sv_all.mean(dim=0).to(self.device),
                sv_all.std(dim=0).to(self.device),
                tv_all.mean(dim=0).to(self.device),
                tv_all.std(dim=0).to(self.device)
            )
    
    def _get_loss_weights(self, step: int) -> Tuple[float, float]:
        """Get MSE and NTP weights based on loss type and training progress."""
        if self.config.loss_type == "mse":
            return 1.0, 0.0
        elif self.config.loss_type == "ntp":
            return 0.0, 1.0
        else:  # curriculum
            progress = step / self.config.train_steps
            mse_w = self.config.mse_weight_start + \
                    (self.config.mse_weight_end - self.config.mse_weight_start) * progress
            ntp_w = self.config.ntp_weight_start + \
                    (self.config.ntp_weight_end - self.config.ntp_weight_start) * progress
            return mse_w, ntp_w
    
    def _compute_ntp_loss(
        self, 
        prompts: List[str], 
        answers: List[str]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Compute next-token prediction loss."""
        # Get source cache
        inputs_a = self.tokenizer_a(
            prompts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.config.max_ctx_tokens
        ).to(self.device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16, 
                                    enabled=self.config.mixed_precision):
                out_a = self.model_a(**inputs_a, use_cache=True)
        
        # Get source caches based on alignment strategy
        if self.use_interpolation:
            # For interpolation, each translator i uses source layer layer_mapping[i]
            src_k = [out_a.past_key_values[self.layer_mapping[i]][0] for i in range(self.num_translate_layers)]
            src_v = [out_a.past_key_values[self.layer_mapping[i]][1] for i in range(self.num_translate_layers)]
        else:
            src_k = [out_a.past_key_values[i][0] for i in range(self.num_translate_layers)]
            src_v = [out_a.past_key_values[i][1] for i in range(self.num_translate_layers)]
        
        # Translate
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16,
                                enabled=self.config.mixed_precision):
            trans_k = [self.translators_k[i](src_k[i]) for i in range(self.num_translate_layers)]
            trans_v = [self.translators_v[i](src_v[i]) for i in range(self.num_translate_layers)]
        
        # Build full cache for Model B
        if self.use_interpolation:
            # Interpolation covers all layers, trans_k/v already has num_layers_b elements
            full_trans_k = trans_k
            full_trans_v = trans_v
        else:
            full_trans_k = []
            full_trans_v = []
            
            # If there's a layer offset, get Model B's native cache for early layers
            if self.layer_offset_b > 0:
                inputs_b = self.tokenizer_b(
                    prompts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=self.config.max_ctx_tokens
                ).to(self.device)
                with torch.no_grad():
                    out_b_prefix = self.model_b(**inputs_b, use_cache=True)
                
                for i in range(self.layer_offset_b):
                    full_trans_k.append(out_b_prefix.past_key_values[i][0])
                    full_trans_v.append(out_b_prefix.past_key_values[i][1])
                del out_b_prefix
            
            full_trans_k.extend(trans_k)
            full_trans_v.extend(trans_v)
        
        cache = create_cache(full_trans_k, full_trans_v)
        
        # Get answer tokens
        ans_inputs = self.tokenizer_b(
            answers, return_tensors="pt", padding=True,
            truncation=True, max_length=32
        ).to(self.device)
        
        ans_ids = ans_inputs.input_ids
        ans_len = ans_ids.shape[1]
        ctx_len = cache.get_seq_length()
        
        attn_mask = torch.ones(len(prompts), ctx_len + ans_len, device=self.device)
        pos_ids = torch.arange(ctx_len, ctx_len + ans_len, device=self.device).unsqueeze(0).expand(len(prompts), -1)
        
        # Forward through Model B
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16,
                                enabled=self.config.mixed_precision):
            out_b = self.model_b(
                input_ids=ans_ids,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_values=cache,
                use_cache=False
            )
        
        # NTP loss
        logits = out_b.logits[:, :-1, :].contiguous()
        labels = ans_ids[:, 1:].contiguous()
        ntp_loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        del out_a, out_b
        return ntp_loss, trans_k, trans_v
    
    def _compute_mse_loss(
        self, 
        prompts: List[str],
        trans_k: List[torch.Tensor], 
        trans_v: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute MSE loss against Model B's native cache."""
        # Get target cache
        inputs_b = self.tokenizer_b(
            prompts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.config.max_ctx_tokens
        ).to(self.device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16,
                                    enabled=self.config.mixed_precision):
                out_b = self.model_b(**inputs_b, use_cache=True)
        
        mse_loss = 0
        for i in range(self.num_translate_layers):
            # Target is layer (layer_offset_b + i) from Model B
            tk = out_b.past_key_values[self.layer_offset_b + i][0].float()
            tv = out_b.past_key_values[self.layer_offset_b + i][1].float()
            pk = trans_k[i].float()
            pv = trans_v[i].float()
            
            mse_loss += nn.MSELoss()(pk, tk) + nn.MSELoss()(pv, tv)
        
        del out_b
        return mse_loss / (2 * self.num_translate_layers)
    
    def train(self):
        """Run training loop."""
        print(f"\n{'='*80}")
        print(f"Training: {self.config.get_name()}")
        print(f"{'='*80}")
        
        params = list(self.translators_k.parameters()) + list(self.translators_v.parameters())
        optimizer = optim.AdamW(params, lr=self.config.learning_rate, weight_decay=0.01)
        
        # LR schedule with warmup and cosine decay
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            progress = (step - self.config.warmup_steps) / (self.config.train_steps - self.config.warmup_steps)
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = torch.amp.GradScaler(enabled=self.config.mixed_precision and self.device == "cuda")
        
        train_rng = random.Random(self.config.seed + 1000)
        running_loss = 0
        
        for step in trange(self.config.train_steps, desc="Training"):
            self.translators_k.train()
            self.translators_v.train()
            
            # Get batch
            batch = self.dataset.get_train_batch(self.config.batch_size, train_rng)
            prompts = [f"Context:\n{ctx}\n\nQuestion: {q} Answer:" for ctx, q, _ in batch]
            answers = [" " + a for _, _, a in batch]
            
            # Get loss weights
            mse_weight, ntp_weight = self._get_loss_weights(step)
            
            optimizer.zero_grad()
            
            # Compute losses
            total_loss = torch.tensor(0.0, device=self.device)
            
            if ntp_weight > 0:
                ntp_loss, trans_k, trans_v = self._compute_ntp_loss(prompts, answers)
                total_loss = total_loss + ntp_weight * ntp_loss
            else:
                # Still need to translate for MSE
                inputs_a = self.tokenizer_a(
                    prompts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=self.config.max_ctx_tokens
                ).to(self.device)
                
                with torch.no_grad():
                    out_a = self.model_a(**inputs_a, use_cache=True)
                
                src_k = [out_a.past_key_values[i][0] for i in range(self.num_translate_layers)]
                src_v = [out_a.past_key_values[i][1] for i in range(self.num_translate_layers)]
                
                trans_k = [self.translators_k[i](src_k[i]) for i in range(self.num_translate_layers)]
                trans_v = [self.translators_v[i](src_v[i]) for i in range(self.num_translate_layers)]
                
                del out_a
            
            if mse_weight > 0:
                mse_loss = self._compute_mse_loss(prompts, trans_k, trans_v)
                total_loss = total_loss + mse_weight * mse_loss
            
            # Backward
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += total_loss.item()
            self.train_losses.append(total_loss.item())
            
            del trans_k, trans_v
            
            # Log
            if (step + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(f"\n[{step+1}] Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.5f}, "
                      f"Weights: MSE={mse_weight:.2f}, NTP={ntp_weight:.2f}")
                running_loss = 0
            
            # Evaluate
            if (step + 1) % self.config.eval_every == 0:
                torch.cuda.empty_cache()
                f1 = self.evaluate()
                self.eval_history.append((step + 1, f1))
                
                # Early stopping check
                if self.config.early_stopping and len(self.eval_history) >= 2:
                    best_f1 = max(f1 for _, f1 in self.eval_history)
                    current_f1 = f1
                    
                    # Count evals since best
                    evals_since_best = 0
                    for s, f in reversed(self.eval_history):
                        if f == best_f1:
                            break
                        evals_since_best += 1
                    
                    if evals_since_best >= self.config.patience:
                        print(f"\n   Early stopping! Best F1: {best_f1:.4f} at step {self.eval_history[len(self.eval_history) - evals_since_best - 1][0]}")
                        print(f"   No improvement for {evals_since_best} evaluations.")
                        break
                
                torch.cuda.empty_cache()
    
    def evaluate(self) -> float:
        """Evaluate on test set."""
        self.translators_k.eval()
        self.translators_v.eval()
        
        total_f1 = 0
        test_data = self.dataset.get_test_data()
        
        for i, (ctx, q, gt, nf) in enumerate(test_data):
            prompt = f"Context:\n{ctx}\n\nQuestion: {q} Answer:"
            
            try:
                inputs_a = self.tokenizer_a(
                    [prompt], return_tensors="pt", padding="max_length",
                    truncation=True, max_length=self.config.max_ctx_tokens
                ).to(self.device)
                
                with torch.no_grad():
                    out_a = self.model_a(**inputs_a, use_cache=True)
                
                # Get source caches based on alignment strategy
                if self.use_interpolation:
                    src_k = [out_a.past_key_values[self.layer_mapping[j]][0] for j in range(self.num_translate_layers)]
                    src_v = [out_a.past_key_values[self.layer_mapping[j]][1] for j in range(self.num_translate_layers)]
                else:
                    src_k = [out_a.past_key_values[j][0] for j in range(self.num_translate_layers)]
                    src_v = [out_a.past_key_values[j][1] for j in range(self.num_translate_layers)]
                
                with torch.no_grad():
                    trans_k = [self.translators_k[j](src_k[j]) for j in range(self.num_translate_layers)]
                    trans_v = [self.translators_v[j](src_v[j]) for j in range(self.num_translate_layers)]
                
                # Build full cache
                if self.use_interpolation:
                    full_trans_k = trans_k
                    full_trans_v = trans_v
                else:
                    full_trans_k = []
                    full_trans_v = []
                    
                    if self.layer_offset_b > 0:
                        inputs_b = self.tokenizer_b(
                            [prompt], return_tensors="pt", padding="max_length",
                            truncation=True, max_length=self.config.max_ctx_tokens
                        ).to(self.device)
                        with torch.no_grad():
                            out_b_prefix = self.model_b(**inputs_b, use_cache=True)
                        
                        for j in range(self.layer_offset_b):
                            full_trans_k.append(out_b_prefix.past_key_values[j][0])
                            full_trans_v.append(out_b_prefix.past_key_values[j][1])
                        del out_b_prefix
                    
                    full_trans_k.extend(trans_k)
                    full_trans_v.extend(trans_v)
                
                cache = create_cache(full_trans_k, full_trans_v)
                response = generate_with_cache(
                    self.model_b, self.tokenizer_b, cache,
                    self.config.max_new_tokens, self.device
                )
                
                # Clean response
                cleaned = response.split('\n')[0].strip()
                for end in ['.', '?', '!']:
                    if end in cleaned:
                        cleaned = cleaned.split(end)[0].strip()
                        break
                
                f1 = calculate_f1(cleaned, gt)
                total_f1 += f1
                
                if i < 3:
                    print(f"   {i+1}. [{nf}F] Q: {q[:35]}... GT: {gt} | Pred: '{cleaned[:40]}' | F1: {f1:.2f}")
                
                del out_a, trans_k, trans_v, cache
                
            except Exception as e:
                print(f"   Eval {i+1} failed: {e}")
        
        avg_f1 = total_f1 / len(test_data)
        print(f"\n   Test F1: {avg_f1:.4f}")
        
        return avg_f1
    
    def save_results(self, output_dir: str):
        """Save experiment results."""
        os.makedirs(output_dir, exist_ok=True)
        
        name = self.config.get_name()
        
        # Save config
        with open(os.path.join(output_dir, f"{name}_config.json"), 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        # Save eval history
        results = {
            'config': asdict(self.config),
            'eval_history': self.eval_history,
            'best_f1': max([f1 for _, f1 in self.eval_history]) if self.eval_history else 0,
            'final_f1': self.eval_history[-1][1] if self.eval_history else 0,
        }
        
        with open(os.path.join(output_dir, f"{name}_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save model
        torch.save({
            'translators_k': self.translators_k.state_dict(),
            'translators_v': self.translators_v.state_dict(),
            'config': asdict(self.config),
            'eval_history': self.eval_history,
        }, os.path.join(output_dir, f"{name}_model.pth"))
        
        print(f"\nResults saved to {output_dir}/{name}_*")
        
        return results


# =============================================================================
# Main
# =============================================================================

def run_single_experiment(args) -> Dict[str, Any]:
    """Run a single experiment with given arguments."""
    config = ExperimentConfig(
        model_family=args.model,
        num_facts=args.facts,
        loss_type=args.loss,
        use_normalization=args.normalize,
        layer_alignment=args.layer_align,
        train_steps=args.steps,
        learning_rate=args.lr,
        eval_every=args.eval_every,
        early_stopping=not args.no_early_stopping,
        patience=args.patience,
        seed=args.seed,
    )
    
    runner = ExperimentRunner(config)
    runner.setup()
    runner.train()
    results = runner.save_results(args.output_dir)
    
    return results


def run_all_experiments(args) -> List[Dict[str, Any]]:
    """Run all ablation combinations for the specified model family."""
    fact_options = ["1", "2", "3", "4", "mixed"]
    loss_options = ["mse", "ntp", "curriculum"]
    norm_options = [True, False]
    
    all_results = []
    
    print(f"\n{'#'*80}")
    print(f"# Running all ablations for model family: {args.model}")
    print(f"# {MODEL_CONFIGS[args.model]['description']}")
    print(f"{'#'*80}")
    
    for facts in fact_options:
        for loss in loss_options:
            for normalize in norm_options:
                print(f"\n{'#'*80}")
                print(f"# Running: {args.model}, facts={facts}, loss={loss}, normalize={normalize}")
                print(f"{'#'*80}")
                
                config = ExperimentConfig(
                    model_family=args.model,
                    num_facts=facts,
                    loss_type=loss,
                    use_normalization=normalize,
                    train_steps=args.steps,
                    seed=args.seed,
                )
                
                try:
                    runner = ExperimentRunner(config)
                    runner.setup()
                    runner.train()
                    results = runner.save_results(args.output_dir)
                    all_results.append(results)
                except Exception as e:
                    print(f"Experiment failed: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        'config': asdict(config),
                        'error': str(e),
                    })
                
                # Clear memory
                torch.cuda.empty_cache()
    
    # Save summary
    summary_path = os.path.join(args.output_dir, f"{args.model}_ablation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"ABLATION SUMMARY: {args.model}")
    print(f"{'='*80}")
    print(f"{'Facts':<8} {'Loss':<12} {'Norm':<8} {'Best F1':<10} {'Final F1':<10}")
    print("-" * 50)
    
    for r in all_results:
        if 'error' in r:
            print(f"{r['config']['num_facts']:<8} {r['config']['loss_type']:<12} "
                  f"{str(r['config']['use_normalization']):<8} ERROR")
        else:
            print(f"{r['config']['num_facts']:<8} {r['config']['loss_type']:<12} "
                  f"{str(r['config']['use_normalization']):<8} {r['best_f1']:<10.4f} {r['final_f1']:<10.4f}")
    
    return all_results


def run_baseline(args) -> Dict[str, Any]:
    """Run baseline evaluation (Model B with direct context)."""
    print(f"\n{'='*80}")
    print(f"BASELINE EVALUATION: {args.model}")
    print(f"{'='*80}")
    
    config = MODEL_CONFIGS[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nLoading {config['model_b']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_b'], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config['model_b'], torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Generate test data
    num_facts = int(args.facts) if args.facts != "mixed" else 4
    dataset = SyntheticQADataset(seed=args.seed)
    dataset.generate_datasets(args.facts, 100, 100)
    test_data = dataset.get_test_data()
    
    print(f"\nEvaluating on {len(test_data)} examples...")
    
    f1_scores = []
    for i, (ctx, q, gt, nf) in enumerate(tqdm(test_data, desc="Baseline eval")):
        prompt = f"Context:\n{ctx}\n\nQuestion: {q} Answer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        cleaned = response.split('\n')[0].strip()
        for end in ['.', '?', '!']:
            if end in cleaned:
                cleaned = cleaned.split(end)[0].strip()
                break
        
        f1 = calculate_f1(cleaned, gt)
        f1_scores.append(f1)
        
        if i < 5:
            print(f"  {i+1}. [{nf}F] GT: {gt} | Pred: '{cleaned[:40]}' | F1: {f1:.2f}")
    
    avg_f1 = sum(f1_scores) / len(f1_scores)
    std_f1 = (sum((x - avg_f1)**2 for x in f1_scores) / len(f1_scores)) ** 0.5
    
    print(f"\n{'='*80}")
    print(f"BASELINE RESULTS ({args.facts} FACTS)")
    print(f"{'='*80}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Std Dev:          {std_f1:.4f}")
    print(f"Min F1:           {min(f1_scores):.4f}")
    print(f"Max F1:           {max(f1_scores):.4f}")
    print(f"Perfect (F1=1.0): {sum(1 for x in f1_scores if x == 1.0)}/{len(f1_scores)}")
    print(f"{'='*80}")
    
    results = {
        'model_family': args.model,
        'model': config['model_b'],
        'num_facts': args.facts,
        'avg_f1': avg_f1,
        'std_f1': std_f1,
        'min_f1': min(f1_scores),
        'max_f1': max(f1_scores),
        'num_samples': len(f1_scores),
    }
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.model}_baseline_{args.facts}facts.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="KV Cache Translation Ablation Study")
    
    # Model selection
    parser.add_argument("--model", type=str, default="qwen",
                        choices=list(MODEL_CONFIGS.keys()),
                        help=f"Model family: {list(MODEL_CONFIGS.keys())}")
    
    # Experiment selection
    parser.add_argument("--run-all", action="store_true", help="Run all ablation combinations")
    parser.add_argument("--baseline", action="store_true", help="Run baseline evaluation only")
    
    # Single experiment settings
    parser.add_argument("--facts", type=str, default="2", 
                        choices=["1", "2", "3", "4", "mixed"],
                        help="Number of facts (1-4 or mixed)")
    parser.add_argument("--loss", type=str, default="ntp",
                        choices=["mse", "ntp", "curriculum"],
                        help="Loss function type")
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="Use statistical normalization")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize",
                        help="Disable statistical normalization")
    parser.add_argument("--layer-align", type=str, default="terminal",
                        choices=["terminal", "interpolate"],
                        help="Layer alignment strategy: terminal (align from end) or interpolate (map all to all)")
    
    # Training settings
    parser.add_argument("--steps", type=int, default=4000, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluation frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="ablation_results",
                        help="Output directory for results")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Disable early stopping")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (number of evals)")
    
    args = parser.parse_args()
    
    # Print model info
    if args.model in MODEL_CONFIGS:
        config = MODEL_CONFIGS[args.model]
        print(f"\n{'='*80}")
        print(f"Model Family: {args.model}")
        print(f"Description: {config['description']}")
        print(f"Model A: {config['model_a']}")
        print(f"Model B: {config['model_b']}")
        print(f"Num Layers: {config['num_layers']}")
        print(f"{'='*80}")
    
    if args.baseline:
        run_baseline(args)
    elif args.run_all:
        run_all_experiments(args)
    else:
        run_single_experiment(args)


if __name__ == "__main__":
    main()
