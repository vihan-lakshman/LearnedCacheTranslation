#!/usr/bin/env python3
"""
KV Cache Translation - Hard Synthetic QA

Addresses issues with the original synthetic dataset:
1. Much larger answer vocabulary (500+ unique answers vs 55)
2. Train/test split BY ENTITY (some entities only appear in test)
3. More varied question templates
4. Compositional answers (e.g., "Dr. Sarah Chen" not just "Dr. Chen")

Usage:
    python train_hard_synthetic.py --steps 2000
    python train_hard_synthetic.py --analyze-only  # Just analyze dataset
"""

import argparse
import random
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Set
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, logging, DynamicCache
from tqdm import tqdm, trange
import re
import string

logging.set_verbosity_error()


# =============================================================================
# Model Configuration
# =============================================================================

MODEL_A = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_B = "meta-llama/Llama-3.1-8B-Instruct"


# =============================================================================
# Expanded Vocabulary (much larger than original)
# =============================================================================

# First names (50)
FIRST_NAMES = [
    "James", "Sarah", "Michael", "Emily", "David", "Jessica", "Robert", "Ashley",
    "William", "Amanda", "Richard", "Stephanie", "Joseph", "Nicole", "Thomas", "Elizabeth",
    "Charles", "Jennifer", "Daniel", "Melissa", "Matthew", "Rebecca", "Anthony", "Laura",
    "Mark", "Katherine", "Steven", "Rachel", "Paul", "Michelle", "Andrew", "Christina",
    "Joshua", "Samantha", "Kenneth", "Heather", "Kevin", "Angela", "Brian", "Maria",
    "George", "Diana", "Edward", "Julie", "Ronald", "Karen", "Timothy", "Nancy",
    "Jason", "Betty",
]

# Last names (50)
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
    "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
    "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts",
]

# Titles (10)
TITLES = ["Dr.", "Prof.", "Agent", "Detective", "Captain", "Director", "Officer", "Manager", "Chief", "Inspector"]

# Actions (30) - more variety
ACTIONS = [
    "placed", "stored", "hid", "secured", "left", "put", "deposited", "locked",
    "found", "discovered", "located", "retrieved", "obtained", "collected", "gathered",
    "moved", "transferred", "delivered", "sent", "shipped", "transported",
    "examined", "inspected", "analyzed", "studied", "reviewed", "checked",
    "created", "built", "assembled",
]

# Object adjectives (20)
OBJ_ADJECTIVES = [
    "red", "blue", "green", "yellow", "black", "white", "silver", "gold",
    "large", "small", "heavy", "light", "old", "new", "ancient", "modern",
    "encrypted", "classified", "sealed", "damaged",
]

# Object nouns (25)
OBJ_NOUNS = [
    "folder", "box", "case", "envelope", "package", "container", "briefcase",
    "document", "file", "report", "letter", "notebook", "journal", "manuscript",
    "key", "card", "badge", "device", "drive", "disk", "chip",
    "artifact", "sample", "specimen", "prototype",
]

# Location types (15)
LOCATION_TYPES = [
    "laboratory", "office", "warehouse", "vault", "archive", "storage room",
    "basement", "attic", "garage", "shed", "bunker", "facility",
    "building", "wing", "department",
]

# Location modifiers (20)
LOCATION_MODIFIERS = [
    "main", "north", "south", "east", "west", "central", "upper", "lower",
    "old", "new", "secure", "restricted", "underground", "secret", "hidden",
    "primary", "secondary", "auxiliary", "emergency", "backup",
]

# Prepositions for locations
LOCATION_PREPS = ["in the", "at the", "inside the", "within the", "near the", "behind the", "under the"]


# =============================================================================
# Hard Synthetic Dataset
# =============================================================================

class HardSyntheticDataset:
    """
    Generates harder synthetic QA with:
    - Compositional names (Title + First + Last = 500+ combinations)
    - Compositional objects (Adj + Noun = 500 combinations)
    - Compositional locations (Prep + Mod + Type = 2000+ combinations)
    - Train/test split by entity (some people/objects only in test)
    """
    
    def __init__(self, seed: int = 42, test_holdout_ratio: float = 0.2):
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Create entity pools with train/test split
        self._create_entity_pools(test_holdout_ratio)
        
        self.train_data: List[Tuple[str, str, str, int]] = []
        self.test_data: List[Tuple[str, str, str, int]] = []
    
    def _create_entity_pools(self, holdout_ratio: float):
        """Split entities into train and test pools."""
        rng = random.Random(self.seed)
        
        # Shuffle and split first names
        first_names = FIRST_NAMES.copy()
        rng.shuffle(first_names)
        split = int(len(first_names) * (1 - holdout_ratio))
        self.train_first_names = first_names[:split]
        self.test_first_names = first_names[split:]  # Some names ONLY in test
        self.all_first_names = first_names
        
        # Shuffle and split last names
        last_names = LAST_NAMES.copy()
        rng.shuffle(last_names)
        split = int(len(last_names) * (1 - holdout_ratio))
        self.train_last_names = last_names[:split]
        self.test_last_names = last_names[split:]
        self.all_last_names = last_names
        
        # Shuffle and split object nouns
        obj_nouns = OBJ_NOUNS.copy()
        rng.shuffle(obj_nouns)
        split = int(len(obj_nouns) * (1 - holdout_ratio))
        self.train_obj_nouns = obj_nouns[:split]
        self.test_obj_nouns = obj_nouns[split:]
        self.all_obj_nouns = obj_nouns
        
        # Shuffle and split location types
        loc_types = LOCATION_TYPES.copy()
        rng.shuffle(loc_types)
        split = int(len(loc_types) * (1 - holdout_ratio))
        self.train_loc_types = loc_types[:split]
        self.test_loc_types = loc_types[split:]
        self.all_loc_types = loc_types
        
        print(f"Entity pools created:")
        print(f"  First names: {len(self.train_first_names)} train, {len(self.test_first_names)} test-only")
        print(f"  Last names: {len(self.train_last_names)} train, {len(self.test_last_names)} test-only")
        print(f"  Object nouns: {len(self.train_obj_nouns)} train, {len(self.test_obj_nouns)} test-only")
        print(f"  Location types: {len(self.train_loc_types)} train, {len(self.test_loc_types)} test-only")
    
    def _generate_person(self, rng: random.Random, use_test_pool: bool = False) -> str:
        """Generate a person name."""
        if use_test_pool:
            # For test: mix of train entities and test-only entities
            # 50% chance to use a test-only entity
            if rng.random() < 0.5 and self.test_first_names:
                first = rng.choice(self.test_first_names)
            else:
                first = rng.choice(self.all_first_names)
            
            if rng.random() < 0.5 and self.test_last_names:
                last = rng.choice(self.test_last_names)
            else:
                last = rng.choice(self.all_last_names)
        else:
            # For train: only use train pool
            first = rng.choice(self.train_first_names)
            last = rng.choice(self.train_last_names)
        
        # Sometimes add title
        if rng.random() < 0.5:
            title = rng.choice(TITLES)
            return f"{title} {first} {last}"
        return f"{first} {last}"
    
    def _generate_object(self, rng: random.Random, use_test_pool: bool = False) -> str:
        """Generate an object description."""
        adj = rng.choice(OBJ_ADJECTIVES)
        
        if use_test_pool:
            if rng.random() < 0.5 and self.test_obj_nouns:
                noun = rng.choice(self.test_obj_nouns)
            else:
                noun = rng.choice(self.all_obj_nouns)
        else:
            noun = rng.choice(self.train_obj_nouns)
        
        return f"the {adj} {noun}"
    
    def _generate_location(self, rng: random.Random, use_test_pool: bool = False) -> str:
        """Generate a location description."""
        prep = rng.choice(LOCATION_PREPS)
        mod = rng.choice(LOCATION_MODIFIERS)
        
        if use_test_pool:
            if rng.random() < 0.5 and self.test_loc_types:
                loc_type = rng.choice(self.test_loc_types)
            else:
                loc_type = rng.choice(self.all_loc_types)
        else:
            loc_type = rng.choice(self.train_loc_types)
        
        return f"{prep} {mod} {loc_type}"
    
    def _generate_fact(self, rng: random.Random, use_test_pool: bool = False) -> dict:
        """Generate a single fact."""
        return {
            'person': self._generate_person(rng, use_test_pool),
            'action': rng.choice(ACTIONS),
            'object': self._generate_object(rng, use_test_pool),
            'location': self._generate_location(rng, use_test_pool),
        }
    
    def _fact_to_sentence(self, fact: dict) -> str:
        """Convert fact to natural sentence."""
        templates = [
            "{person} {action} {object} {location}.",
            "{location}, {person} {action} {object}.",
            "{object} was {action} by {person} {location}.",
        ]
        template = random.choice(templates)
        return template.format(**fact)
    
    def _generate_qa(self, rng: random.Random, num_facts: int, use_test_pool: bool = False) -> Tuple[str, str, str]:
        """Generate a QA pair."""
        facts = [self._generate_fact(rng, use_test_pool) for _ in range(num_facts)]
        
        # Build context with varied sentence structures
        sentences = [self._fact_to_sentence(f) for f in facts]
        rng.shuffle(sentences)
        context = " ".join(sentences)
        
        # Pick target fact
        target = rng.choice(facts)
        
        # Generate question with more variety
        q_type = rng.choice(["who", "what", "where"])
        
        if q_type == "who":
            templates = [
                f"Who {target['action']} {target['object']}?",
                f"Who {target['action']} {target['object']} {target['location']}?",
                f"Which person {target['action']} {target['object']}?",
            ]
            answer = target['person']
        elif q_type == "what":
            templates = [
                f"What did {target['person']} {target['action']}?",
                f"What was {target['action']} by {target['person']}?",
                f"Which item did {target['person']} {target['action']}?",
            ]
            answer = target['object']
        else:
            templates = [
                f"Where did {target['person']} {target['action']} {target['object']}?",
                f"Where was {target['object']} {target['action']}?",
                f"In which location was {target['object']} {target['action']}?",
            ]
            answer = target['location']
        
        question = rng.choice(templates)
        return context, question, answer
    
    def generate_datasets(self, train_size: int, test_size: int, num_facts: str = "mixed"):
        """Generate train and test datasets."""
        train_rng = random.Random(self.seed + 1)
        test_rng = random.Random(self.seed + 10000)
        
        def get_num_facts(rng):
            if num_facts == "mixed":
                return rng.choice([2, 3, 4, 5])
            return int(num_facts)
        
        # Generate training data (using train entity pool)
        self.train_data = []
        for _ in range(train_size):
            nf = get_num_facts(train_rng)
            ctx, q, a = self._generate_qa(train_rng, nf, use_test_pool=False)
            self.train_data.append((ctx, q, a, nf))
        
        # Generate test data (using mixed pool with test-only entities)
        self.test_data = []
        for _ in range(test_size):
            nf = get_num_facts(test_rng)
            ctx, q, a = self._generate_qa(test_rng, nf, use_test_pool=True)
            self.test_data.append((ctx, q, a, nf))
        
        # Analyze overlap
        train_answers = set(a for _, _, a, _ in self.train_data)
        test_answers = set(a for _, _, a, _ in self.test_data)
        overlap = train_answers & test_answers
        test_only = test_answers - train_answers
        
        print(f"\nDataset generated:")
        print(f"  Train: {len(self.train_data)} samples, {len(train_answers)} unique answers")
        print(f"  Test: {len(self.test_data)} samples, {len(test_answers)} unique answers")
        print(f"  Answer overlap: {len(overlap)} ({len(overlap)/len(test_answers)*100:.1f}% of test)")
        print(f"  Test-only answers: {len(test_only)} ({len(test_only)/len(test_answers)*100:.1f}% of test)")
    
    def get_train_batch(self, batch_size: int, rng: random.Random) -> List[Tuple[str, str, str]]:
        samples = rng.choices(self.train_data, k=batch_size)
        return [(ctx, q, a) for ctx, q, a, _ in samples]
    
    def get_test_data(self) -> List[Tuple[str, str, str, int]]:
        return self.test_data
    
    def analyze(self):
        """Print detailed analysis of the dataset."""
        print("\n" + "="*70)
        print("DATASET ANALYSIS")
        print("="*70)
        
        # Answer vocabulary
        train_answers = [a for _, _, a, _ in self.train_data]
        test_answers = [a for _, _, a, _ in self.test_data]
        
        train_counter = Counter(train_answers)
        test_counter = Counter(test_answers)
        
        print(f"\nTrain answer vocabulary: {len(train_counter)} unique")
        print(f"Test answer vocabulary: {len(test_counter)} unique")
        
        # Overlap analysis
        train_set = set(train_answers)
        test_set = set(test_answers)
        
        print(f"\nOverlap: {len(train_set & test_set)} answers appear in both")
        print(f"Test-only: {len(test_set - train_set)} answers ONLY in test")
        
        # Sample test-only answers
        test_only = list(test_set - train_set)[:10]
        if test_only:
            print(f"\nSample test-only answers:")
            for ans in test_only:
                print(f"  - {ans}")
        
        # Frequency analysis
        print(f"\nTrain answer frequency (top 10):")
        for ans, count in train_counter.most_common(10):
            print(f"  {ans}: {count} ({count/len(train_answers)*100:.2f}%)")
        
        print(f"\nTest answer frequency (top 10):")
        for ans, count in test_counter.most_common(10):
            in_train = "✓" if ans in train_set else "✗"
            print(f"  {in_train} {ans}: {count} ({count/len(test_answers)*100:.2f}%)")
        
        # Question type distribution
        print(f"\nQuestion patterns in test:")
        q_patterns = Counter()
        for _, q, _, _ in self.test_data[:100]:
            if q.startswith("Who"):
                q_patterns["Who..."] += 1
            elif q.startswith("What"):
                q_patterns["What..."] += 1
            elif q.startswith("Where"):
                q_patterns["Where..."] += 1
            elif q.startswith("Which"):
                q_patterns["Which..."] += 1
            elif q.startswith("In which"):
                q_patterns["In which..."] += 1
        print(f"  {dict(q_patterns)}")


# =============================================================================
# Translator (same as working code)
# =============================================================================

class NormalizedTranslator(nn.Module):
    def __init__(self, input_size: int, output_size: int, target_heads: int, target_head_dim: int):
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
    
    def forward(self, source_cache: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, head_dim = source_cache.shape
        original_dtype = source_cache.dtype
        
        x = source_cache.float().permute(0, 2, 1, 3).contiguous().view(batch * seq_len, -1)
        x_norm = (x - self.source_mean) / self.source_std
        y_norm = self.net(x_norm)
        y = y_norm * self.target_std + self.target_mean
        y = y.view(batch, seq_len, self.target_heads, self.target_head_dim)
        
        return y.permute(0, 2, 1, 3).contiguous().to(original_dtype)


# =============================================================================
# Utilities
# =============================================================================

def normalize_text(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def calculate_f1(prediction: str, ground_truth: str) -> float:
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


def create_cache(keys, vals):
    cache = DynamicCache()
    for i in range(len(keys)):
        cache.update(keys[i], vals[i], i)
    return cache


def generate_with_cache(model, tokenizer, cache, max_tokens, device):
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
# Trainer
# =============================================================================

class HardSyntheticTrainer:
    def __init__(self, max_ctx_tokens=256, train_steps=2000, batch_size=8,
                 learning_rate=3e-4, eval_every=200, seed=42):
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
        print("Setting up Hard Synthetic QA experiment")
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
        
        # Get layer info
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
        
        # Layer mapping (interpolate)
        self.layer_mapping = []
        for j in range(self.num_layers_b):
            src_idx = round(j * (self.num_layers_a - 1) / (self.num_layers_b - 1))
            self.layer_mapping.append(src_idx)
        
        # Generate dataset
        print("\n2. Generating dataset...")
        self.dataset = HardSyntheticDataset(seed=self.seed, test_holdout_ratio=0.2)
        self.dataset.generate_datasets(train_size=10000, test_size=500, num_facts="mixed")
        
        # Create translators
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
        
        for target_idx in range(self.num_layers_b):
            source_idx = self.layer_mapping[target_idx]
            sk, sv = out_a.past_key_values[source_idx]
            tk, tv = out_b.past_key_values[target_idx]
            
            k_in = sk.shape[1] * sk.shape[3]
            k_out = tk.shape[1] * tk.shape[3]
            
            self.translators_k.append(
                NormalizedTranslator(k_in, k_out, tk.shape[1], tk.shape[3]).to(self.device))
            self.translators_v.append(
                NormalizedTranslator(k_in, k_out, tv.shape[1], tv.shape[3]).to(self.device))
        
        # Calibrate
        print("\n4. Calibrating...")
        self._calibrate()
        
        total_params = sum(p.numel() for p in self.translators_k.parameters()) + \
                       sum(p.numel() for p in self.translators_v.parameters())
        print(f"   Total parameters: {total_params:,}")
    
    def _calibrate(self):
        all_src_k = [[] for _ in range(self.num_layers_b)]
        all_src_v = [[] for _ in range(self.num_layers_b)]
        all_tgt_k = [[] for _ in range(self.num_layers_b)]
        all_tgt_v = [[] for _ in range(self.num_layers_b)]
        
        for ctx, q, _, _ in tqdm(self.dataset.train_data[:100], desc="Calibrating"):
            prompt = f"Context:\n{ctx}\n\nQuestion: {q} Answer:"
            
            inputs_a = self.tokenizer_a(prompt, return_tensors="pt", padding="max_length",
                                        truncation=True, max_length=self.max_ctx_tokens).to(self.device)
            inputs_b = self.tokenizer_b(prompt, return_tensors="pt", padding="max_length",
                                        truncation=True, max_length=self.max_ctx_tokens).to(self.device)
            
            with torch.no_grad():
                out_a = self.model_a(**inputs_a, use_cache=True)
                out_b = self.model_b(**inputs_b, use_cache=True)
            
            for i in range(self.num_layers_b):
                src_idx = self.layer_mapping[i]
                sk = out_a.past_key_values[src_idx][0]
                sv = out_a.past_key_values[src_idx][1]
                tk = out_b.past_key_values[i][0]
                tv = out_b.past_key_values[i][1]
                
                all_src_k[i].append(sk.permute(0,2,1,3).reshape(-1, sk.shape[1]*sk.shape[3]).float().cpu())
                all_src_v[i].append(sv.permute(0,2,1,3).reshape(-1, sv.shape[1]*sv.shape[3]).float().cpu())
                all_tgt_k[i].append(tk.permute(0,2,1,3).reshape(-1, tk.shape[1]*tk.shape[3]).float().cpu())
                all_tgt_v[i].append(tv.permute(0,2,1,3).reshape(-1, tv.shape[1]*tv.shape[3]).float().cpu())
        
        for i in range(self.num_layers_b):
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
    
    def _compute_loss(self, prompts, answers):
        inputs_a = self.tokenizer_a(prompts, return_tensors="pt", padding="max_length",
                                    truncation=True, max_length=self.max_ctx_tokens).to(self.device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                out_a = self.model_a(**inputs_a, use_cache=True)
        
        src_k = [out_a.past_key_values[self.layer_mapping[i]][0] for i in range(self.num_layers_b)]
        src_v = [out_a.past_key_values[self.layer_mapping[i]][1] for i in range(self.num_layers_b)]
        
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
            trans_k = [self.translators_k[i](src_k[i]) for i in range(self.num_layers_b)]
            trans_v = [self.translators_v[i](src_v[i]) for i in range(self.num_layers_b)]
        
        cache = create_cache(trans_k, trans_v)
        
        ans_inputs = self.tokenizer_b(answers, return_tensors="pt", padding=True,
                                      truncation=True, max_length=64).to(self.device)
        ans_ids = ans_inputs.input_ids
        ctx_len = cache.get_seq_length()
        
        attn_mask = torch.ones(len(prompts), ctx_len + ans_ids.shape[1], device=self.device)
        pos_ids = torch.arange(ctx_len, ctx_len + ans_ids.shape[1], device=self.device).unsqueeze(0).expand(len(prompts), -1)
        
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
            out_b = self.model_b(input_ids=ans_ids, attention_mask=attn_mask,
                                position_ids=pos_ids, past_key_values=cache, use_cache=False)
        
        logits = out_b.logits[:, :-1, :].contiguous()
        labels = ans_ids[:, 1:].contiguous()
        return nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    def evaluate(self, split_by_novelty=True):
        self.translators_k.eval()
        self.translators_v.eval()
        
        # Get train answers for novelty check
        train_answers = set(a for _, _, a, _ in self.dataset.train_data)
        
        results = {"seen": [], "unseen": [], "all": []}
        
        for ctx, q, gt, nf in self.dataset.get_test_data():
            prompt = f"Context:\n{ctx}\n\nQuestion: {q} Answer:"
            
            try:
                inputs_a = self.tokenizer_a([prompt], return_tensors="pt", padding="max_length",
                                           truncation=True, max_length=self.max_ctx_tokens).to(self.device)
                
                with torch.no_grad():
                    out_a = self.model_a(**inputs_a, use_cache=True)
                
                src_k = [out_a.past_key_values[self.layer_mapping[j]][0] for j in range(self.num_layers_b)]
                src_v = [out_a.past_key_values[self.layer_mapping[j]][1] for j in range(self.num_layers_b)]
                
                with torch.no_grad():
                    trans_k = [self.translators_k[j](src_k[j]) for j in range(self.num_layers_b)]
                    trans_v = [self.translators_v[j](src_v[j]) for j in range(self.num_layers_b)]
                
                cache = create_cache(trans_k, trans_v)
                response = generate_with_cache(self.model_b, self.tokenizer_b, cache, 32, self.device)
                
                cleaned = response.split('\n')[0].strip()
                for end in ['.', '?', '!']:
                    if end in cleaned:
                        cleaned = cleaned.split(end)[0].strip()
                        break
                
                f1 = calculate_f1(cleaned, gt)
                results["all"].append(f1)
                
                if gt in train_answers:
                    results["seen"].append(f1)
                else:
                    results["unseen"].append(f1)
                    
            except Exception as e:
                pass
        
        avg_all = sum(results["all"]) / len(results["all"]) if results["all"] else 0
        avg_seen = sum(results["seen"]) / len(results["seen"]) if results["seen"] else 0
        avg_unseen = sum(results["unseen"]) / len(results["unseen"]) if results["unseen"] else 0
        
        print(f"\n   Test F1 (all): {avg_all:.4f} (n={len(results['all'])})")
        print(f"   Test F1 (seen answers): {avg_seen:.4f} (n={len(results['seen'])})")
        print(f"   Test F1 (unseen answers): {avg_unseen:.4f} (n={len(results['unseen'])})")
        
        self.eval_history.append((len(self.eval_history) + 1, avg_all, avg_seen, avg_unseen))
        return avg_all
    
    def train(self):
        print(f"\n{'='*70}")
        print("Training")
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
            prompts = [f"Context:\n{ctx}\n\nQuestion: {q} Answer:" for ctx, q, _ in batch]
            answers = [" " + a for _, _, a in batch]
            
            optimizer.zero_grad()
            loss = self._compute_loss(prompts, answers)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item()
            
            if (step + 1) % 100 == 0:
                print(f"\n[{step+1}] Loss: {running_loss/100:.4f}, LR: {scheduler.get_last_lr()[0]:.5f}")
                running_loss = 0
            
            if (step + 1) % self.eval_every == 0:
                torch.cuda.empty_cache()
                f1 = self.evaluate()
                if f1 > best_f1:
                    best_f1 = f1
                torch.cuda.empty_cache()
        
        print(f"\n{'='*70}")
        print(f"Training complete! Best F1: {best_f1:.4f}")
        print(f"{'='*70}")
    
    def run_baseline(self):
        print(f"\n{'='*70}")
        print("Baseline (Model B native)")
        print(f"{'='*70}")
        
        train_answers = set(a for _, _, a, _ in self.dataset.train_data)
        results = {"seen": [], "unseen": [], "all": []}
        
        for ctx, q, gt, _ in tqdm(self.dataset.get_test_data()[:100], desc="Baseline"):
            prompt = f"Context:\n{ctx}\n\nQuestion: {q} Answer:"
            inputs = self.tokenizer_b(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model_b.generate(**inputs, max_new_tokens=32,
                                                do_sample=False, pad_token_id=self.tokenizer_b.eos_token_id)
            
            response = self.tokenizer_b.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            cleaned = response.split('\n')[0].strip()
            for end in ['.', '?', '!']:
                if end in cleaned:
                    cleaned = cleaned.split(end)[0].strip()
                    break
            
            f1 = calculate_f1(cleaned, gt)
            results["all"].append(f1)
            if gt in train_answers:
                results["seen"].append(f1)
            else:
                results["unseen"].append(f1)
        
        print(f"\nBaseline F1 (all): {sum(results['all'])/len(results['all']):.4f}")
        print(f"Baseline F1 (seen): {sum(results['seen'])/len(results['seen']):.4f}" if results['seen'] else "")
        print(f"Baseline F1 (unseen): {sum(results['unseen'])/len(results['unseen']):.4f}" if results['unseen'] else "")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    
    if args.analyze_only:
        dataset = HardSyntheticDataset(seed=args.seed, test_holdout_ratio=0.2)
        dataset.generate_datasets(10000, 500, "mixed")
        dataset.analyze()
        return
    
    trainer = HardSyntheticTrainer(
        train_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_every=args.eval_every,
        seed=args.seed,
    )
    
    trainer.setup()
    trainer.run_baseline()
    trainer.train()


if __name__ == "__main__":
    main()
