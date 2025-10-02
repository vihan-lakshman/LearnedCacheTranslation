import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
import traceback
import sys

# Suppress informational warnings
logging.set_verbosity_error()

if sys.argv[1] == 'qwen':
    MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"  # Source Model
    MODEL_B = "Qwen/Qwen2.5-7B-Instruct"   # Target Model
elif sys.argv[1] == 'mistral':
    MODEL_A = "mistralai/Mistral-7B-Instruct-v0.2"  # Source Model
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"    # Target Model
else:
    raise ValueError("Invalid model")

if "Qwen" in MODEL_A:
    NUM_LAYERS = 28
elif "Mistral" in MODEL_A:
    NUM_LAYERS = 32
else:
    raise ValueError("Invalid model")

LOAD_PATH = "kv_translators.pth" # Path to load the trained models from

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CTX_TOKENS = 20
MIXED_PRECISION = True


class FastKVTranslator(nn.Module):
    def __init__(self, input_size, output_size, target_heads, target_head_dim):
        super().__init__()
        self.target_heads, self.target_head_dim = target_heads, target_head_dim
        hidden_size = (input_size + output_size) // 2
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, output_size, bias=False),
        )

    def forward(self, cache_tensor_a):
        batch, _, seq_len, _ = cache_tensor_a.shape
        x = cache_tensor_a.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch * seq_len, -1)
        y = self.net(x)
        y = y.view(batch, seq_len, self.target_heads, self.target_head_dim)
        return y.permute(0, 2, 1, 3).contiguous()


def gen_secret_phrase():
    templates = ["The agent's secret activation code is {}-{}-{}."]
    codenames = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
    template = random.choice(templates)
    a, b = random.sample(codenames, 2)
    c = random.randint(100, 999)
    return template.format(a, b, c)


def generate_kv_cache_pair_batch(prompts, tokenizer_a, model_a, device, max_length):
    inputs_a = tokenizer_a(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=MIXED_PRECISION):
            out_a = model_a(**inputs_a, use_cache=True)
        kv_a = out_a.past_key_values
    source_keys = [kv_a[i][0] for i in range(NUM_LAYERS)]
    source_vals = [kv_a[i][1] for i in range(NUM_LAYERS)]
    del out_a, kv_a, inputs_a
    return source_keys, source_vals


def load_models():
    print("--- Loading Base Models and Tokenizers ---")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B, trust_remote_code=True)

    if tokenizer_a.pad_token is None: tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None: tokenizer_b.pad_token = tokenizer_b.eos_token

    model_a = AutoModelForCausalLM.from_pretrained(MODEL_A, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_B, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    model_a.eval(); model_b.eval()
    
    print("--- Determining Translator Dimensions ---")
    
    # Get source dimensions from Model A (1.5B)
    dummy_sk, dummy_sv = generate_kv_cache_pair_batch(["dummy"], tokenizer_a, model_a, DEVICE, MAX_CTX_TOKENS)
    
    # Get target dimensions from Model B (7B) by running a separate forward pass
    with torch.no_grad():
        dummy_inputs_b = tokenizer_b(["dummy"], return_tensors="pt", padding="max_length", max_length=MAX_CTX_TOKENS).to(DEVICE)
        dummy_out_b = model_b(**dummy_inputs_b, use_cache=True)
        dummy_kv_b = dummy_out_b.past_key_values
    dummy_tk = [dummy_kv_b[i][0] for i in range(NUM_LAYERS)]
    dummy_tv = [dummy_kv_b[i][1] for i in range(NUM_LAYERS)]
    del dummy_out_b, dummy_kv_b, dummy_inputs_b # Clean up GPU memory

    print("--- Creating Empty Translator Models with Correct Dimensions ---")
    translators_k, translators_v = nn.ModuleList(), nn.ModuleList()
    for i in range(NUM_LAYERS):
        # Now, sk/sv and tk/tv have the correct, potentially different, dimensions
        sk, sv, tk, tv = dummy_sk[i], dummy_sv[i], dummy_tk[i], dummy_tv[i]
        
        k_in_size = sk.shape[1] * sk.shape[3]; k_out_size = tk.shape[1] * tk.shape[3]
        v_in_size = sv.shape[1] * sv.shape[3]; v_out_size = tv.shape[1] * tv.shape[3]
        
        translators_k.append(FastKVTranslator(k_in_size, k_out_size, tk.shape[1], tk.shape[3]))
        translators_v.append(FastKVTranslator(v_in_size, v_out_size, tv.shape[1], tv.shape[3]))

    print(f"--- Loading Trained Weights from {LOAD_PATH} ---")
    checkpoint = torch.load(LOAD_PATH, map_location=DEVICE)
    
    k_state_dict = checkpoint['translators_k_state_dict']
    v_state_dict = checkpoint['translators_v_state_dict']

    # Clean the keys by removing the '_orig_mod.' prefix if it exists
    clean_k_state_dict = {k.replace('_orig_mod.', ''): v for k, v in k_state_dict.items()}
    clean_v_state_dict = {k.replace('_orig_mod.', ''): v for k, v in v_state_dict.items()}

    # Load the cleaned state dictionaries
    translators_k.load_state_dict(clean_k_state_dict)
    translators_v.load_state_dict(clean_v_state_dict)

    translators_k.to(DEVICE).eval()
    translators_v.to(DEVICE).eval()
    
    return model_a, model_b, tokenizer_a, tokenizer_b, translators_k, translators_v


def run_test():
    model_a, model_b, tokenizer_a, tokenizer_b, translators_k, translators_v = load_models()
    print("\n--- Starting Translation Test ---")
    success_count = 0
    NUM_TESTS = 10
    for test_i in range(NUM_TESTS):
        print(f"\n--- Test {test_i + 1} ---")
        unseen_secret = gen_secret_phrase()
        print(f"Secret Context: '{unseen_secret}'")

        try:
            sk, sv = generate_kv_cache_pair_batch([unseen_secret], tokenizer_a, model_a, DEVICE, MAX_CTX_TOKENS)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
                    translated_k = [translators_k[i](sk[i]) for i in range(NUM_LAYERS)]
                    translated_v = [translators_v[i](sv[i]) for i in range(NUM_LAYERS)]
            
            translated_cache = tuple(zip(translated_k, translated_v))
            
            question = " Repeat the secret context"
            q_inputs = tokenizer_b(question, return_tensors="pt").to(DEVICE)
            
            context_len = translated_cache[0][0].shape[2]
            question_len = q_inputs.input_ids.shape[1]
            attention_mask = torch.ones(1, context_len + question_len, device=DEVICE)
            cache_position = torch.arange(context_len, context_len + question_len, device=DEVICE)

            with torch.no_grad():
                generated = model_b.generate(
                    input_ids=q_inputs.input_ids, 
                    attention_mask=attention_mask,
                    past_key_values=translated_cache,
                    cache_position=cache_position,
                    max_new_tokens=30, 
                    do_sample=False, 
                    pad_token_id=tokenizer_b.eos_token_id)
            
            response = tokenizer_b.decode(generated[0][question_len:], skip_special_tokens=True).strip()
            print(f"Model Response: '{response}'")

            core_code = [p for p in unseen_secret.split() if "-" in p][0].replace('.','')
            if core_code.lower() in response.lower():
                success_count += 1
                print("SUCCESS")
            else:
                print("FAILURE")
        except Exception as e:
            print(f"Test failed with an error: {e}")
            traceback.print_exc()

    print(f"\nFinal Results: {success_count}/{NUM_TESTS} successful transfers.")


if __name__ == "__main__":
    run_test()
