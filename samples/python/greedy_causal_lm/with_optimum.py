from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from tqdm import tqdm
import time
import os

MODEL_ID = "meta-llama/Llama-2-7b-hf"
MODEL_DIR = ".ov_models/Llama2"
PROMPT = "The sun is yellow because"

if not os.path.exists(f"{MODEL_DIR}/openvino_model.xml"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = OVModelForCausalLM.from_pretrained(MODEL_ID, load_in_8bit=True, export=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = OVModelForCausalLM.from_pretrained(MODEL_DIR, load_in_8bit=True)

total_time = 0.0
generated_tokens = 0
inputs = tokenizer.encode(PROMPT, return_tensors="pt")

for _ in tqdm(range(2)):
    start = time.time()
    result = model.generate(input_ids=inputs, max_new_tokens=10)
    end = time.time()
    total_time += end - start
    generated_tokens += result.shape[-1] - inputs.shape[-1]
    
    response = tokenizer.batch_decode(result, skip_special_tokens=True)
    print("Respose:", response)
    
print("* Total tokens:", generated_tokens)
print("* Total time (sec):", total_time)
print("* Tokens/sec:", generated_tokens / total_time)