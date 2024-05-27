import openvino as ov
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
import numpy as np
import os
import time

MODEL_ID = "meta-llama/Llama-2-7b-hf"
MODEL_DIR = ".ov_models/Llama2"
PROMPT = "The sun is yellow because"
MAX_NEW_TOKENS = 50

# Load the model and save it in IR format
if not os.path.exists(f"{MODEL_DIR}/openvino_model.xml"):
    model = OVModelForCausalLM.from_pretrained(MODEL_ID, load_in_8bit=True, export=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Compile model
print("[INFO] Preparing model")
core = ov.Core()
compiled_model = core.compile_model(f"{MODEL_DIR}/openvino_model.xml", "CPU")
lm = compiled_model.create_infer_request()

# Generate input tokens and start inference
print("[INFO] Running first inference to populate kv-cache")
inputs = tokenizer(PROMPT, return_tensors="pt")
inp_ids = inputs["input_ids"].numpy().astype(np.int64) 
att_mask = inputs["attention_mask"].numpy().astype(np.int64)
pos_ids = np.arange(inp_ids.shape[-1]).reshape(1, -1).astype(np.int64)
beam_idx = np.array([0], dtype=np.int32)

# Perform first inference to populate kv-cache
lm_inputs = {
    "input_ids": ov.Tensor(inp_ids),
    "attention_mask": ov.Tensor(att_mask),
    "position_ids": ov.Tensor(pos_ids),
    "beam_idx": ov.Tensor(beam_idx),
}
result = lm.infer(lm_inputs)
out_token = result["logits"][0, -1, :].argmax(-1)

# Run inference loop
print("[INFO] Starting inference")
seq_len = inp_ids.shape[-1] + 1
max_seq_len = seq_len + MAX_NEW_TOKENS
token_ids = inp_ids.reshape(-1).tolist()

generated_tokens = 0
total_time = 0.0

while (out_token != tokenizer.eos_token_id) and (seq_len < max_seq_len):
    inputs = {
        "input_ids": ov.Tensor(np.array([[out_token]], dtype=np.int64)), 
        "attention_mask": ov.Tensor(np.array([[1]], dtype=np.int64)),
        "position_ids": ov.Tensor(np.array([[seq_len - 1]], dtype=np.int64)),
        "beam_idx": ov.Tensor(np.array([0], dtype=np.int32)),
    }
    start = time.time()
    lm.start_async(inputs)
    lm.wait()
    end = time.time()
    total_time += (end - start) 

    logits = lm.get_output_tensor(0).data
    out_token = logits[0, -1, :].argmax(-1).item()
    seq_len += 1
    generated_tokens += 1
    token_ids.append(out_token)
    
# Print response
response = tokenizer.decode(token_ids, skip_special_tokens=True)
print("Response:", response)
    
lm.cancel()
    
print("\n=========================================")
print("Total time (sec) :", total_time)
print("Total tokens     :", generated_tokens)
print("Tokens / sec     :", generated_tokens / total_time)