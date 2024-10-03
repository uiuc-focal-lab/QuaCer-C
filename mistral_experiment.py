import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch
import argparse
import copy
from experiment_utils import *

BATCH_NUM = 1
qa_model = None
GPU_MAP = {0: "0GiB", 1: "20GiB", 2: "20GiB", 3: "0GiB", "cpu":"0GiB"}
INPUT_DEVICE = 'cuda:1'
MAX_CONTEXT_LEN = 28000
NUM_GEN = 0

def get_args():
    parser = get_base_args()
    parser.add_argument('--qa_llm', type=str, default='mistralai/Mistral-7B-Instruct-v0.2', help='Path to the QA model, like the huggingface model name or according to an API')
    parser.add_argument('--quant_type', type=str, default=None, choices=['8_bit', '4_bit'], help='quantization mode')  # Explicitly set choices
    parser.set_defaults(num_queries=250) # override if needed
    return parser.parse_args()

def load_model(model_name="mistralai/Mistral-7B-Instruct-v0.2", only_tokenizer=False, gpu_map={0: "26GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB", "cpu":"120GiB"}, quant_type=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if not only_tokenizer:
        if quant_type is not None:
            if quant_type == '8_bit':
                print("loading 8 bit model")
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, torch_dtype=torch.float16, load_in_8bit=True)
            elif quant_type == '4_bit':
                print("loading 4 bit model")
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, bnb_4bit_quant_type="nf4", load_in_4bit=True,  bnb_4bit_compute_dtype=torch.float16)
        else:
            print('no quantization, loading in fp16')
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, torch_dtype=torch.float16)
        #check device of all model tensors
        for name, param in model.named_parameters():
            if 'cuda' not in str(param.device):
                print(f"param {name} not on cuda")
        return tokenizer, model
    else:
        return tokenizer, None

def query_model(prompts, qa_model, tokenizer, do_sample=True, top_k=10, 
                num_return_sequences=1, max_length=240, temperature=1.0, INPUT_DEVICE='cuda:0'):
    global NUM_GEN
    # preprocess prompts:
    import time
    assert len(prompts) == 1
    
    messages = [{"role": "user", "content": f"{prompts[0]}"},]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(INPUT_DEVICE)
    start_time = time.time()
    generated_ids = qa_model.generate(model_inputs, max_new_tokens=max_length, do_sample=do_sample, temperature=temperature)
    print(f"Time taken for model: {time.time() - start_time}")
    generated_ids = generated_ids[:, model_inputs.shape[-1]:]
    decoded = tokenizer.batch_decode(generated_ids.detach().cpu())

    NUM_GEN += 1
    if NUM_GEN % 50 == 0:
        torch.cuda.empty_cache()
        gc.collect()
    model_ans = decoded[0].strip()
    del model_inputs, generated_ids
    return [model_ans]

def main():
    args = get_args()
    all_times, num_certificates_generated = run_experiment(args, load_model=load_model, query_model_func=query_model, 
                                                           GPU_MAP=GPU_MAP, BATCH_NUM=BATCH_NUM, INPUT_DEVICE=INPUT_DEVICE, model_context_length=MAX_CONTEXT_LEN)
    print(f'Completed {num_certificates_generated} certificates')
    print(f'Average time = {np.mean(all_times)}')
if __name__ == '__main__':
    main()