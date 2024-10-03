import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch
import argparse
import copy
from experiment_utils import *
import gc
import time

BATCH_NUM = 1
qa_model = None
GPU_MAP = {0: "40GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB", "cpu":"120GiB"}
INPUT_DEVICE = 'cuda:0'
MODEL_CONTEXT_LEN = 120000
NUM_GEN = 0

def get_args():
    parser = get_base_args()
    parser.add_argument('--qa_llm', type=str, default='microsoft/Phi-3-mini-128k-instruct', help='Path to the QA model, like the huggingface model name or according to an API')
    parser.add_argument('--quant_type', type=str, default=None, choices=['8_bit', '4_bit'], help='quantization mode')  # Explicitly set choices
    parser.set_defaults(num_queries=250) # override if needed
    return parser.parse_args()

def load_model(model_name="microsoft/Phi-3-mini-128k-instruct", only_tokenizer=False, gpu_map={0: "26GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB", "cpu":"120GiB"}, quant_type=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not only_tokenizer:
        if quant_type is not None:
            if quant_type == '8_bit':
                print("loading 8 bit model")
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, torch_dtype=torch.float16, load_in_8bit=True, trust_remote_code=True, attn_implementation='flash_attention_2')
            elif quant_type == '4_bit':
                print("loading 4 bit model")
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, bnb_4bit_quant_type="nf4", load_in_4bit=True,  bnb_4bit_compute_dtype=torch.float16, trust_remote_code=True, attn_implementation='flash_attention_2')
        else:    
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation='flash_attention_2')
        # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
        for name, param in model.named_parameters():
            if 'cuda' not in str(param.device):
                print(f"param {name} not on cuda")
        # assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"
        return tokenizer, model
    else:
        return tokenizer, None

def query_model(prompts, model, tokenizer, do_sample=True, top_k=10, 
                num_return_sequences=1, max_length=240, temperature=1.0, INPUT_DEVICE='cuda:0'):
    global NUM_GEN
    NUM_GEN += 1
    # preprocess prompts:
    start_time = time.time()
    PHI_SYS_PROMPT = "You are a helpful AI assistant. Answer questions as needed"
    chats = []
    if len(prompts) > 1:
        for prompt in prompts:
            message_template = [{"role": "system", "content": PHI_SYS_PROMPT}, {"role":"user", "content":f"{prompt}"}]
            chats.append([copy.deepcopy(message_template)])
    else:
        chats = [{"role": "system", "content": PHI_SYS_PROMPT}, {"role":"user", "content":f"{prompts[0]}"}]
        
    input_ids = tokenizer.apply_chat_template(chats, return_tensors="pt", add_generation_prompt=True, padding=True).to(INPUT_DEVICE)
    if input_ids.shape[-1] > 128000:
        print("Input too long, input too long, number of tokens: ", input_ids.shape)
        input_ids = input_ids[:, :128000]
    
    NUM_GEN += 1
    torch.cuda.empty_cache()
    if NUM_GEN % 50 == 0:
        gc.collect()

    start_t = time.time()
    generated_ids= model.generate(input_ids, max_new_tokens=max_length, do_sample=do_sample, temperature=temperature)
    print(f"Time taken for generating: {time.time() - start_t}")
    responses = tokenizer.batch_decode(generated_ids[:, input_ids.shape[-1]:].detach().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    del input_ids, generated_ids
    # print(responses)
    # torch.cuda.empty_cache()
    print(f"Time taken for generating: {time.time() - start_time}")
    return responses

def main():
    args = get_args()
    all_times, num_certificates_generated = run_experiment(args, load_model=load_model, query_model_func=query_model, 
                                                           GPU_MAP=GPU_MAP, BATCH_NUM=BATCH_NUM, INPUT_DEVICE=INPUT_DEVICE, model_context_length=MODEL_CONTEXT_LEN)
    print(f'Completed {num_certificates_generated} certificates')
    print(f'Average time = {np.mean(all_times)}')
if __name__ == '__main__':
    main()