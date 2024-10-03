import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch
import argparse
import copy
from experiment_utils import *
import google.generativeai as genai
import time
from openai import OpenAI


BATCH_NUM = 1
qa_model = None
GPU_MAP = {0: "6GiB", 1: "8GiB", 2: "8GiB", 3: "8GiB", "cpu":"120GiB"}
INPUT_DEVICE = 'cuda:1'

def get_args():
    parser = get_base_args()
    parser.add_argument('--qa_llm', type=str, default='gpt-4o-mini', help='Path to the QA model, like the huggingface model name or according to an API')
    parser.add_argument('--quant_type', type=str, default=None, choices=[None], help='quantization mode, always None for Gemini/GPT models')  # Explicitly set choices
    parser.set_defaults(num_queries=250) # override if needed
    return parser.parse_args()

def load_model(model_name="gpt-4o", only_tokenizer=False, gpu_map={0: "26GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB", "cpu":"120GiB"}):
    tokenizer = None
    if not only_tokenizer:
        model = OpenAI()
        return tokenizer, model
    else:
        return tokenizer, None

def query_model(prompts, model, tokenizer, do_sample=True, top_k=10, 
                num_return_sequences=1, max_length=240, temperature=1.0, INPUT_DEVICE='cuda:0'):
    # preprocess prompts:
    responses = []
    for prompt in prompts:
        try:
            completion = model.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{prompt}"},
                ],
                temperature=temperature,
                max_tokens=max_length,
                )
            responses.append(completion.choices[0].message.content)
        except Exception as e:
            #safety error
            print(e)
            response = "I'm sorry, I can't generate a response to that prompt."
            responses.append(response)
    time.sleep(0.01)
    return responses

def main():
    args = get_args()
    all_times, num_certificates_generated = run_experiment(args, load_model=load_model, query_model_func=query_model, 
                                                           GPU_MAP=GPU_MAP, BATCH_NUM=BATCH_NUM, INPUT_DEVICE=INPUT_DEVICE, model_context_length=7200)
    print(f'Completed {num_certificates_generated} certificates')
    print(f'Average time = {np.mean(all_times)}')
if __name__ == '__main__':
    main()