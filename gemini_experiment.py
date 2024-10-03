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

BATCH_NUM = 1
qa_model = None
GPU_MAP = {0: "6GiB", 1: "8GiB", 2: "8GiB", 3: "8GiB", "cpu":"120GiB"}
INPUT_DEVICE = 'cuda:1'
CONTINUOUS_SAFE = 0

def get_args():
    parser = get_base_args()
    parser.add_argument('--qa_llm', type=str, default='gemini-1.5-flash', help='Path to the QA model, like the huggingface model name or according to an API')
    parser.add_argument('--quant_type', type=str, default=None, choices=[None], help='quantization mode, always None for Gemini/GPT models')  # Explicitly set choices
    parser.set_defaults(num_queries=250) # override if needed
    return parser.parse_args()

def load_model(model_name="gemini-1.5-flash", only_tokenizer=False, gpu_map={0: "26GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB", "cpu":"120GiB"}, quant_type=None):
    global CONTINUOUS_SAFE
    tokenizer = None
    if not only_tokenizer:
        genai.configure(api_key=os.environ["API_KEY"])
        model = genai.GenerativeModel(model_name)
        return tokenizer, model
    else:
        return tokenizer, None

def query_model(prompts, model, tokenizer, do_sample=True, top_k=10, 
                num_return_sequences=1, max_length=120, temperature=1.0, INPUT_DEVICE='cuda:0'):
    # preprocess prompts:
    global CONTINUOUS_SAFE
    safe = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
    responses = []
    generation_config = genai.types.GenerationConfig(temperature=temperature, max_output_tokens=max_length)
    for prompt in prompts:
        try:
            response = model.generate_content(prompt, safety_settings=safe, generation_config=generation_config)
            responses.append(response.text)
            CONTINUOUS_SAFE = 0
        except Exception as e:
            #safety error
            print(e)
            response = "I'm sorry, I can't generate a response to that prompt."
            responses.append(response)
            time.sleep(1) #maybe rate limit error
            CONTINUOUS_SAFE += 1
            if CONTINUOUS_SAFE >= 4:
                print("Continuous safety error, too many", CONTINUOUS_SAFE, "possible rate limit error")
                exit(1)
    time.sleep(0.15)
    return responses

def main():
    args = get_args()
    all_times, num_certificates_generated = run_experiment(args, load_model=load_model, query_model_func=query_model, 
                                                           GPU_MAP=GPU_MAP, BATCH_NUM=BATCH_NUM, INPUT_DEVICE=INPUT_DEVICE, model_context_length=12800)
    print(f'Completed {num_certificates_generated} certificates')
    print(f'Average time = {np.mean(all_times)}')
if __name__ == '__main__':
    main()