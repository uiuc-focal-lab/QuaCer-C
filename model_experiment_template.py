import numpy as np
from experiment_utils import *

# Define constants
BATCH_NUM = 1
GPU_MAP = {0: "40GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB", "cpu":"120GiB"}
INPUT_DEVICE = 'cuda:0'

def get_args():
    parser = get_base_args()
    parser.add_argument('--qa_llm', type=str, default='path/to/model')
    parser.add_argument('--quant_type', type=str, default=None, choices=[None]) # This is needed always, set to None if not used in your load_model and query_model functions
    # Add any model-specific arguments here
    return parser.parse_args()

def load_model(model_name, only_tokenizer=False, gpu_map=GPU_MAP, quant_type=None):
    # TODO: Implement model loading logic
    pass

def query_model(prompts, model, tokenizer, do_sample=True, top_k=10, 
                num_return_sequences=1, max_length=240, temperature=1.0, INPUT_DEVICE='cuda:0'):
    # TODO: Implement model querying logic
    pass

def main():
    args = get_args()
    all_times, num_certificates_generated = run_experiment(args, load_model=load_model, 
                                                           query_model_func=query_model, 
                                                           GPU_MAP=GPU_MAP, BATCH_NUM=BATCH_NUM, 
                                                           INPUT_DEVICE=INPUT_DEVICE)
    print(f'Completed {num_certificates_generated} certificates')
    print(f'Average time = {np.mean(all_times)}')

if __name__ == '__main__':
    main()