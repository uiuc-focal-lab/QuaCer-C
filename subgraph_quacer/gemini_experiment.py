import numpy as np
import argparse
import torch
import google.generativeai as genai
import time
from experiment_utils import *
from experiment_utils import get_base_args, run_experiment
from subgraph_utils import *
from discovery_functions import *

BATCH_NUM = 1
GPU_MAP = {0: "6GiB", 1: "8GiB", 2: "8GiB", 3: "8GiB", "cpu":"120GiB"}
INPUT_DEVICE = 'cuda:1'
CONTINUOUS_SAFE = 0

def get_args():
    parser = get_base_args()
    parser.add_argument('--qa_llm', type=str, default='gemini-1.5-flash',
                       help='Model name for Gemini API')
    parser.add_argument('--quant_type', type=str, default=None, 
                       choices=[None], help='No quantization for API models')
    parser.set_defaults(num_queries=250)
    return parser.parse_args()

def load_model(model_name="gemini-1.5-flash", only_tokenizer=False, 
              gpu_map=None, quant_type=None):
    global CONTINUOUS_SAFE
    tokenizer = None
    if not only_tokenizer:
        genai.configure(api_key='')
        model = genai.GenerativeModel(model_name)
        return tokenizer, model
    else:
        return tokenizer, None

def query_model(prompts, model, tokenizer, do_sample=True, top_k=10,
                num_return_sequences=1, max_length=120, temperature=1.0, 
                INPUT_DEVICE='cuda:0'):
    global CONTINUOUS_SAFE
    safe = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    responses = []
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_length
    )
    
    for prompt in prompts:
        try:
            response = model.generate_content(
                prompt,
                safety_settings=safe,
                generation_config=generation_config
            )
            responses.append(response.text)
            CONTINUOUS_SAFE = 0
        except Exception as e:
            print(e)
            response = "I'm sorry, I can't generate a response to that prompt."
            responses.append(response)
            time.sleep(1)
            CONTINUOUS_SAFE += 1
            if CONTINUOUS_SAFE >= 4:
                print("Continuous safety errors:", CONTINUOUS_SAFE)
                exit(1)
    
    time.sleep(0.15)
    return responses

def main():
    args = get_args()
    
    # Define all discovery functions and their names
    discovery_funcs = [
        lambda graph, **kwargs: off_label_discoverer(graph, **kwargs),
        lambda graph, **kwargs: dual_indication_discoverer(graph, **kwargs),
        lambda graph, **kwargs: synergistic_discoverer(graph, **kwargs),
        lambda graph, **kwargs: gene_target_discoverer(graph, **kwargs),
        lambda graph, **kwargs: phenotype_drug_contraindication_discoverer(graph, **kwargs),
        lambda graph, **kwargs: drug_contraindication_discoverer(graph, **kwargs),
        lambda graph, **kwargs: exposure_drug_discoverer(graph, **kwargs),
        lambda graph, **kwargs: phenotype_group_disease_discoverer(graph, **kwargs),
        lambda graph, **kwargs: least_side_effects_discoverer(graph, **kwargs),
        lambda graph, **kwargs: contraindication_indication_discoverer(graph, **kwargs)
    ]

    discovery_names = [
        'off_label',
        'dual_indication',
        'synergistic',
        'gene_target',
        'phenotype_drug_contraindication',
        'drug_contraindication',
        'exposure_drug',
        'phenotype_group_disease',
        'least_side_effects',
        'contraindication_indication'
    ]
    
    # Can specify which certificates to generate:
    # discovery_idx = [0, 2]  # Only generate certificates for specific functions
    # discovery_idx = 0       # Generate certificate for a single function
    discovery_idx = None      # Generate all missing certificates
    
    # Run experiment
    results = run_experiment(
        args,
        load_model=load_model,
        query_model_func=query_model,
        discovery_funcs=discovery_funcs,
        discovery_names=discovery_names,
        GPU_MAP=GPU_MAP,
        BATCH_NUM=BATCH_NUM,
        INPUT_DEVICE=INPUT_DEVICE,
        model_context_length=12800,
        discovery_idx=discovery_idx
    )
    
    # Print results
    print("\nExperiment Results:")
    for func_name, result in results.items():
        if result["completed"]:
            print(f"{func_name}: Generated successfully - Time: {result['time']:.2f} seconds")
        else:
            print(f"{func_name}: Already existed - Skipped")

if __name__ == "__main__":
    main()