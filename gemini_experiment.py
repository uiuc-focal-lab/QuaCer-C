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

def get_args():
    parser = argparse.ArgumentParser('Run Global experiments')
    parser.add_argument('--qa_llm', type=str, default='gemini-1.5-flash')
    parser.add_argument('--qa_graph_path', type=str, default='/home/vvjain3/new_repo/rag-llm-verify/wikidata_graphs/wikidata_util.json')
    parser.add_argument('--context_graph_edge_path', type=str, default='/home/vvjain3/new_repo/rag-llm-verify/wikidata_graphs/wikidata_text_edge.json')
    parser.add_argument('--results_dir', type=str, default='final_results/mistral_distractorfull/')
    parser.add_argument('--entity_aliases_path', type=str, default='/home/vvjain3/new_repo/rag-llm-verify/wikidata5m_entity.txt')
    parser.add_argument('--id2name_path', type=str, default='/home/vvjain3/new_repo/rag-llm-verify/wikidata_graphs/wikidata_name_id.json')
    parser.add_argument('--sentencized_path', type=str, default='/home/vvjain3/new_repo/rag-llm-verify/wikidata_graphs/wikidata_sentencized.json')
    parser.add_argument('--relation_aliases_path', type=str, default='/home/vvjain3/new_repo/rag-llm-verify/wikidata5m_relation.txt')
    parser.add_argument('--distractor_query', action='store_true', default=False, help=' best distractor based query?')
    parser.add_argument('--shuffle_context', action='store_true', default=False, help='Shuffle context in the context of query?')
    parser.add_argument('--k', type=int, default=4)

    parser.add_argument('--num_queries', type=int, default=1000)
    parser.add_argument('--num_certificates', type=int, default=50)
    return parser.parse_args()

def load_model(model_name="gemini-1.5-flash", only_tokenizer=False, gpu_map={0: "26GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB", "cpu":"120GiB"}):
    tokenizer = None
    if not only_tokenizer:
        genai.configure(api_key=os.environ["API_KEY"])
        model = genai.GenerativeModel(model_name)
        return tokenizer, model
    else:
        return tokenizer, None

def query_model(prompts, model, tokenizer, do_sample=True, top_k=10, 
                num_return_sequences=1, max_length=240, temperature=1.0, INPUT_DEVICE='cuda:0'):
    # preprocess prompts:
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