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
    parser = argparse.ArgumentParser('Run Global experiments')
    parser.add_argument('--qa_llm', type=str, default='gpt-4o-mini')
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