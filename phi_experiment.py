import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch
import argparse
import copy
from experiment_utils import *
import gc

BATCH_NUM = 1
qa_model = None
GPU_MAP = {0: "10GiB", 1: "30GiB", 2: "0GiB", 3: "30GiB", "cpu":"120GiB"}
INPUT_DEVICE = 'cuda:0'
MODEL_CONTEXT_LEN = 120000
NUM_GEN = 0

def get_args():
    parser = argparse.ArgumentParser('Run Global experiments')
    parser.add_argument('--qa_llm', type=str, default='microsoft/Phi-3-mini-128k-instruct')
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

def load_model(model_name="microsoft/Phi-3-mini-128k-instruct", only_tokenizer=False, gpu_map={0: "26GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB", "cpu":"120GiB"}):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not only_tokenizer:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, torch_dtype=torch.float16, trust_remote_code=True)
        # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
        # model.to(INPUT_DEVICE)
        # assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"
        return tokenizer, model
    else:
        return tokenizer, None

def query_model(prompts, model, tokenizer, do_sample=True, top_k=10, 
                num_return_sequences=1, max_length=240, temperature=1.0, INPUT_DEVICE='cuda:0'):
    global NUM_GEN
    NUM_GEN += 1
    # preprocess prompts:
    PHI_SYS_PROMPT = "You are a helpful AI assistant. who answers multiple choice reasoning questions in a specified format choosing from only the options available"
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
    torch.cuda.empty_cache()
    NUM_GEN += 1
    if NUM_GEN % 200 == 0:
        gc.collect()
    generated_ids= model.generate(input_ids, max_new_tokens=max_length, do_sample=do_sample, temperature=temperature)
    responses = tokenizer.batch_decode(generated_ids[:, input_ids.shape[-1]:].detach().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    del input_ids, generated_ids
    # print(responses)
    torch.cuda.empty_cache()
    return responses

def main():
    args = get_args()
    all_times, num_certificates_generated = run_experiment(args, load_model=load_model, query_model_func=query_model, 
                                                           GPU_MAP=GPU_MAP, BATCH_NUM=BATCH_NUM, INPUT_DEVICE=INPUT_DEVICE, model_context_length=MODEL_CONTEXT_LEN)
    print(f'Completed {num_certificates_generated} certificates')
    print(f'Average time = {np.mean(all_times)}')
if __name__ == '__main__':
    main()