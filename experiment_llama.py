import os
import json
import numpy as np
from utils import *
import argparse
from statsmodels.stats.proportion import proportion_confint
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
from unidecode import unidecode
import pickle
import time
import torch
import gc
from llama_local import example_chat_completion as ecc
import socket

BATCH_NUM = 2
qa_model = None
checker_host, checker_port = None, None

def get_args():
    parser = argparse.ArgumentParser('Run Global experiments')
    llama_path = os.getenv('LLAMA_PATH')
    parser.add_argument('--qa_llm_path', type=str, default=os.path.join(llama_path, 'llama-2-7b-chat'))
    parser.add_argument('--tokenizer_path', type=str, default=os.path.join(llama_path, 'tokenizer.model'))
    parser.add_argument('--qa_graph_path', type=str, default='wikidata5m_en_util_unidecoded.json')
    parser.add_argument('--context_graph_path', type=str, default='wikidata5m_en_text.json')
    parser.add_argument('--results_path', type=str, default='results_exprimentmist.pkl')
    parser.add_argument('--entity_aliases_path', type=str, default='wikidata5m_entity.txt')
    parser.add_argument('--id2name_path', type=str, default='wikidata_name_id_uni1.json')
    parser.add_argument('--relation_aliases_path', type=str, default='wikidata5m_relation.txt')

    parser.add_argument('--num_queries', type=int, default=1000)

    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    return parser.parse_args()

def simple_checker(model_answer, correct_answer, correct_answer_aliases, name2id):
    model_answer = unidecode(model_answer).lower()
    correct_answer = unidecode(correct_answer).lower()
    if correct_answer in model_answer:
        return 1

    if correct_answer not in name2id:
        return 0
    correct_id = name2id[correct_answer]
    if correct_id not in correct_answer_aliases:
        return 0
    for answer_alias in correct_answer_aliases[correct_id]:
        if answer_alias in model_answer:
            return 1
    return 0

def check_answer(question, correct_answer, model_answer, entity_aliases, name2id):
    global qa_model, checker_host, checker_port
    if simple_checker(model_answer, correct_answer, entity_aliases, name2id) == 1:
        return 1
    result_dict = None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((checker_host, checker_port))
        # Prepare and send a request
        request_data = {'question':question, 'correct_answer':correct_answer, 'model_answer': model_answer}
        s.sendall(json.dumps(request_data).encode('utf-8'))
        response = s.recv(1024)
        result_dict = json.loads(response.decode('utf-8'))
    if result_dict is not None:
        return result_dict['result']
    raise RuntimeError('Checker server did not return a result')
    return 0

def build_llama(llm_path, tokenizer_path, max_length=4096):
    # return ecc.build_llama_model('/home/vvjain3/rag-llm-verify/llama/llama-2-13b-chat', '/home/vvjain3/rag-llm-verify/llama/tokenizer.model', max_seq_len=max_length, max_batch_size=BATCH_NUM)
    return ecc.build_llama_model(llm_path, tokenizer_path, max_seq_len=max_length, max_batch_size=BATCH_NUM)

def query_llama_model(sys_prompts, prompts, gen, do_sample=True, top_k=10, 
                num_return_sequences=1, temperature=1.0):
    
    # preprocess prompts:
    # command: torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:25200 --nproc_per_node 1 attack.py --ckpt_dir /share/models/llama2/llama-2-7b-chat/ --tokenizer_path /share/models/llama2/tokenizer.model --max_seq_len 1024 --max_batch_size 40
    prom = []
    # print('querying llama model')
    for s, p in zip(sys_prompts,prompts):
        prom.append([
                {"role": "system", "content": s,},
                {"role": "user", "content": p},
            ])
        # print('Prompts:', prom)
        
    out = ecc.my_llama(prom, gen, temperature=temperature, max_gen_len=40)
    # print("Output=",out)
    # exit(0)
    out_list = [o['generation']['content'] for o in out]
    
    return out_list

def experiment_pipeline(graph_algos, k=5, num_queries=5, graph_text=None, entity_aliases=None, name2id=None, source=None, relation_aliases=None):
    global qa_model, tokenizer, checker_llm
    results = []
    correct = 0
    total = 0
    all_queries = []
    all_keys = list(graph_text.keys())
    sys_prompts = ['You are a helpful honest assistant question answerer that answers queries succintly' for _ in range(BATCH_NUM)]
    with torch.no_grad():
        for num_iter in range(num_queries//BATCH_NUM):
            torch.cuda.empty_cache()
            gc.collect()
            prompts = []
            queries_data = []
            for j in range(BATCH_NUM):
                prompt = None
                while prompt is None:
                    query_results = graph_algos.generate_random_query(k, return_path=True, source=source) # allow sampling with replacement
                    query_inf, _, correct_answer, path = query_results
                    query, k_num = query_inf
                    query, entity_alias = form_alias_question(query, path, entity_aliases, relation_aliases, name2id, graph_algos)
                    if 'country of' in query:
                        query = query.replace('country of', 'country location of') # to avoid ambiguity
                    all_queries.append(query)
                    add_keys = random.sample(all_keys, 1)
                    true_path = path.copy()
                    for key in add_keys:
                        if key not in path:
                            path.append(key)
                    random.shuffle(path)
                    context = ""
                    for i, key in enumerate(path):
                        if i == 0:
                            all_aliases_text = f"{key} is also known as {entity_alias}."
                        else:
                            all_aliases_text = ''
                        context += graph_text[key] + all_aliases_text + "\n"
                    prompt = f"Given the context: {context} Answer the query: {query}. Start with the answer in one word or phrase."
                    if len(prompt) > 13000:
                        prompt = None # too long, get another query
                        continue
                    prompts.append(prompt)
                    queries_data.append({'query':query, 'correct_answer':correct_answer, 'path':true_path, 'context':context})
            model_answers= query_llama_model(sys_prompts, prompts, qa_model)
            for i, model_ans in enumerate(model_answers):
                model_ans = model_ans.strip()
                model_answers[i] = model_ans
            assert len(queries_data) == len(model_answers)
            for i in range(len(queries_data)):
                query = queries_data[i]['query']
                correct_answer = queries_data[i]['correct_answer']
                path = queries_data[i]['path']
                context = queries_data[i]['context']
                model_ans = model_answers[i]
                eval_ans = check_answer(question=query, correct_answer=correct_answer, model_answer=model_ans, entity_aliases=entity_aliases, name2id=name2id)
                results.append({ 'question':query, 'correct_answer':correct_answer, 'model_answer':model_ans, 
                                'path':path, 'context':context, 'result':(eval_ans, None)})
                print(correct_answer, model_ans, eval_ans)
                correct += results[-1]['result'][0]
                total += 1
            print(f'Completed {num_iter} queries, {correct} correct out of {total} total')
            interval_conf = proportion_confint(correct, total, method='beta', alpha=0.05)
            low = round(interval_conf[0], 2)
            up = round(interval_conf[1], 2)
            if round(up - low, 2) < 0.1:
                break
            del model_answers
            torch.cuda.empty_cache()
            gc.collect()
        print(f'Completed {num_queries} queries, {correct} correct out of {total} total')
    return results, correct, total

def main():
    global qa_model, checker_host, checker_port
    args = get_args()
    qa_model = build_llama(args.qa_llm_path, args.tokenizer_path)
    qa_graph = json.load(open(args.qa_graph_path))
    context_graph = json.load(open(args.context_graph_path))
    id2name = json.load(open(args.id2name_path))
    name2id = {v:k for k,v in id2name.items()}
    checker_host, checker_port = args.host, args.port
    entity_aliases = load_aliases(args.entity_aliases_path)
    relation_aliases = load_aliases(args.relation_aliases_path)

    count = 0
    all_times = []
    qa_graph_algos = GraphAlgos(qa_graph)
    best_vertices = qa_graph_algos.get_best_vertices(num=1000)
    random.shuffle(best_vertices)
    for i, vertex_id in enumerate(best_vertices):
        vertex = id2name[vertex_id]
        start_time = time.time()
        subgraph = qa_graph_algos.create_subgraph_within_radius(vertex, 4)
        subgraph_algos = GraphAlgos(subgraph)
        if len(subgraph) < 40:
            continue
        print(vertex, name2id[vertex])
        expriment_results = experiment_pipeline(graph_algos=subgraph_algos, k=4, num_queries=args.num_queries, 
                                                graph_text=context_graph, entity_aliases=entity_aliases, 
                                                name2id=name2id, source=vertex, relation_aliases=relation_aliases)
        end_time = time.time()
        print(f'Time taken for {vertex} = {end_time - start_time}')
        all_times.append(end_time - start_time)
        with open(args.results_path[:-4]+str(name2id[vertex])+'.pkl', 'wb') as f:
            pickle.dump(expriment_results, f)
        count += 1
        break
    print(f'Average time = {np.mean(all_times)}')
if __name__ == '__main__':
    main()