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

BATCH_NUM = 1
qa_model = None
checker_host, checker_port = None, None

def get_args():
    parser = argparse.ArgumentParser('Run Global experiments')
    llama_path = os.getenv('LLAMA_PATH')
    parser.add_argument('--qa_llm_path', type=str, default='meta-llama/Llama-2-13b-chat-hf')
    parser.add_argument('--qa_graph_path', type=str, default='/home/vvjain3/new_repo/rag-llm-verify/wikidata_graphs/wikidata_util.json')
    parser.add_argument('--context_graph_path', type=str, default='/home/vvjain3/new_repo/rag-llm-verify/wikidata_graphs/wikidata_text.json')
    parser.add_argument('--results_dir', type=str, default='/home/vvjain3/new_repo/rag-llm-verify/16bit_llama13/')
    parser.add_argument('--entity_aliases_path', type=str, default='/home/vvjain3/rag-llm-verify/wikidata5m_entity.txt')
    parser.add_argument('--id2name_path', type=str, default='/home/vvjain3/new_repo/rag-llm-verify/wikidata_graphs/wikidata_name_id.json')
    parser.add_argument('--relation_aliases_path', type=str, default='/home/vvjain3/rag-llm-verify/wikidata5m_relation.txt')

    parser.add_argument('--num_queries', type=int, default=1000)

    parser.add_argument('--gpu_map', type=dict, default={0: "0GiB", 1: "25GiB", 2: "0GiB", 3: "32GiB", "cpu":"120GiB"})
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    return parser.parse_args()

def simple_checker(model_answer, correct_answer, correct_answer_aliases, correct_id):
    model_answer = unidecode(model_answer).lower()
    correct_answer = unidecode(correct_answer).lower()
    if correct_answer in model_answer:
        return 1

    if correct_id not in correct_answer_aliases:
        return 0
    for answer_alias in correct_answer_aliases[correct_id]:
        if answer_alias in model_answer:
            return 1
    return 0

def check_answer(question, correct_answer, model_answer, entity_aliases, correct_id):
    global qa_model, checker_host, checker_port
    if simple_checker(model_answer, correct_answer, entity_aliases, correct_id) == 1:
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

def experiment_pipeline(graph_algos, graph_text, entity_aliases, entity_name2id, relation_name2id, source, relation_aliases, id2name, k=5, num_queries=5):
    global qa_model, tokenizer, checker_llm
    results = []
    correct = 0
    total = 0
    all_queries = []
    all_keys = list(graph_text.keys())
    sys_prompts = ['You are a helpful honest assistant question answerer that answers queries succintly']
    with torch.no_grad():
        for num_iter in range(num_queries):
            prompts = []
            queries_data = []
            query_results = graph_algos.generate_random_query(k, return_path=True, source=source) # allow sampling with replacement
            query_inf, _, correct_id, ids_path = query_results
            query, entity_alias, k_num = query_inf
            # query, entity_alias = form_alias_question(query, path, entity_aliases, relation_aliases, entity_name2id, relation_name2id, graph_algos)
            path = [id2name[ids_path[i]] for i in range(len(ids_path))]
            if 'country of' in query:
                query = query.replace('country of', 'country location of') # to avoid ambiguity
            all_queries.append(query)
            add_keys = random.sample(all_keys, 1)
            true_ids_path = ids_path.copy()
            for key in add_keys:
                if key not in ids_path:
                    ids_path.append(key)
            random.shuffle(ids_path)
            context = ""
            for i, key in enumerate(ids_path):
                context += graph_text[key] + "\n"
            context += f"{id2name[true_ids_path[0]]} is also known as {entity_alias}."
            queries_data.append({'query':query, 'correct_answer':id2name[correct_id], 'path':path, 'context':context, 'correct_id':correct_id})
            prom = []
            prom.append({"role": "system", "content": sys_prompts[0]})
            prom.extend([{"role": "user", "content": f"""Given the context: {context} Answer the query: {query}. Start with the answer in one word or phrase, then explain."""}])
            input_ids = tokenizer.apply_chat_template(prom, return_tensors="pt")
            input_ids = input_ids.to('cuda:1')
            generated_ids= qa_model.generate(input_ids, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.9, top_k=50, top_p=0.9, num_return_sequences=1)
            generated_ids = generated_ids[:, input_ids.shape[-1]:]
            decoded = tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            model_answers = [decoded[0]]
            assert len(queries_data) == len(model_answers)
            for i, model_ans in enumerate(model_answers):
                model_ans = model_ans.strip()
                model_answers[i] = model_ans
            assert len(queries_data) == len(model_answers)
            for i in range(len(queries_data)):
                query = queries_data[i]['query']
                correct_answer = queries_data[i]['correct_answer']
                path = queries_data[i]['path']
                context = queries_data[i]['context']
                correct_id = queries_data[i]['correct_id']
                eval_ans = check_answer(question=query, correct_answer=correct_answer, model_answer=model_ans, entity_aliases=entity_aliases, correct_id=correct_id)
                results.append({ 'question':query, 'correct_answer':correct_answer, 'model_answer':model_ans, 
                                'path':path, 'context':context, 'result':(eval_ans, None)})
                print(correct_answer, model_ans, eval_ans, query, len(prompts[i]))
                correct += results[-1]['result'][0]
                total += 1
            print(f'Completed {num_iter} queries, {correct} correct out of {total} total')
            interval_conf = proportion_confint(correct, total, method='beta', alpha=0.05)
            low = round(interval_conf[0], 2)
            up = round(interval_conf[1], 2)
            if round(up - low, 2) < 0.1:
                break
    print(f'Completed {num_queries} queries, {correct} correct out of {total} total')
    return results, correct, total

def main():
    global qa_model, checker_host, checker_port
    args = get_args()
    qa_model = AutoModelForCausalLM.from_pretrained(args.qa_llm_path, device_map='auto', max_memory=args.gpu_map, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.qa_llm_path)
    qa_graph = json.load(open(args.qa_graph_path))
    context_graph = json.load(open(args.context_graph_path))
    id2name = json.load(open(args.id2name_path))
    entity_name2id = None #{v:k for k,v in id2name.items() if 'Q' in k}
    relation_name2id = None #{v:k for k,v in id2name.items() if 'P' in k}
    checker_host, checker_port = args.host, args.port
    entity_aliases = load_aliases(args.entity_aliases_path)
    relation_aliases = load_aliases(args.relation_aliases_path)

    sys_prompts = ['You are a helpful honest assistant question answerer that answers queries succintly' for _ in range(BATCH_NUM)]
    count = 0
    all_times = []
    qa_graph_algos = GraphAlgos(qa_graph, entity_aliases, relation_aliases)
    print('Running New')
    # best_vertices = qa_graph_algos.get_best_vertices(num=1000)
    best_vertices = best_vertices = ['Q312408', 'Q329988', 'Q5231203', 'Q900947', 'Q76508', 'Q546591', 'Q1793865', 'Q2062573', 'Q2090699', 'Q16264506', 'Q1928626', 'Q1251814', 'Q3740786', 'Q200482', 'Q375855', 'Q279164', 'Q2546120', 'Q229808', 'Q2502106', 'Q505788', 'Q212965', 'Q7958900', 'Q5981732', 'Q1124384', 'Q36033', 'Q679516', 'Q10546329', 'Q974437', 'Q20090095', 'Q36740', 'Q2805655', 'Q245392', 'Q1596236', 'Q946151', 'Q6110803', 'Q219776', 'Q211196', 'Q271830', 'Q1911276', 'Q1217787', 'Q115448', 'Q4351860', 'Q22', 'Q5669183', 'Q425992', 'Q3814812', 'Q1350705', 'Q1359838', 'Q451716', 'Q7901264', 'Q11458011', 'Q1995861']
    # random.shuffle(best_vertices)
    already_done = []
    for file in os.listdir(args.results_dir):
        idx = file.index('Q')
        vertex_id = file[idx:-4]
        already_done.append(vertex_id)

    for i, vertex_id in enumerate(best_vertices):
        vertex = id2name[vertex_id]
        start_time = time.time()
        subgraph = qa_graph_algos.create_subgraph_within_radius(vertex_id, 4)
        subgraph_algos = GraphAlgos(subgraph, entity_aliases, relation_aliases)
        if len(subgraph) < 30:
            print(len(subgraph))
            continue
        print(vertex, vertex_id)
        expriment_results = experiment_pipeline(graph_algos=subgraph_algos, k=4, num_queries=args.num_queries, 
                                                graph_text=context_graph, entity_aliases=entity_aliases, 
                                                entity_name2id=entity_name2id, relation_name2id=relation_name2id, source=vertex_id, relation_aliases=relation_aliases, id2name=id2name)
        end_time = time.time()
        print(f'Time taken for {vertex} = {end_time - start_time}')
        all_times.append(end_time - start_time)
        with open(os.path.join(args.results_dir,str(vertex_id)+'.pkl'), 'wb') as f:
            pickle.dump(expriment_results, f)
        count += 1
    print(f'Average time = {np.mean(all_times)}')
if __name__ == '__main__':
    main()