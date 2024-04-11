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

qa_model = None
checker_host, checker_port = None, None

def get_args():
    parser = argparse.ArgumentParser('Run Global experiments')
    parser.add_argument('--qa_llm', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--qa_llm_device', type=str, default='cuda:3')
    parser.add_argument('--qa_graph_path', type=str, default='wikidata_graphs1/wikidata_util.json')
    parser.add_argument('--context_graph_path', type=str, default='wikidata_graphs1/wikidata_text.json')
    parser.add_argument('--results_dir', type=str, default='mistral_dist_random/')
    parser.add_argument('--entity_aliases_path', type=str, default='wikidata5m_entity.txt')
    parser.add_argument('--id2name_path', type=str, default='wikidata_graphs1/wikidata_name_id.json')
    parser.add_argument('--relation_aliases_path', type=str, default='wikidata5m_relation.txt')

    parser.add_argument('--num_queries', type=int, default=1000)

    parser.add_argument('--gpu_map', type=dict, default={0: "25GiB", 1: "0GiB", 2: "0GiB", 3: "28GiB", "cpu":"120GiB"})
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

def experiment_pipeline(graph_algos, graph_text, entity_aliases, source, relation_aliases, id2name, k=5, num_queries=5, model_device='cuda:0'):
    global qa_model, tokenizer, checker_llm
    results = []
    correct = 0
    total = 0
    all_queries = []
    all_keys = list(graph_text.keys())
    sys_prompts = ['You are a helpful honest assistant question answerer that answers queries succintly']
    num_iter = 0
    with torch.no_grad():
        while num_iter < num_queries:
            torch.cuda.empty_cache()
            gc.collect()
            prompts = []
            queries_data = []
            query_results = graph_algos.generate_random_query(k, return_path=True, source=source) # allow sampling with replacement
            distractor = graph_algos.get_best_distractor(query_results[1], query_results[3])
            if distractor is None:
                continue
            query_inf, _, correct_ids, ids_path = query_results
            query, entity_alias, k_num = query_inf
            path = [id2name[ids_path[i]] for i in range(len(ids_path))]
            if 'country of' in query:
                query = query.replace('country of', 'country location of') # to avoid ambiguity
            all_queries.append(query)
            true_ids_path = ids_path.copy()
            add_keys = random.sample(all_keys, 1)
            for key in add_keys:
                if key not in ids_path:
                    ids_path.append(key)
            # ids_path.append(distractor)
            random.shuffle(ids_path)
            context = ""
            for i, key in enumerate(ids_path):
                context += graph_text[key] + "\n"
            context += f"{id2name[true_ids_path[0]]} is also known as {entity_alias}."
            messages = [
                {"role": "user", "content": f"context: {context}"},
                {"role": "assistant", "content": "I understand the context. What can I help you with?"},
                {"role": "user", "content": f"Using the above context, answer succiently the query: {query}. Start with the answer in one word or phrase, then explain."},
            ]
            queries_data.append({'query':query, 'correct_answers':[id2name[correct_id] for correct_id in correct_ids], 'path':path, 'context':context, 'correct_ids':correct_ids})
            encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
            model_inputs = encodeds.to(model_device)
            generated_ids = qa_model.generate(model_inputs, max_new_tokens=40, do_sample=True, temperature=1.0)
            decoded = tokenizer.batch_decode(generated_ids.detach().cpu())
            model_ans = decoded[0][decoded[0].index('Start with the answer in one word or phrase, then explain.')+len('Start with the answer in one word or phrase, then explain.'):]
            model_answers= [model_ans.strip()]
            assert len(queries_data) == len(model_answers)
            for i in range(len(queries_data)):
                query = queries_data[i]['query']
                correct_answers = queries_data[i]['correct_answers']
                path = queries_data[i]['path']
                context = queries_data[i]['context']
                correct_ids = queries_data[i]['correct_ids']
                eval_ans = 0
                for num_correct, correct_answer in enumerate(correct_answers):
                    eval_ans = check_answer(question=query, correct_answer=correct_answer, model_answer=model_ans, entity_aliases=entity_aliases, correct_id=correct_ids[num_correct])
                    assert id2name[correct_ids[num_correct]] == correct_answer
                    if eval_ans == 1:
                        break
                results.append({ 'question':query, 'correct_answers':correct_answers, 'model_answer':model_ans, 
                                'path':path, 'context':context, 'result':(eval_ans, None)})
                print(correct_answers, model_ans, eval_ans, query)
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
            num_iter += 1
        print(f'Completed {num_queries} queries, {correct} correct out of {total} total')
    return results, correct, total

def main():
    global qa_model, checker_host, checker_port, tokenizer
    args = get_args()
    qa_model = AutoModelForCausalLM.from_pretrained(args.qa_llm, torch_dtype=torch.float16).to(args.qa_llm_device)
    tokenizer = AutoTokenizer.from_pretrained(args.qa_llm)
    qa_graph = json.load(open(args.qa_graph_path))
    context_graph = json.load(open(args.context_graph_path))
    id2name = json.load(open(args.id2name_path))
    checker_host, checker_port = args.host, args.port
    entity_aliases = load_aliases(args.entity_aliases_path)
    relation_aliases = load_aliases(args.relation_aliases_path)

    count = 0
    all_times = []
    qa_graph_algos = GraphAlgos(qa_graph, entity_aliases, relation_aliases)
    # print('Running New')
    # best_vertices = qa_graph_algos.get_best_vertices(num=1000)
    best_vertices = ['Q2805655', 'Q36740', 'Q1911276', 'Q453934', 'Q3740786', 'Q36033', 'Q1596236', 'Q1124384', 'Q200482', 'Q2062573', 'Q7958900', 'Q931739', 'Q2090699', 'Q505788', 'Q5981732', 'Q1217787', 'Q115448', 'Q5231203', 'Q2502106', 'Q1793865', 'Q329988', 'Q546591', 'Q229808', 'Q974437', 'Q219776', 'Q271830', 'Q279164', 'Q76508', 'Q20090095', 'Q245392', 'Q2546120', 'Q312408', 'Q6110803', 'Q10546329', 'Q211196']
    # random.shuffle(best_vertices)
    already_done = ['Q1251814']
    for file in os.listdir(args.results_dir):
        idx = file.index('Q')
        vertex_id = file[idx:-4]
        already_done.append(vertex_id)

    
    for i, vertex_id in enumerate(best_vertices):
        vertex = id2name[vertex_id]
        start_time = time.time()
        subgraph = qa_graph_algos.create_subgraph_within_radius(vertex_id, 4)
        subgraph_algos = GraphAlgos(subgraph, entity_aliases, relation_aliases)
        if len(subgraph) < 900:
            print(len(subgraph))
            continue
        if vertex_id in already_done:
            continue
        print(vertex, vertex_id, len(subgraph))
        expriment_results = experiment_pipeline(graph_algos=subgraph_algos, k=4, num_queries=args.num_queries, 
                                                graph_text=context_graph, entity_aliases=entity_aliases, 
                                                source=vertex_id, relation_aliases=relation_aliases, id2name=id2name, model_device=args.qa_llm_device)
        end_time = time.time()
        print(f'Time taken for {vertex} = {end_time - start_time}')
        all_times.append(end_time - start_time)
        with open(os.path.join(args.results_dir, str(vertex_id)+'.pkl'), 'wb') as f:
            pickle.dump(expriment_results, f)
        count += 1
    print(f'Average time = {np.mean(all_times)}')
if __name__ == '__main__':
    main()