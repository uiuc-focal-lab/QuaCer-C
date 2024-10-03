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
import socket
import gc
import argparse
import pickle
import copy

import argparse

def get_base_args():
    parser = argparse.ArgumentParser('Run Global experiments')
    parser.add_argument('--qa_graph_path', type=str, default='wikidata_graphs/wikidata_util.json', help='Path to the QA graph, the util.json file')
    parser.add_argument('--context_graph_edge_path', type=str, default='wikidata_graphs/wikidata_text_edge.json', help='Path to the context graph edge file, the text_edge.json file')
    parser.add_argument('--results_dir', type=str, default='final_results/mistral_distractorfull/', help='Directory to save the results')
    parser.add_argument('--entity_aliases_path', type=str, default='wikidata5m_entity.txt', help='Path to the entity aliases file, the entity.txt file')
    parser.add_argument('--id2name_path', type=str, default='wikidata_graphs/wikidata_name_id.json', help='Path to the id2name file, the name_id.json file')
    parser.add_argument('--sentencized_path', type=str, default='wikidata_graphs/wikidata_sentencized.json', help='Path to the sentencized file, the sentencized.json file')
    parser.add_argument('--relation_aliases_path', type=str, default='wikidata5m_relation.txt', help='Path to the relation aliases file, the relation.txt file')
    parser.add_argument('--distractor_query', action='store_true', default=False, help=' best distractor based query?')
    parser.add_argument('--shuffle_context', action='store_true', default=False, help='Shuffle context in the context of query?')
    parser.add_argument('--k', type=int, default=4, help='Max number of hops in the graph')
    parser.add_argument('--num_queries', type=int, default=250, help='Number of queries to run')
    parser.add_argument('--num_certificates', type=int, default=50, help='Number of certificates to generate')
    return parser

def experiment_pipeline(graph_algos, graph_text_edge, graph_text_sentencized, entity_aliases, source, relation_aliases, id2name, query_model, qa_model, tokenizer, model_context_length, k=5, distractor_query=False, num_queries=5, shuffle_context=True, BATCH_NUM=1, INPUT_DEVICE='cuda:0'):
    results = []
    correct = 0
    total = 0
    with torch.no_grad():
        for num_iter in range(num_queries//BATCH_NUM):
            prompts = []
            queries_data = []
            for j in range(BATCH_NUM):
                query_data = None
                while query_data is None:
                    query_data = get_query_data(graph_algos, source, id2name, graph_text_edge, graph_text_sentencized, tokenizer, 
                                                distractor_query=distractor_query, k=k, shuffle_context=shuffle_context, max_context_length=model_context_length)
                options_str = '\n'.join([f'{i+1}. {id2name[option]}' for i, option in enumerate(query_data['answer_options'])])
                prompt = LLM_PROMPT_TEMPLATE.format(context=query_data['context'], query=query_data['query'], options=options_str, few_shot_examples=FEW_SHOT_EXAMPLES)
                prompts.append(prompt)
                queries_data.append(query_data)
                if distractor_query:
                    assert len(query_data['path_id']) >= 2
            model_answers= query_model(prompts, qa_model, tokenizer, temperature=0.000001, INPUT_DEVICE=INPUT_DEVICE, do_sample=False) #do_sample=False, so no sampling or temp=0
            for i, model_ans in enumerate(model_answers):
                model_ans = model_ans.strip()
                model_answers[i] = model_ans
            assert len(queries_data) == len(model_answers)
            for i in range(len(queries_data)):
                query = queries_data[i]['query']
                correct_answers = queries_data[i]['correct_answers']
                path = queries_data[i]['path_en']
                path_id = queries_data[i]['path_id']
                context = queries_data[i]['context']
                correct_ids = queries_data[i]['correct_ids']
                distractor = queries_data[i]['distractor']
                model_ans = model_answers[i]
                if len(model_ans) == 0:
                    continue
                eval_ans = 0
                for num_correct, correct_answer in enumerate(correct_answers):
                    eval_ans = dumb_checker(model_ans, queries_data[i]['correct_ans_num'])
                    assert id2name[correct_ids[num_correct]] == correct_answer
                    if eval_ans == 1:
                        break
                results.append({ 'question':query, 'correct_answers':correct_answers, 'model_answer':model_ans, 
                                'path_en':path, 'path_id':path_id, 'context':context, 'result':(eval_ans, None), 
                                'distractor':distractor, 'correct_ids':correct_ids, 'options':queries_data[i]['answer_options'], 'correct_ans_num':queries_data[i]['correct_ans_num']})
                correct += results[-1]['result'][0]
                total += 1
            print(f'Completed {num_iter+1} queries, {correct} correct out of {total} total')
            del model_answers
        print(f'Completed {num_queries} queries, {correct} correct out of {total} total')
    return results, correct, total

def load_experiment_setup(args, load_model, GPU_MAP):
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    tokenizer, qa_model = load_model(args.qa_llm, only_tokenizer=False, gpu_map=GPU_MAP, quant_type=args.quant_type)
    qa_graph = json.load(open(args.qa_graph_path))
    context_graph_edge = json.load(open(args.context_graph_edge_path))
    graph_text_sentencized = json.load(open(args.sentencized_path))
    id2name = json.load(open(args.id2name_path))
    entity_aliases = load_aliases(args.entity_aliases_path)
    relation_aliases = load_aliases(args.relation_aliases_path)
    print(f"Best Distractor Task: {args.distractor_query}")
    qa_graph_algos = GraphAlgos(qa_graph, entity_aliases, relation_aliases)
    # best_vertices = qa_graph_algos.get_best_vertices(num=1000)
    best_vertices =  ['Q38', 'Q1055', 'Q838292', 'Q34433', 'Q254', 'Q31', 'Q270', 'Q200482', 'Q36740', 'Q1911276', 'Q3740786', 'Q1124384', 'Q931739', 'Q2090699', 'Q505788', 'Q1217787', 'Q115448', 'Q2502106', 'Q1793865', 'Q229808', 'Q974437', 'Q219776', 'Q271830', 'Q279164', 'Q76508', 'Q245392', 'Q2546120', 'Q312408', 'Q6110803', 'Q211196', 'Q18407657', 'Q18602670', 'Q21979809', 'Q23010088', 'Q1338555', 'Q5516100', 'Q1765358', 'Q105624', 'Q166262', 'Q33', 'Q36', 'Q16', 'Q96', 'Q36687', 'Q282995', 'Q858401', 'Q850087', 'Q864534', 'Q291244', 'Q159', 'Q668', 'Q211', 'Q183', 'Q1603', 'Q408', 'Q218'][:50]
    # random.shuffle(best_vertices)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
    
    return qa_graph_algos, context_graph_edge, graph_text_sentencized, entity_aliases, relation_aliases, id2name, qa_model, tokenizer, best_vertices

def run_experiment(args, load_model, query_model_func, GPU_MAP, model_context_length, BATCH_NUM=1, INPUT_DEVICE='cuda:0'):
    qa_graph_algos, context_graph_edge, graph_text_sentencized, entity_aliases, relation_aliases, id2name, qa_model, tokenizer, best_vertices = load_experiment_setup(args, load_model, GPU_MAP)
    already_done = []
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
    
    for file in os.listdir(args.results_dir):
        idx = file.index('Q')
        vertex_id = file[idx:-4]
        already_done.append(vertex_id)
    all_times = []
    num_certificates_generated = 0
    for i, vertex_id in enumerate(best_vertices):
        if num_certificates_generated >= args.num_certificates:
            break
        vertex = id2name[vertex_id]
        if vertex_id in already_done:
            print("Already done", vertex_id, vertex)
            continue
        start_time = time.time()
        subgraph = qa_graph_algos.create_subgraph_within_radius(vertex_id, 4)
        subgraph_algos = GraphAlgos(subgraph, entity_aliases, relation_aliases)
        if len(subgraph) < 900:
            print(len(subgraph), "Skipping", vertex_id, vertex) # Skip small subgraphs
            continue
        print(vertex, vertex_id, len(subgraph))
        num_certificates_generated += 1
        
        experiment_results = experiment_pipeline(graph_algos=subgraph_algos, graph_text_edge=context_graph_edge, graph_text_sentencized=graph_text_sentencized, 
                                                 entity_aliases=entity_aliases, source=vertex_id, relation_aliases=relation_aliases, id2name=id2name, 
                                                 query_model=query_model_func, qa_model=qa_model, tokenizer=tokenizer, k=args.k, 
                                                 distractor_query=args.distractor_query, num_queries=args.num_queries, 
                                                 shuffle_context=args.shuffle_context, BATCH_NUM=BATCH_NUM, INPUT_DEVICE=INPUT_DEVICE, model_context_length=model_context_length)
        end_time = time.time()
        print(f'Time taken for {vertex} = {end_time - start_time}')
        all_times.append(end_time - start_time)
        with open(os.path.join(args.results_dir, str(vertex_id)+'.pkl'), 'wb') as f:
            pickle.dump(experiment_results, f)
    return all_times, num_certificates_generated