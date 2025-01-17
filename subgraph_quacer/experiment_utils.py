import os
import json
import numpy as np
from utils import *
import argparse
from statsmodels.stats.proportion import proportion_confint
import torch
import gc
from unidecode import unidecode
import pickle
import time
import copy
import string
from subgraph_utils import CustomQueryGenerator, CustomQueryResult

def get_base_args():
    parser = argparse.ArgumentParser('Run Relation-based Certificate experiments')
    parser.add_argument('--qa_graph_path', type=str, default='qa_graph.json', 
                       help='Path to the QA graph JSON file')
    parser.add_argument('--context_graph_edge_path', type=str, default='context_graph_edge.json',
                       help='Path to the context graph edge file')
    parser.add_argument('--results_dir', type=str, default='results/',
                       help='Directory to save results')
    parser.add_argument('--entity_aliases_path', type=str, default='entity_aliases.txt',
                       help='Path to entity aliases file')
    parser.add_argument('--relation_aliases_path', type=str, default='relation_aliases.txt',
                       help='Path to relation aliases file')
    parser.add_argument('--id2name_path', type=str, default='id2name.json',
                       help='Path to id2name mapping file')
    parser.add_argument('--shuffle_context', action='store_true', default=False,
                       help='Shuffle context sentences')
    parser.add_argument('--num_queries', type=int, default=250,
                       help='Number of queries per certificate')
    parser.add_argument('--distractor_query', action='store_true', default=False, help=' best distractor based query?')
    return parser

def load_experiment_setup(args, load_model, GPU_MAP):
    """Load and initialize all required components for the experiment"""
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
        
    tokenizer, qa_model = load_model(args.qa_llm, only_tokenizer=False, 
                                   gpu_map=GPU_MAP, quant_type=args.quant_type)
    
    qa_graph = json.load(open(args.qa_graph_path))
    context_graph_edge = json.load(open(args.context_graph_edge_path))
    id2name = json.load(open(args.id2name_path))
    
    entity_aliases = load_aliases(args.entity_aliases_path)
    relation_aliases = load_aliases(args.relation_aliases_path)
    
    # Add names to entity aliases
    for key, value in id2name.items():
        if 'p' in key or 'P' in key or 'R' in key or 'r' in key:
            continue
        entity_aliases[key] = [value]
    
    qa_graph_algos = GraphAlgos(qa_graph, entity_aliases, relation_aliases)
    
    return (qa_graph_algos, context_graph_edge, entity_aliases, 
            relation_aliases, id2name, qa_model, tokenizer)

def experiment_pipeline(graph_algos, graph_text_edge, entity_aliases, 
                       relation_aliases, id2name, query_generator,
                       query_model, qa_model, tokenizer, model_context_length,
                       discovery_func_idx, num_queries=5, shuffle_context=False, 
                       BATCH_NUM=1, INPUT_DEVICE='cuda:0', distractor_query=False):
    """Run the experiment pipeline for a single discovery function"""
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
                    query_result = query_generator.get_single_query(discovery_func_idx, id2name=id2name, distractor_setting=distractor_query)
                    if query_result is not None:
                        query_data = query_generator.generate_query_data(
                            query_result,
                            id2name,
                            graph_text_edge,
                            tokenizer,
                            shuffle_context=shuffle_context,
                            max_context_length=model_context_length,
                            distractor_query=distractor_query
                        )
                
                options_str = '\n'.join([f'{i+1}. {id2name[option]}' 
                                       for i, option in enumerate(query_data['answer_options'])])
                prompt = LLM_PROMPT_TEMPLATE.format(
                    context=query_data['context'],
                    query=query_data['query'],
                    options=options_str,
                    few_shot_examples=FEW_SHOT_EXAMPLES
                )
                
                prompts.append(prompt)
                queries_data.append(query_data)
            
            model_answers = query_model(
                prompts, qa_model, tokenizer,
                temperature=0.000001,
                INPUT_DEVICE=INPUT_DEVICE,
                do_sample=False
            )
            
            for i, model_ans in enumerate(model_answers):
                query_data = queries_data[i]
                model_ans = model_ans.strip()
                
                if len(model_ans) == 0:
                    continue
                    
                eval_ans = dumb_checker(model_ans, query_data['correct_ans_num'])
                
                results.append({
                    'question': query_data['query'],
                    'correct_answers': query_data['correct_answers'],
                    'model_answer': model_ans,
                    'path_en': query_data['path_en'],
                    'path_id': query_data['path_id'],
                    'context': query_data['context'],
                    'result': (eval_ans, None),
                    'correct_ids': query_data['correct_ids'],
                    'options': query_data['answer_options'],
                    'correct_ans_num': query_data['correct_ans_num'],
                    'other_correct_answers': query_data.get('other_correct_answers', [])
                })
                
                correct += eval_ans
                total += 1
                
            print(f'Completed {num_iter+1} queries, {correct} correct out of {total} total')
            del model_answers
            
    print(f'Completed all {num_queries} queries, {correct} correct out of {total} total')
    return results, correct, total

def run_experiment(args, load_model, query_model_func, discovery_funcs, discovery_names, GPU_MAP, 
                  model_context_length, BATCH_NUM=1, INPUT_DEVICE='cuda:0', discovery_idx=None):
    """Run the experiment for specified discovery functions
    
    Args:
        discovery_funcs: List of all discovery functions
        discovery_names: List of names corresponding to discovery functions
        discovery_idx: Optional index or list of indices to run specific functions.
                      If None, runs all functions that don't have certificates yet.
    """
    
    # Load experiment components
    experiment_components = load_experiment_setup(args, load_model, GPU_MAP)
    (qa_graph_algos, context_graph_edge, entity_aliases, 
     relation_aliases, id2name, qa_model, tokenizer) = experiment_components

    query_generator = CustomQueryGenerator(qa_graph_algos, discovery_funcs)

    # Determine which functions to run
    if discovery_idx is None:
        indices = range(len(discovery_funcs))
    elif isinstance(discovery_idx, int):
        indices = [discovery_idx]
    else:
        indices = discovery_idx

    results = {}
    for idx in indices:
        func_name = discovery_names[idx]
        
        func_path = os.path.join(args.results_dir, f'{func_name}.pkl')
        if os.path.exists(func_path):
            print(f"Certificate for {func_name} already exists, skipping...")
            results[func_name] = {"time": None, "completed": False}
            continue

        print(f"\nGenerating certificate for {func_name}")
        start_time = time.time()
        
        experiment_results = experiment_pipeline(
            graph_algos=qa_graph_algos,
            graph_text_edge=context_graph_edge,
            entity_aliases=entity_aliases,
            relation_aliases=relation_aliases,
            id2name=id2name,
            query_generator=query_generator,
            discovery_func_idx=idx,  # Pass the index to pipeline
            query_model=query_model_func,
            qa_model=qa_model,
            tokenizer=tokenizer,
            num_queries=args.num_queries,
            shuffle_context=args.shuffle_context,
            BATCH_NUM=BATCH_NUM,
            INPUT_DEVICE=INPUT_DEVICE,
            model_context_length=model_context_length,
            distractor_query=args.distractor_query
        )
        
        end_time = time.time()
        time_taken = end_time - start_time
        
        
        with open(func_path, 'wb') as f:
            pickle.dump(experiment_results, f)
            
        print(f'Completed certificate for {func_name}')
        print(f'Time taken: {time_taken:.2f} seconds')
        
        results[func_name] = {"time": time_taken, "completed": True}
    
    return results