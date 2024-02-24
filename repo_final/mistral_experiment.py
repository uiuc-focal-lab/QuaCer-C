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

qa_model, tokenizer, checker_llm = None, None, None

def get_args():
    parser = argparse.ArgumentParser('Run Global experiments')
    parser.add_argument('--qa_llm', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--checker_llm', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--qa_graph_path', type=str, default='wikidata5m_en_util_unidecoded1.json')
    parser.add_argument('--context_graph_path', type=str, default='wikidata5m_en_text1.json')
    parser.add_argument('--qa_llm_device', type=str, default='cuda:1')
    parser.add_argument('--checker_llm_device', type=str, default='cuda:3')
    parser.add_argument('--results_path', type=str, default='results_exprimentmist.pkl')
    parser.add_argument('--entity_aliases_path', type=str, default='wikidata5m_entity.txt')
    parser.add_argument('--id2name_path', type=str, default='wikidata_name_id_uni1.json')
    parser.add_argument('--relation_aliases_path', type=str, default='wikidata5m_relation.txt')

    parser.add_argument('--num_queries', type=int, default=1000)

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
    global checker_llm, qa_model, tokenizer
    if simple_checker(model_answer, correct_answer, entity_aliases, name2id) == 1:
        return 1
    return checker_llm.raw_checker(question=question, correct_ans=correct_answer, model_ans=model_answer)[0]

def experiment_pipeline(graph_algos, k=5, num_queries=5, graph_text=None, model_device='cuda:0', entity_aliases=None, name2id=None, source=None, relation_aliases=None):
    global qa_model, tokenizer, checker_llm
    results = []
    correct = 0
    total = 0
    all_queries = []
    all_keys = list(graph_text.keys())
    with torch.no_grad():
        for num_iter in range(num_queries):
            torch.cuda.empty_cache()
            gc.collect()
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
            messages = [
                {"role": "user", "content": f"context: {context}"},
                {"role": "assistant", "content": "I understand the context. What can I help you with?"},
                {"role": "user", "content": f"Using the above context, answer succiently the query: {query}. Start with the answer in one word or phrase."},
            ]

            encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

            model_inputs = encodeds.to(model_device)
            print(model_inputs.shape)
            generated_ids = qa_model.generate(model_inputs, max_new_tokens=40, do_sample=True, temperature=1.0)
            decoded = tokenizer.batch_decode(generated_ids.detach().cpu())
            model_ans = decoded[0][decoded[0].index('Start with the answer in one word or phrase.')+len('Start with the answer in one word or phrase.'):]
            eval_ans = check_answer(question=query, correct_answer=correct_answer, model_answer=model_ans, entity_aliases=entity_aliases, name2id=name2id)
            results.append({ 'question':query, 'correct_answer':correct_answer, 'model_answer':model_ans, 
                                'path':true_path, 'context':context, 'result':(eval_ans, None)})
            print(correct_answer, model_ans, eval_ans)
            correct += results[-1]['result'][0]
            total += 1
            del model_inputs, generated_ids, decoded
            torch.cuda.empty_cache()
            gc.collect()
            if num_iter %50 ==1:
                (low, high) = proportion_confint(correct, total, method='beta')
                if high - low < 0.1:
                    break
                print(f'Completed {num_iter} queries, {correct} correct out of {total} total')
        print(f'Completed {num_queries} queries, {correct} correct out of {total} total')
    vertex_id = name2id[source]
    with open(f'/home/vvjain3/rag-llm-verify/wikidata-scripts/mistral7b_answers/exp7b_{vertex_id}.pkl', 'wb') as f:
        pickle.dump(results, f)
    return results, correct, total

def main():
    global qa_model, tokenizer, checker_llm
    args = get_args()
    qa_graph = json.load(open(args.qa_graph_path))
    context_graph = json.load(open(args.context_graph_path))
    id2name = json.load(open(args.id2name_path))
    name2id = {v:k for k,v in id2name.items()}
    qa_model = AutoModelForCausalLM.from_pretrained(args.qa_llm).to(args.qa_llm_device)
    tokenizer = AutoTokenizer.from_pretrained(args.qa_llm)

    if args.checker_llm == 'Gemini':
        checker_llm = GeminiChecker()
    else:
        checker_llm = MistralChecker(args.checker_llm, args.checker_llm_device)
    
    entity_aliases = load_aliases(args.entity_aliases_path)
    relation_aliases = load_aliases(args.relation_aliases_path)

    count = 0
    all_times = []
    qa_graph_algos = GraphAlgos(qa_graph)
    best_vertices = qa_graph_algos.get_best_vertices(num=500)
    random.shuffle(best_vertices)
    for i, vertex in enumerate(best_vertices):
        start_time = time.time()
        subgraph = qa_graph_algos.create_subgraph_within_radius(vertex, 4)
        subgraph_algos = GraphAlgos(subgraph)
        if len(subgraph) < 40:
            continue
        print(vertex, name2id[vertex])
        expriment_results = experiment_pipeline(graph_algos=subgraph_algos, k=4, num_queries=args.num_queries, 
                                                graph_text=context_graph, model_device=args.qa_llm_device, entity_aliases=entity_aliases, 
                                                name2id=name2id, source=vertex, relation_aliases=relation_aliases)
        end_time = time.time()
        print(f'Time taken for {vertex} = {end_time - start_time}')
        all_times.append(end_time - start_time)
        with open(args.results_path[:-4]+str(name2id[vertex])+'.pkl', 'wb') as f:
            pickle.dump(expriment_results, f)
        count += 1
    print(f'Average time = {np.mean(all_times)}')
if __name__ == '__main__':
    main()