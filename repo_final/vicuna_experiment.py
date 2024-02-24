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
from fastchat.model import load_model, get_conversation_template, add_model_args
from accelerate import infer_auto_device_map

BATCH_NUM = 2
qa_model, tokenizer, checker_llm = None, None, None

def get_args():
    parser = argparse.ArgumentParser('Run Global experiments')
    parser.add_argument('--qa_llm', type=str, default='lmsys/vicuna-7b-v1.5')
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

    parser.add_argument('--gpu_map', type=dict, default={0: "0GiB", 1: "25GiB", 2: "0GiB", 3: "0GiB", "cpu":"120GiB"})
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

def load_model(model_name="lmsys/vicuna-13b-v1.5", only_tokenizer=False, gpu_map={0: "0GiB", 1: "25GiB", 2: "0GiB", 3: "32GiB", "cpu":"120GiB"}):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if not only_tokenizer:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16, max_memory=gpu_map)
        return tokenizer, model
    else:
        return tokenizer, None

def query_vicuna_model(prompts: list[str], model, tokenizer, do_sample=True, top_k=10, 
                num_return_sequences=1, max_length=40, temperature=1.0):
    
    # preprocess prompts:
    def update_ids(prompt):
        conv_template = get_conversation_template('vicuna')
        conv_template.append_message(conv_template.roles[0], f"{prompt}")
        conv_template.append_message(conv_template.roles[1], '')
        prompt = conv_template.get_prompt()
        return prompt

    input_ids = [update_ids(p) for p in prompts]
    input_ids = tokenizer(input_ids, return_tensors="pt", padding=True).to('cuda:1')
    input_ids = input_ids.input_ids
    generated_ids = model.generate(
        input_ids,
        do_sample=do_sample,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_length,
        temperature=temperature
    )
    generated_ids = generated_ids[:, input_ids.shape[-1]:]
    decoded = tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

    del input_ids, generated_ids
    torch.cuda.empty_cache()
    gc.collect()
    return decoded

def experiment_pipeline(graph_algos, k=5, num_queries=5, graph_text=None, model_device='cuda:0', entity_aliases=None, name2id=None, source=None, relation_aliases=None):
    global qa_model, tokenizer, checker_llm
    results = []
    correct = 0
    total = 0
    all_queries = []
    all_keys = list(graph_text.keys())
    with torch.no_grad():
        for num_iter in range(num_queries//BATCH_NUM):
            torch.cuda.empty_cache()
            gc.collect()
            prompts = []
            queries_data = []
            for j in range(BATCH_NUM):
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
                prompts.append(prompt)
                queries_data.append({'query':query, 'correct_answer':correct_answer, 'path':true_path, 'context':context})
            model_answers = query_vicuna_model(prompts, qa_model, tokenizer)
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
            if num_iter % 50 ==1:
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
    global qa_model, tokenizer, checker_llm
    args = get_args()
    qa_graph = json.load(open(args.qa_graph_path))
    context_graph = json.load(open(args.context_graph_path))
    id2name = json.load(open(args.id2name_path))
    name2id = {v:k for k,v in id2name.items()}
    tokenizer, qa_model = load_model(args.qa_llm, only_tokenizer=False, gpu_map=args.gpu_map)

    if args.checker_llm == 'Gemini':
        checker_llm = GeminiChecker()
    else:
        checker_llm = MistralChecker(args.checker_llm, args.checker_llm_device)
    
    entity_aliases = load_aliases(args.entity_aliases_path)
    relation_aliases = load_aliases(args.relation_aliases_path)

    count = 0
    all_times = []
    qa_graph_algos = GraphAlgos(qa_graph)
    best_vertices = qa_graph_algos.get_best_vertices(num=1000)
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
        break
    print(f'Average time = {np.mean(all_times)}')
if __name__ == '__main__':
    main()