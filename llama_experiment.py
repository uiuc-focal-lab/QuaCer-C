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
import gc
import argparse
import pickle
import copy
import os
from fastchat.model import load_model, get_conversation_template, add_model_args

BATCH_NUM = 2
qa_model = None
checker_host, checker_port = None, None
GPU_MAP = {0: "15GiB", 1: "20GiB", 2: "12GiB", 3: "20GiB", "cpu":"120GiB"}
INPUT_DEVICE = 'cuda:0'

def get_args():
    parser = argparse.ArgumentParser('Run Global experiments')
    parser.add_argument('--qa_llm', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--qa_graph_path', type=str, default='wikidata_graphs4/wikidata_util.json')
    parser.add_argument('--context_graph_edge_path', type=str, default='wikidata_graphs4/wikidata_text_edge.json')
    parser.add_argument('--results_dir', type=str, default='final_results/mistral_distractorfull/')
    parser.add_argument('--entity_aliases_path', type=str, default='wikidata5m_entity.txt')
    parser.add_argument('--id2name_path', type=str, default='wikidata_graphs4/wikidata_name_id.json')
    parser.add_argument('--relation_aliases_path', type=str, default='wikidata5m_relation.txt')
    parser.add_argument('--distractor_query', action='store_true', default=False, help=' best distractor based query?')

    parser.add_argument('--num_queries', type=int, default=1000)
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    return parser.parse_args()

def simple_checker(model_answer, correct_answer, correct_answer_aliases, correct_id):
    """
    Performs a simple check to see if the model answer is correct.

    Checks if the correct answer or any of its aliases are present in the model answer (case-insensitive).

    :param model_answer: The answer generated by the model.
    :param correct_answer: The ground truth correct answer.
    :param correct_answer_aliases: A dictionary mapping correct answer IDs to their aliases.
    :param correct_id: The ID of the correct answer.
    :return: 1 if the model answer is considered correct, 0 otherwise.
    """
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
    """
    Checks the correctness of the model answer using a combination of simple checks and an external checker server.

    First attempts the simple_checker. If that fails, sends a request to the checker server and returns its response.

    :param question: The original question.
    :param correct_answer: The ground truth correct answer.
    :param model_answer: The answer generated by the model.
    :param entity_aliases: A dictionary mapping entity IDs to their aliases.
    :param correct_id: The ID of the correct answer.
    :return: 1 if the model answer is considered correct, 0 otherwise.
    :raises RuntimeError: If the checker server does not return a result. 
    """
    global qa_model, checker_host, checker_port
    if simple_checker(model_answer, correct_answer, entity_aliases, correct_id) == 1:
        return 1
    result_dict = None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((checker_host, checker_port))
        # Prepare and send a request
        request_data = {
            'question': question,
            'correct_answer': correct_answer,
            'model_answer': model_answer
        }
        s.sendall(pickle.dumps(request_data))  # Use pickle to serialize
        response = s.recv(4096)
        result_dict = pickle.loads(response)  # Use pickle to deserialize
    if result_dict is not None:
        return result_dict['result']
    raise RuntimeError('Checker server did not return a result')
    return 0

def load_model(model_name="lmsys/vicuna-13b-v1.5", only_tokenizer=False, gpu_map={0: "26GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB", "cpu":"120GiB"}):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if not only_tokenizer:
    #     model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, load_in_4bit=True, bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, torch_dtype=torch.float16)
        # model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', max_memory=gpu_map, torch_dtype=torch.bfloat16, load_in_8bit=True)
        if '<pad>' not in tokenizer.get_vocab():
            # Add the pad token
            tokenizer.add_special_tokens({"pad_token":"<pad>"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"
        tokenizer.padding_side = 'right'
        return tokenizer, model
    else:
        return tokenizer, None

def query_llama_model(prompts, model, tokenizer, do_sample=True, top_k=10, 
                num_return_sequences=1, max_length=240, temperature=1.0):
    global INPUT_DEVICE
    # preprocess prompts:
    chats = []
    for prompt in prompts:
        message_template = {"role":"user", "content":f"{prompt}"}
        chats.append([copy.deepcopy(message_template)])
    
    input_ids = tokenizer.apply_chat_template(prompts, return_tensors="pt", padding=True).to(INPUT_DEVICE)
    generated_ids= model.generate(input_ids, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=1.0)
    responses = tokenizer.batch_decode(generated_ids[:, input_ids.shape[-1]:].detach().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    del input_ids, generated_ids

    return responses

def get_query_data(graph_algos, source, id2name, graph_text_edge, distractor_query=False, k=5):
    while True:
        distractor=None
        node_distracted=None
        query_results = graph_algos.generate_random_query(k, return_path=True, source=source) # allow sampling with replacement
        if distractor_query:
            distractor_tuple = graph_algos.get_best_distractor(query_results[1], query_results[3])
            if distractor_tuple is None:
                continue
            distractor, node_distracted = distractor_tuple
        query_inf, _, correct_ids, ids_path = query_results
        query, entity_alias, k_num = query_inf
        path = [id2name[ids_path[i]] for i in range(len(ids_path))]
        if 'country of' in query:
            query = query.replace('country of', 'country location of') # to avoid ambiguity
        true_ids_path = ids_path.copy()
        if not distractor_query:
            random_distractor_parent = random.choice(list(graph_text_edge.keys()))
            try:
                random_distractor = random.choice(list(graph_text_edge[random_distractor_parent].keys()))
            except:
                random_distractor = None
            if random_distractor not in true_ids_path and random_distractor is not None:
                distractor = random_distractor
                node_distracted = random_distractor_parent
        context_list = form_context_list(true_ids_path, graph_text_edge)
        if distractor is not None:
            ids_path.append(distractor)
            context_list.append(graph_text_edge[node_distracted][distractor])
        
        random.shuffle(context_list)
        context = '\n'.join(context_list)
        context += f" {id2name[true_ids_path[0]]} is also known as {entity_alias}."
        return {'query':query, 'correct_answers':[id2name[correct_id] for correct_id in correct_ids], 'path_id':true_ids_path, 'path_en':path, 'context':context, 'correct_ids':correct_ids, 'distractor':distractor}
    
def experiment_pipeline(graph_algos, graph_text_edge, entity_aliases, source, relation_aliases, id2name, k=5, distractor_query=False, num_queries=5):
    global qa_model, tokenizer, checker_llm
    results = []
    correct = 0
    total = 0
    all_queries = []
    all_keys = list(graph_text_edge.keys())
    sys_prompts = ['You are a helpful honest assistant question answerer that answers queries succintly' for _ in range(BATCH_NUM)]
    with torch.no_grad():
        for num_iter in range(num_queries//BATCH_NUM):
            prompts = []
            queries_data = []
            for j in range(BATCH_NUM):
                query_data = get_query_data(graph_algos, source, id2name, graph_text_edge, distractor_query=distractor_query, k=k)
                
                prompt = LLM_PROMPT_TEMPLATE.format(context=query_data['context'], query=query_data['query'], few_shot_examples=FEW_SHOT_EXAMPLES)
                prompts.append(prompt)
                queries_data.append(query_data)
            model_answers= query_llama_model(prompts, qa_model, tokenizer, temperature=1.0)
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
                eval_ans = 0
                for num_correct, correct_answer in enumerate(correct_answers):
                    eval_ans = check_answer(question=query, correct_answer=correct_answer, model_answer=model_ans, entity_aliases=entity_aliases, correct_id=correct_ids[num_correct])
                    assert id2name[correct_ids[num_correct]] == correct_answer
                    if eval_ans == 1:
                        break
                # print("Time taken for checking answer: ", end_time_temp2 - end_time_temp1)
                results.append({ 'question':query, 'correct_answers':correct_answers, 'model_answer':model_ans, 
                                'path_en':path, 'path_id':path_id, 'context':context, 'result':(eval_ans, None), 
                                'distractor':distractor, 'correct_ids':correct_ids})
                correct += results[-1]['result'][0]
                total += 1
            print(f'Completed {num_iter} queries, {correct} correct out of {total} total')
            interval_conf = proportion_confint(correct, total, method='beta', alpha=0.05)
            low = float(interval_conf[0])
            up = float(interval_conf[1])
            if round(up - low, 6) <= 0.1:
                print(f"Reached 0.1 interval: {low} {up} {up - low} {correct} {total} {num_iter} {num_queries}")
                break
            del model_answers
        print(f'Completed {num_queries} queries, {correct} correct out of {total} total')
    return results, correct, total

def main():
    global qa_model, checker_host, checker_port, tokenizer
    args = get_args()
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    tokenizer, qa_model = load_model(args.qa_llm, only_tokenizer=False, gpu_map=GPU_MAP)
    qa_graph = json.load(open(args.qa_graph_path))
    context_graph_edge = json.load(open(args.context_graph_edge_path))
    id2name = json.load(open(args.id2name_path))
    checker_host, checker_port = args.host, args.port
    entity_aliases = load_aliases(args.entity_aliases_path)
    relation_aliases = load_aliases(args.relation_aliases_path)

    print(f"Best Distractor Task: {args.distractor_query}")
    count = 0
    all_times = []
    qa_graph_algos = GraphAlgos(qa_graph, entity_aliases, relation_aliases)
    # best_vertices = qa_graph_algos.get_best_vertices(num=1000)
    best_vertices = ['Q200482', 'Q2805655', 'Q36740', 'Q1911276', 'Q453934', 'Q3740786', 'Q36033', 'Q1596236', 'Q1124384', 'Q2062573', 'Q7958900', 'Q931739', 'Q2090699', 'Q505788', 'Q5981732', 'Q1217787', 'Q115448', 'Q5231203', 'Q2502106', 'Q1793865', 'Q329988', 'Q546591', 'Q229808', 'Q974437', 'Q219776', 'Q271830', 'Q279164', 'Q76508', 'Q20090095', 'Q245392', 'Q2546120', 'Q312408', 'Q6110803', 'Q10546329', 'Q211196', 'Q18407657', 'Q18602670', 'Q21979809', 'Q23010088', 'Q1338555', 'Q5516100', 'Q6499669', 'Q1765358', 'Q105624', 'Q166262', 'Q33', 'Q31', 'Q36', 'Q16', 'Q96']
    # random.shuffle(best_vertices)
    already_done = []
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
    
    for file in os.listdir(args.results_dir):
        idx = file.index('Q')
        vertex_id = file[idx:-4]
        already_done.append(vertex_id)

    
    for i, vertex_id in enumerate(best_vertices):
        vertex = id2name[vertex_id]
        if vertex_id in already_done:
            print("Already done", vertex_id, vertex)
            continue
        start_time = time.time()
        subgraph = qa_graph_algos.create_subgraph_within_radius(vertex_id, 4)
        subgraph_algos = GraphAlgos(subgraph, entity_aliases, relation_aliases)
        if len(subgraph) < 900:
            print(len(subgraph), "Skipping", vertex_id, vertex)
            continue
        print(vertex, vertex_id, len(subgraph))
        expriment_results = experiment_pipeline(graph_algos=subgraph_algos, k=4, num_queries=args.num_queries, 
                                                graph_text_edge=context_graph_edge, entity_aliases=entity_aliases, 
                                                source=vertex_id, relation_aliases=relation_aliases, id2name=id2name, distractor_query=args.distractor_query)
        end_time = time.time()
        print(f'Time taken for {vertex} = {end_time - start_time}')
        all_times.append(end_time - start_time)
        with open(os.path.join(args.results_dir, str(vertex_id)+'.pkl'), 'wb') as f:
            pickle.dump(expriment_results, f)
        count += 1
    print(f'Average time = {np.mean(all_times)}')
if __name__ == '__main__':
    main()