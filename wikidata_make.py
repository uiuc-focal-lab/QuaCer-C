import os
import json
import gc
from unidecode import unidecode
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import re
import time
import copy

wikidata_graph = {}
wikidata_name_id = {}
wikidata_en_graph = {}
wikidata_en_graph = {}
wikidata_text = {}
wikidata_en_text = {}
codex_graph = {}

find = ['P31', 'P279', 'P361'] # instance of, subclass of, part of, these relations are very vague so we discard them
with open('/home/vvjain3/rag-llm-verify/wikidata5m_all_triplet.txt', 'r') as file:
    for line in file:
        line = line.strip()
        line = line.split('\t')
        line = [x.strip() for x in line]
        if line[1] in find:
            continue
        if line[0] not in wikidata_graph:
            wikidata_graph[line[0]] = {}
        if line[1] not in wikidata_graph[line[0]]:
            wikidata_graph[line[0]][line[1]] = []
        wikidata_graph[line[0]][line[1]].append(line[2])
        
with open('wikidata5m_text.txt', 'r') as f:
    for line in f:
        line = line.strip()
        line = line.split('\t')
        line = [x.strip() for x in line]
        text = line[1].strip()
        if len(line) > 2:
            for i in range(2, len(line)):
                text += ' ' + line[i].strip()
        if line[0] in wikidata_graph:
            wikidata_text[line[0]] = text


possible_entity_names = {}
with open('wikidata5m_entity.txt', 'r') as f:
    for line in f:
        line = line.strip()
        line = line.split('\t')
        line = [x.strip() for x in line]
        if line[0] not in wikidata_text:
            continue
        possible_entity_names[line[0]] = line[1:]

print("Starting Names")
possible_entity_names = {}
with open('wikidata5m_entity.txt', 'r') as f:
    for line in f:
        line = line.strip()
        line = line.split('\t')
        line = [x.strip() for x in line]
        if line[0] not in wikidata_text:
            continue
        possible_entity_names[line[0]] = line[1:]
        possible = line[1:min(30, len(line))] #select most common label from the first 30 aliases
        common = ''
        all_text = wikidata_text[line[0]]
        all_text = unidecode(all_text)
        all_text = all_text.lower()
        all_text = all_text.replace('–', '-') #replace en dash with hyphen
        min_ind = 1000000
        #choose common according to the order of the aliases appearing in the text
        for name in possible:
            name = unidecode(name)
            name = name.lower()
            pattern = r'\b' + re.escape(name) + r'\b'
            matches = re.search(pattern, all_text)
            if matches:
                ind = all_text.index(name)
                if ind < min_ind:
                    min_ind = ind
                    common = name
                if min_ind <= 15:
                    break
        if common == '':
            continue
        wikidata_name_id[line[0]] = common.strip()
with open('wikidata5m_relation.txt', 'r') as f:
    for line in f:
        line = line.strip()
        line = line.split('\t')
        line = [x.strip() for x in line]
        wikidata_name_id[line[0]] = unidecode(line[1]).lower()
gc.collect()

print("len wikidata name id:", len(wikidata_name_id))

wikidata_en_graph = {}
wikidata_id_name = {}
for key, value in wikidata_name_id.items():
    wikidata_id_name[value] = key
    
wikidata_util = {}
for key, value in wikidata_graph.items():
    wikidata_util[key] = {}
    for rel, objs  in value.items():
        for obj in objs:
            wikidata_util[key][obj] = rel

print(len(wikidata_util), len(wikidata_graph))

wikidata_text_edge = {}


wikidata_text_sentencized = {}
for key, value in wikidata_text.items():
    sentencized = sent_tokenize(value)
    if len(sentencized) == 0:
        print(key, value)
        raise ValueError('No sentence found')
    sentencized = [x.strip() for x in sentencized]
    wikidata_text_sentencized[key] = sentencized

wikidata_text_sentencized_format = {}
for key, value in wikidata_text_sentencized.items():
    wikidata_text_sentencized_format[key] = [unidecode(sent).lower().strip() for sent in value]
len(wikidata_text_sentencized_format), len(wikidata_text_sentencized)

print("Starting Context Trimming")
new_wikidata = {}
wikidata_text_edge = {}
count = 0
total = 0
keys_done_num = 0
keys_considered = 0
start_time = time.time()
for key, value in wikidata_util.items():
    keys_considered += 1
    if key not in wikidata_text or key not in wikidata_name_id:
        continue
    keys_done_num += 1
    wikidata_text_edge[key] = {}
    possible_key_names = possible_entity_names[key]
    possible_key_names = [unidecode(x).lower().strip() for x in possible_key_names]
    new_value = {}
    for obj, rel in value.items():
        total += 1
        if obj not in wikidata_name_id or rel not in wikidata_name_id:
            continue
        possible_names = possible_entity_names[obj]
        possible_names = [unidecode(x).lower().strip() for x in possible_names]
        key_relevant_ids = []
        obj_relevant_ids = []
        key_relevant_ids.append(0)
        # relevant_sentences.append(wikidata_text_sentencized[key][0])
        found = False
        for i, sent in enumerate(wikidata_text_sentencized_format[key]):
            for name in possible_names:
                pattern = r'\b' + re.escape(name) + r'\b'
                matches = re.search(pattern, sent)
                if matches:
                    # print(i, len(wikidata_text_sentencized_format[key]))
                    if i > 0:
                        key_relevant_ids.append(i)
                    new_value[obj] = rel
                    # if i > 0:
                    #     # for add_i in range(max(1, i-2), i):
                    #     #     relevant_sentences.append(wikidata_text_sentencized[key][add_i])
                    #     key_relevant_ids.add(i)
                        # relevant_sentences.append(wikidata_text_sentencized[key][i])
                    # relevant_sentences.append(sent)
                    # for add_i in range(i+1, min(i+3, len(wikidata_text_sentencized_format[key]))):
                    #     relevant_sentences.append(wikidata_text_sentencized[key][add_i])
                    found = True
                    break
        if not found:
            if obj not in wikidata_text:
                continue
            obj_relevant_ids.append(0)
            # relevant_sentences.append(wikidata_text_sentencized[obj][0])
            for i, sent in enumerate(wikidata_text_sentencized_format[obj]):
                for name in possible_key_names:
                    pattern = r'\b' + re.escape(name) + r'\b'
                    matches = re.search(pattern, sent)
                    if matches:
                        # print(i, len(wikidata_text_sentencized_format[obj]))
                        if i > 0:
                            obj_relevant_ids.append(i)
                        new_value[obj] = rel
                        # if i > 0:
                        #     for add_i in range(max(1, i-2), i):
                        #         relevant_sentences.append(wikidata_text_sentencized[obj][add_i])
                        #     relevant_sentences.append(wikidata_text_sentencized[obj][i])
                        # # relevant_sentences.append(sent)
                        # for add_i in range(i+1, min(i+3, len(wikidata_text_sentencized_format[obj]))):
                        #     relevant_sentences.append(wikidata_text_sentencized[obj][add_i])
                        found = True
                        break
        if found:
            relevant_sentences = []
            num_key_sents = len(wikidata_text_sentencized[key])
            k_done = 0 
            k_todo = 5 - len(key_relevant_ids) - len(obj_relevant_ids) # additional sentences to be added as 5 is min in context
            
            #add all key sentences as well as filler sentences in between to get min length context
            for i in range(len(key_relevant_ids)):
                relevant_sentences.append(wikidata_text_sentencized[key][key_relevant_ids[i]])
                if k_done >= k_todo:
                    continue
                if i < len(key_relevant_ids) - 1:
                    if key_relevant_ids[i+1] - key_relevant_ids[i] > 1:
                        #add atmost 2 sentences between key sentences if needed
                        for j in range(max(key_relevant_ids[i]+1, key_relevant_ids[i+1]-2), key_relevant_ids[i+1]):
                            if k_done >= k_todo:
                                break
                            relevant_sentences.append(wikidata_text_sentencized[key][j])
                            k_done += 1
                else:
                    #add atmost 2 sentences after the last key sentence if needed
                    if key_relevant_ids[i] < num_key_sents - 1:
                        for j in range(max(key_relevant_ids[i]+1, num_key_sents-2), num_key_sents):
                            if k_done >= k_todo:
                                break
                            relevant_sentences.append(wikidata_text_sentencized[key][j])
                            k_done += 1
            
            #max context sentences number is 8
            if len(relevant_sentences) > 8:
                relevant_sentences = relevant_sentences[:8]
            #as we add and remove in order the intro sentence about the entity is always present
            assert wikidata_text_sentencized[key][0] in relevant_sentences 
            for idx in obj_relevant_ids:
                if len(relevant_sentences) >= 8:
                    break
                relevant_sentences.append(wikidata_text_sentencized[obj][idx])
            wikidata_text_edge[key][obj] = copy.deepcopy(relevant_sentences) #' '.join(relevant_sentences)
            new_value[obj] = rel
            if (count-8) % 1000000 == 0:
                print(key, obj, len(wikidata_text_edge[key][obj]), wikidata_text_edge[key][obj])
            count += 1
    if len(new_value) > 0:
        new_wikidata[key] = new_value
    if keys_done_num % 100000 == 0:
        end_time = time.time()
        print("Time taken for 100000 keys:", end_time - start_time)
        start_time = time.time()
        print(keys_done_num, keys_considered, count, total)

print("Total keys:", keys_done_num, keys_considered, count, total)

remove = set()
new_graph = {}
for key, value in new_wikidata.items():
    assert key in wikidata_name_id
    assert key in wikidata_text
    new_value = {}
    for k, v in value.items():
        assert k in wikidata_name_id
        if v not in wikidata_name_id:
            remove.add(v)
            continue
        assert k in wikidata_text
        new_value[k] = v
    if len(new_value) > 0:
        new_graph[key] = new_value

wikidata_graph_util = new_graph
for key, value in wikidata_graph_util.items():
    assert key in wikidata_name_id
    assert key in wikidata_text
    for k, v in value.items():
        assert k in wikidata_name_id
        assert v in wikidata_name_id
        assert k in wikidata_text
        assert k in wikidata_text_edge[key]

os.makedirs('wikidata_graphs1', exist_ok=True)
with open('wikidata_graphs1/wikidata_name_id.json', 'w') as f:
    json.dump(wikidata_name_id, f)
with open('wikidata_graphs1/wikidata_text.json', 'w') as f:
    json.dump(wikidata_text, f)
with open('wikidata_graphs1/wikidata_util.json', 'w') as f:
    json.dump(wikidata_graph_util, f)
with open('wikidata_graphs1/wikidata_sentencized.json', 'w') as f:
    json.dump(wikidata_text_sentencized, f)
with open('wikidata_graphs1/wikidata_text_edge.json', 'w') as f:
    json.dump(wikidata_text_edge, f)