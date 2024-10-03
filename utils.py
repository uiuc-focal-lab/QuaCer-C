import os
import requests
from concurrent.futures import ThreadPoolExecutor
import time
import queue
import concurrent
import threading
import random
import copy
from collections import deque
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch
import gc
from unidecode import unidecode
import re
import math

FEW_SHOT_EXAMPLES = """
Example questions and correct answers:
Common Context: entity_B is the son of entity_A. entity_E is the sister of entity_A. entity_B leads entity_C. Entity_D is a member of Entity_C. Entity_D is a friend of entity_E. entity_E has mother entity_F who likes the services of entity_C.
question 1: entity_A->(father of)->(leader of)->?
Options: 1. entity_F,\n 2. entity_C,\n 3. entity_D,\n 4. entity_E,\n 5. entity_B
answer: 2. entity_C
explanation: entity_A->(father of)entity_B->(leader of)entity_C
how to get answer: find who entity_A is father of to get entity_B, then find what B is the leader of to get entity_C which the final answer.

question 2: entity_B->(chief of)->(constitues)->(companion of)->?
Options: 1. entity_F,\n 2. entity_C,\n 3. entity_D,\n 4. entity_E,\n 5. entity_A
answer: 4. entity_E
explanation: entity_B->(chief of)entity_C->(constitues)entity_D->(companion of)entity_E
how to get answer: find what entity_B is the chief of to get entity_C, find what entity_C constitutes of to get entity_D, then find the companion of entity_D of to get entity_E.
"""

# question 3: entity_C->(has membership of)->(friend)->(mom)->?
# Options: 1. entity_F,\n 2. entity_C,\n 3. entity_D,\n 4. entity_E,\n 5. entity_A
# answer: 1. entity_F
# explanation: entity_C->(has membership of)entity_D->(friend)entity_E->(mom)entity_F
# how to get answer: find who entity_C has membership of to get entity_D, find who is the friend of entity_D to get entity_E, then find the mom of entity_E to get entity_F.

# question 4: entity_A->(sister)->(mother)->?
# Options: 1. entity_F,\n 2. entity_C,\n 3. entity_D,\n 4. entity_E,\n 5. entity_A
# answer: 5. entity_F
# explanation: entity_A->(sister)entity_E->(mother)entity_F
# how to get answer: find who entity_A's sister is to get entity_E, then find the mother of entity_E to get entity_F.

# question 5: entity_A->(son)->(leader)->(member)->(friend)?
# Options: 1. entity_F,\n 2. entity_C,\n 3. entity_D,\n 4. entity_E,\n 5. entity_A
# answer: 3. entity_E
# explanation: entity_A->(son)entity_B->(leader)entity_C->(member)entity_D->(friend)entity_E
# how to get answer: find who entity_A's son is to get entity_B, then find the leader of entity_B to get entity_C, find the member of entity_C to get entity_D, then find the friend of entity_D to get entity_E.

LLM_PROMPT_TEMPLATE = """
{few_shot_examples}

Actual Query:
Given Context:
{context}

Answer the question:
{query}

answer the question by selecting the correct answer from the following options:
{options}

The format for beginning your response is:
correct answer: <option_number>. <answer>, because <succinct reason>

follow this exact format and only choose from the given options
"""

# CHECKER_INITIAL_PROMPT = f"""
# You are a correct answer evaluator. Your inputs will consist of a question and a correct answer, and a answer from a model.
# The questions will be of the form:
# {FEW_SHOT_EXAMPLES}

# Your response should start with 'correct' or 'wrong' based on whether the model's answer means the correct answer in both technical and semantic terms.

# Start response with 'correct' or 'wrong' only and nothing else. Then explain the reasons.
# """

def sort_vertices_by_outdegree(graph):
    """
    Sorts the vertices in the graph based on their outdegree in descending order.
    
    :param graph: A dict of dicts representing the graph, where graph[node_start] = {node1: edge_weight, node2: edge_weight, ...}
    :return: A list of vertices sorted by their outdegree in descending order.
    """
    # Calculate outdegree for each vertex
    outdegree = {vertex: len(edges) for vertex, edges in graph.items()}
    
    # Sort vertices based on outdegree in descending order
    sorted_vertices = sorted(outdegree, key=outdegree.get, reverse=True)
    
    return sorted_vertices

def sort_vertices_by_measure(graph, k, weights):
    """
    Sorts the vertices in the graph based on a custom measure. 

    The measure considers the number of neighbors at different distances (up to k) and weights assigned to each distance.

    :param graph: A dict of dicts representing the graph, where graph[node_start] = {node1: edge_weight, node2: edge_weight, ...}
    :param k: The maximum distance to consider for neighbors.
    :param weights: A dictionary mapping distances to weights.
    :return: A list of vertices sorted by the calculated measure in descending order.
    """
    # Calculate the measure for each vertex
    measure = {vertex : 0 for vertex in graph.keys()}
    num_neighbors = {(vertex, 0): 1 for vertex in graph.keys()}
    for i in range(1, k+1):
        for vertex, ents_rels in graph.items():
            key = (vertex, i)
            total = 0
            for neighbor in ents_rels.keys():
                total += num_neighbors.get((neighbor, i-1), 1)
            num_neighbors[key] = total
    
    for key, value in num_neighbors.items():
        if key[1] <= 0:
            continue
        if num_neighbors[(key[0], key[1]-1)] == 0:
            continue
        measure[key[0]] += (value/num_neighbors[(key[0], key[1]-1)]) * weights[key[1]]
    # Sort vertices based on the measure
    sorted_vertices = sorted(measure, key=measure.get, reverse=True)
    
    return sorted_vertices

def generate_answer_options(correct_ans, distractors, path_entities, random_entities, wikidata_util,
                            min_num_options=4):
    # assumption: distractors are ordered by decreasing distracting power
    # assumption: path_entities is also ordered by increasing proximity to the correct answer
    #random entities: list[list[str]], 0th element: list[ents] that are from path[-1], so on
    #   A
    #  / \r
    # B   E
    # |r
    # C
    assert correct_ans == path_entities[0]
    options = [(correct_ans, path_entities[1])]
    options.extend(distractors)
    if len(options) >= min_num_options:
        options = options[:min_num_options]
        random.shuffle(options)
        return options
    if len(path_entities) > 0:
        options.extend([(path_entities[i], path_entities[i+1]) for i in range(1, len(path_entities)-1)])
    if len(options) >= min_num_options:
        options = options[:min_num_options]
        random.shuffle(options)
        return options
    rel_to_find = wikidata_util[path_entities[1]][path_entities[0]]

    good_random_entities = [] # has the same relation as that connecting correct answer to its parent

    for i in range(1, len(random_entities)):
        parent_ent = path_entities[i]
        for pos_ent in random_entities[i]:
            if i > 0:
                if pos_ent == path_entities[i-1]:
                    continue
            if wikidata_util[parent_ent][pos_ent] == rel_to_find:
                good_random_entities.append((pos_ent, parent_ent))
    options.extend(good_random_entities)
    num_random = 2
    random_options = []
    for i in range(1, len(random_entities)):
        parent_ent = path_entities[i]
        pos_ents = [(ent, parent_ent) for ent in random_entities[i] if (ent, parent_ent) not in options]
        random_options.extend(random.sample(pos_ents, min(num_random, len(pos_ents))))  
    options.extend(random_options)
    if len(options) >= min_num_options:
        options = options[:min_num_options] #ensure correct answer is in the options as it is at index 0
        random.shuffle(options)
    else:
        #in case we still don't have enough options, override num_random on origin vertex to get more options
        origin_vertex = path_entities[-1]
        pos_ents = [(ent, origin_vertex) for ent in random_entities[-1] if (ent, origin_vertex) not in options]
        options_needed = min(min_num_options - len(options), len(pos_ents)) #ensure we only have min_num_options options
        options.extend(random.sample(pos_ents, options_needed))
        random.shuffle(options) 
    return options
        

        

class GraphAlgos():
    def __init__(self, graph: dict, entity_aliases:dict, relation_aliases:dict) -> None:
        """
        Initializes the GraphAlgos object with the given graph and alias dictionaries.

        :param graph: A dict of dicts representing the graph.
        :param entity_aliases: A dictionary mapping entity IDs to lists of aliases.
        :param relation_aliases: A dictionary mapping relation IDs to lists of aliases.
        """
        all_vertices = set(graph.keys()) | {neighbor for neighbors in graph.values() for neighbor in neighbors.keys()}
        self.graph = {vertex: graph.get(vertex, {}) for vertex in all_vertices}
        self.rel2ent_graph = {vertex: {rel:[]  for rel in neighbors.values()} for vertex, neighbors in graph.items()}
        for vertex, neighbors in self.graph.items():
            if vertex not in self.rel2ent_graph:
                assert self.graph[vertex] == {}, f"Vertex {vertex} in graph but not in rel2ent_graph"
                self.rel2ent_graph[vertex] = {}
                continue
            for neighbor, rel in neighbors.items():
                self.rel2ent_graph[vertex][rel].append(neighbor)
        self.entity_aliases = entity_aliases
        self.relation_aliases = relation_aliases
    def bfs(self, start):
        """
        Performs Breadth-First Search (BFS) on the graph starting from the specified vertex.

        :param start: The starting vertex for the BFS.
        :return: A dictionary mapping vertices to their distances from the starting vertex.
        """
        distances = {vertex: -float('infinity') for vertex in self.graph}
        distances[start] = 0
        queue = deque([start])

        while queue:
            current_vertex = queue.popleft()
            for neighbor in self.graph[current_vertex].keys():
                if distances[neighbor] == -float('infinity'):
                    distances[neighbor] = distances[current_vertex] + 1
                    queue.append(neighbor)

        return distances

    def compute_diameter(self):
        max_distance = 0
        for vertex in self.graph:
            distances = self.bfs(vertex)
            max_distance = max(max_distance, max(distances.values()))

        return max_distance

    def dfs(self, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)

        for next in self.graph[start].keys():
            if next not in visited:
                continue
            self.dfs(next, visited)
        return visited
    
    def dfs_path(self, start, length, path=None):
        if path is None:
            path = [start]

        if len(path) - 1 == length:  # -1 because the length of path includes the starting node
            return path  # Return the path if it's of the desired length
        if start not in self.graph:
            return None
        neighbors = self.graph[start]
        if not neighbors:
            return None # Return None if there are no neighbors
        neighbors_sample = random.sample(neighbors.keys(), len(neighbors))
        for neighbor in neighbors_sample:
            if neighbor not in path:
                extended_path = path + [neighbor]
                result = self.dfs_path(neighbor, length, extended_path)
                if result:
                    return result  # Return the first path of the desired length found
        return None  # Return None if no path of the desired length is found
    
    def get_vertices(self):
        return list(self.graph.keys())

    def get_relations(self):
        return list({relation for neighbors in self.graph.values() for relation in neighbors.values()})

    def get_relation_for_vertex(self, start_vertex, target_vertex):
        if target_vertex in self.graph[start_vertex]:
            return self.graph[start_vertex][target_vertex]
        return None
    
    def get_path_for_vertices(self, start, end, k=5):
        """
        Finds a path between two vertices in the graph using Breadth-First Search (BFS) with a limited depth.

        :param start: The starting vertex for the path search.
        :param end: The target ending vertex for the path.
        :param k: The maximum depth to explore in the BFS (default: 5).
        :return: A list of vertices representing the found path, or None if no path is found within the specified depth.
        """
        queue = deque([[start]])
        visited = set()
        while queue:
            path = queue.popleft()
            vertex = path[-1]
            if vertex == end:
                return path
            elif vertex not in visited and len(path) < k:
                for neighbor in self.graph[vertex].keys():
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
                visited.add(vertex)
        return None
    
    def get_queries_for_relpath(self, rel_path: list, start_vertex: str) -> list:
        """
        Finds all possible paths in the graph that follow the given sequence of relations, starting from the specified vertex. 

        This function recursively explores the graph, considering neighbors connected by the relations in the `rel_path` list.

        :param rel_path: A list of relations representing the desired path.
        :param start_vertex: The starting vertex for the path search.
        :return: A list of lists, where each inner list represents a valid path found in the graph following the specified relations.
        """
        if len(rel_path) == 0:
            return [[start_vertex]]
        rel_use = rel_path[0]
        neighbors = self.rel2ent_graph[start_vertex].get(rel_use, [])
        if len(neighbors) == 0:
            return []
        possible_paths = []
        for neighbor in neighbors:
            possible_neighbor_paths = self.get_queries_for_relpath(rel_path[1:], neighbor)
            if len(possible_neighbor_paths) == 0:
                continue
            for path in possible_neighbor_paths:
                possible_paths.append([start_vertex] + path)
        return possible_paths
    
    def generate_query_for_path(self, path):
        def query_path_aux(path):
            if len(path) == 1:
                entity_alias = get_alias(path[0], self.entity_aliases)
                return entity_alias, entity_alias
            
            rel = self.get_relation_for_vertex(path[-2], path[-1])
            rel_alias = get_alias(rel, self.relation_aliases)
            rest_query, entity_alias = query_path_aux(path[:-1])
            query = f"{rest_query}->({rel_alias}) "
            return query, entity_alias
            
        query = ""
        rest_query, entity_alias = query_path_aux(path)
        query += rest_query + '->?'
        return query, entity_alias, len(path)
    
    def generate_query_for_vertices(self, start, end, k=5, path=None):
        if path is None:
            path = self.get_path_for_vertices(start, end, k)
        if path is None:
            return None
        return self.generate_query_for_path(path)
    
    def sample_random_vertex(self, vertex_list=None):
        if vertex_list is None:
            vertex_list = list(self.graph.keys())
        return random.choice(vertex_list)
    
    def generate_random_path(self, path_len=25, source=None):
        """
        Generates a random path in the graph with the specified length.

        The generated path attempts to be unique, meaning there should be only one possible path 
        corresponding to the sequence of relations in the path.

        :param path_len: The desired length of the path.
        :param source: (Optional) The starting vertex for the path. If not provided, a random vertex is chosen.
        :return: A list of vertices representing the generated path, or None if a unique path cannot be found within 500 attempts. 
        """
        path = None
        i = 0
        while path is None and i < 500:
            if source is None:
                start = self.sample_random_vertex()
            else:
                start = source
            path = self.dfs_path(start, path_len)
            #check if path in unique
            if path is not None:
                rel_path = [self.get_relation_for_vertex(path[i], path[i+1]) for i in range(len(path)-1)]
                # if len(self.rel2ent_graph[path[-2]][rel_path[-1]]) > 1:
                #     path = None #answer should be unique
                #     continue
                possible_paths = self.get_queries_for_relpath(rel_path, start)
                assert len(possible_paths) > 0, f"No possible paths found for {start} -> {path[-1]}"
                if len(possible_paths) > 1:
                    path = None #not unique
            i += 1
        return path
    
    def get_best_distractor(self, start_vertex, path, do_choose=True):
        """
        Gets the best distractor vertex for a given path.
        Assumes that the path is unique. If the path is not unique, the behavior is undefined.

        Returns a Union(tuple(string, string), list(tuple(string, string)))
        the first string is the distractor and the second string in the node in the path that leads to the distractor
        do_choose=True returns a single tuple, do_choose=False returns a list of tuples of possible distractors
        """
        rel_path = [self.get_relation_for_vertex(path[i], path[i+1]) for i in range(len(path)-1)]
        best_distractor = None
        all_distractors_pos = []
        distractor_weights = []
        for i in range(len(rel_path)-2, -1, -1):
            cur_vertex = path[i]
            neighbors = self.rel2ent_graph[cur_vertex].get(rel_path[i], [])
            if len(neighbors) == 0:
                continue
            pos_distractors = [neighbor for neighbor in neighbors if neighbor not in path]
            for distractor in pos_distractors:
                all_distractors_pos.append((distractor, cur_vertex))
                distractor_weights.append(i)
                
        if not do_choose:
            return all_distractors_pos
        if len(all_distractors_pos) > 0:
            best_distractor_index = random.choices(range(len(all_distractors_pos)), weights=distractor_weights, k=1)[0]
            # print(f"Best Distractor Weight: {distractor_weights[best_distractor_index]}")
            best_distractor = all_distractors_pos[best_distractor_index]
        if best_distractor is not None:
            assert best_distractor not in path, f"Best distractor {best_distractor} is in path {path}"
        return best_distractor
    
    
    def generate_random_query(self, k=5, return_path=False, source=None):
        """
        Generates a random query by sampling a random path in the graph.

        :param k: The maximum path length.
        :param return_path: If True, returns the path used to generate the query.
        :param source: Optional starting vertex for the path.
        :return: A tuple containing the generated query, starting vertex, correct answer(s), and optionally the path.
        """
        path_len = random.randint(1, k)
        path = self.generate_random_path(path_len, source)
        if path is None:
            return None
        # print(f"Query Path: {str(path[start_idx:path.index(end)+1])}")
        start = path[0]
        end = path[-1]
        rel_correct = self.get_relation_for_vertex(path[-2], path[-1])
        # print(rel_correct, "rel correct")
        correct_answers = self.rel2ent_graph[path[-2]][rel_correct]
        if return_path:
            return self.generate_query_for_vertices(start, end, k, path), start, correct_answers, path
        return self.generate_query_for_vertices(start, end, k, path), start, correct_answers, None
    
    def create_subgraph_within_radius(self, start_vertex, k):
        """
        Creates a subgraph containing all vertices within a radius of k from the given start_vertex.
        :param start_vertex: The vertex from which the radius is measured.
        :param k: The radius within which vertices will be included in the subgraph.
        :return: A subgraph as a dict of dicts including all vertices within the specified radius.
        """
        visited = {start_vertex: 0}  # Tracks visited vertices and their levels
        queue = [(start_vertex, 0)]  # Queue for BFS, storing tuples of (vertex, level)
        subgraph = {}  # The resulting subgraph
        
        while queue:
            current_vertex, level = queue.pop(0)
            
            if level > k:
                break  # Stop if the current level exceeds k
            
            # Initialize subgraph entry for the current vertex if not already present
            if current_vertex not in subgraph:
                subgraph[current_vertex] = {}
            
            # Explore neighbors
            for neighbor, weight in self.graph.get(current_vertex, {}).items():
                if neighbor not in visited or visited[neighbor] > level + 1:
                    visited[neighbor] = level + 1
                    queue.append((neighbor, level + 1))
                    
                    # Add the neighbor to the subgraph with the corresponding edge weight
                    if level + 1 <= k:
                        subgraph[current_vertex][neighbor] = weight
                        
        return subgraph
    def get_best_vertices(self, num=1000, method='outdegree', **kwargs):
        """
        Gets the best vertices from the graph based on the specified method.
        :param num: The number of vertices to return.
        :param method: The method to use for selecting the best vertices. Supported methods are:
            - 'outdegree': Selects vertices with the highest outdegree.
            - 'measure': Selects vertices based on a custom measure considering neighbor distances and weights.
        :param kwargs: Additional keyword arguments for the chosen method. 
            For 'measure':
                - k (int, optional): The maximum distance to consider for neighbors (default: 4).
                - weights (dict, optional): A dictionary mapping distances to weights (default: {1: 0.75, 2: 1.25, 3: 1, 4: 0.25}).
        :return: A list of the best vertices, sorted according to the chosen method.
        :raises ValueError: If an unsupported method is specified.
        """
        graph_copy = self.graph.copy()
        if method == 'outdegree':
            vertices_out = sort_vertices_by_outdegree(graph_copy)
        elif method == 'measure':
            k = kwargs.get('k', 4)
            weights = kwargs.get('weights', {1: 0.75, 2: 1.25, 3: 1, 4: 0.25})
            vertices_out = sort_vertices_by_measure(graph_copy, k, weights)
        else:
            raise ValueError(f"Method {method} not supported")
        return vertices_out[:num]

class MistralChecker():
    def __init__(self, model_path, device='cuda:1') -> None:
        """
        Checks the correctness of a model answer using the Mistral language model.

        :param prompt: The prompt containing the question, ground truth answer, and model answer.
        :param return_response: If True, returns the raw response from Mistral.
        :return: A tuple containing 1 if the answer is correct, 0 otherwise, and optionally the raw response from Mistral.
        """
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.generation_config = {'temperature':0}
        self.count = 0
    def form_prompt(self, question, correct_ans, model_ans):
        return f"Question: {question} Ground Truth: {correct_ans}. Model Answer: {model_ans}"

    def checker(self, prompt: str, return_response=False):
        """
        input in form: "Question: What is David Beckham's National
        ity? Ground Truth: England. Model Answer: British"
        returns 1 for yes 0 for no and response if return response = True
        """
        
        messages = [
            {"role": "user", "content": f"{CHECKER_INITIAL_PROMPT}"},
            {"role": "assistant", "content": f"Okay, I will evaluate the correctness of the answers and the reasoning of the model_answer with respect to the correct answers technically and semantically."},
            {"role": "user", "content": f"Evaluate Input data: [{prompt}] "}
        ]
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=2, do_sample=False, pad_token_id=self.tokenizer.eos_token_id, temperature=0.0)
        decoded = self.tokenizer.batch_decode(generated_ids[:, model_inputs.size(1):])
        print(decoded)
        decoded = str(decoded[0]).strip().lower().strip()
        correct_ans = int('correct' in decoded)
        del model_inputs, generated_ids
        if return_response:
            return correct_ans, decoded
        return correct_ans, None

    def raw_checker(self, question, correct_ans, model_ans, return_response=False):
        prompt = self.form_prompt(question, correct_ans, model_ans)
        return self.checker(prompt, return_response)

def get_alias(id, aliases):
    """
    Retrieves a random alias for the given ID from the provided aliases dictionary.

    :param id: The ID for which to retrieve an alias.
    :param aliases: A dictionary mapping IDs to lists of aliases.
    :return: A randomly chosen alias from the list associated with the ID.
    """
    aliases_choices = aliases[id]
    return random.choice(aliases_choices)
    
def form_alias_question(question, path, entity_aliases, relation_aliases, entity_name2id, relation_name2id, graph_algos):
    """
    Transforms a question by replacing entities and relations with their aliases.

    Redundant, Not being used because of errors in name2id
    """
    entity_replace = path[0]
    entity_id = entity_name2id[entity_replace]
    entity_alias = get_alias(entity_id, entity_aliases)
    entity_alias = unidecode(entity_alias)
    question = question.replace(entity_replace, entity_alias)
    for i in range(len(path)-1):
        relation_name = graph_algos.get_relation_for_vertex(path[i], path[i+1])
        relation_name = unidecode(relation_name).lower()
        try:
            relation_id = relation_name2id[relation_name]
            try:
                relation_alias = get_alias(relation_id, relation_aliases)
            except KeyError:
                try:
                    relation_alias = get_alias(relation_id, entity_aliases) #relations maybe entities too(eg. country)
                except KeyError:
                    relation_alias = relation_name
                    print(f"Relation {relation_name, relation_id} not found in aliases using default")
        except KeyError:
            relation_alias = relation_name
            print(f"Relation {relation_name} not found in name2id using default")
        relation_alias = unidecode(relation_alias)
        question = question.replace(relation_name, relation_alias)

    return question, entity_alias

def load_aliases(path):
    """
    Loads aliases from a file.

    :param path: The path to the file containing aliases.
    :return: A dictionary mapping entity/relation IDs to lists of aliases. 
    """
    possible_entities = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            line = [x.strip() for x in line]
            possible_entities[line[0]] = line[1:]
    return possible_entities

def form_context_list(query_path, wikidata_text_edge, wikidata_util, entity_top_alias):
    context_list = []
    for i in range(len(query_path)-1):
        source, dest = query_path[i], query_path[i+1]
        if dest not in wikidata_text_edge[source]:
            print(source, dest)
            return None
        relevant_text = wikidata_text_edge[source][dest]
        if wikidata_util[source][dest] == 'P20': # death place
            relevant_text = [f"{entity_top_alias[source]} died at {entity_top_alias[dest]}"]
        else:
            relevant_text = []
        relevant_text.extend(wikidata_text_edge[source][dest])
        # if type(relevant_text) == list:
        #     relevant_text = ' '.join(relevant_text)
        context_list.append(relevant_text)
    return context_list

# def simple_checker(model_answer, correct_answer, correct_answer_aliases, correct_id):
#     """
#     Performs a simple check to see if the model answer is correct.

#     Checks if the correct answer or any of its aliases are present in the model answer (case-insensitive).

#     :param model_answer: The answer generated by the model.
#     :param correct_answer: The ground truth correct answer.
#     :param correct_answer_aliases: A dictionary mapping correct answer IDs to their aliases.
#     :param correct_id: The ID of the correct answer.
#     :return: 1 if the model answer is considered correct, 0 otherwise.
#     """
#     model_answer = unidecode(model_answer).lower()
#     correct_answer = unidecode(correct_answer).lower()
#     pattern = r'\b' + re.escape(correct_answer) + r'\b'
#     matches = re.search(pattern, model_answer)
#     if matches:
#         return 1

#     if correct_id not in correct_answer_aliases:
#         return 0
#     for answer_alias in correct_answer_aliases[correct_id]:
#         pattern = r'\b' + re.escape(unidecode(answer_alias).lower()) + r'\b'
#         matches = re.search(pattern, model_answer)
#         if matches:
#             return 1
#     return 0

def dumb_checker(model_answer, correct_answer_num):
    """
    Performs a simple check to see if the model answer is correct.

    Checks if the correct answer or any of its aliases are present in the model answer (case-insensitive).

    :param model_answer: The answer generated by the model.
    :param correct_answer_num: The ground truth correct answer number.
    :return: 1 if the model answer is considered correct, 0 otherwise.
    """
    model_answer = unidecode(model_answer).lower()
    correct_answer_num = unidecode(str(correct_answer_num)).lower()
    pattern = r'\bcorrect answer: \s*\**\s*[\[<{(]?' + re.escape(correct_answer_num) + '\s*\**\s*[\]>})]?[.]?'
    matches = re.search(pattern, model_answer)
    if matches:
        return 1
    return 0

def create_context_list(all_sents, relevant_sents_path, relevant_sents_opts, tokenizer, max_length=15000):
    # Flatten relevant_sents and calculate its total tokenized length
    if tokenizer is None:
        #for API only models
        combined_list = []
        all_sents_set = set()
        for a_sublist in all_sents:
            add_list = []
            for a_item in a_sublist:
                if a_item not in add_list and a_item not in all_sents_set:
                    add_list.append(a_item)
                    all_sents_set.add(a_item)
            if len(add_list) > 0:
                combined_list.append(add_list)
        return combined_list
    flat_relevant_sents = [item for sublist in relevant_sents_path for item in sublist]
    tokenized_relevant_sents = tokenizer(flat_relevant_sents, add_special_tokens=False)
    total_length_relevant_sents = sum(len(ids) for ids in tokenized_relevant_sents['input_ids'])

    if total_length_relevant_sents > max_length:
        print(f"Total length of relevant sentences ({total_length_relevant_sents}) exceeds maximum length ({max_length}).")
        return []

    option_sentences_add = []
    # Tokenize option sentences if we can
    for option_sents in relevant_sents_opts:
        tokenized_relevant_sents = tokenizer(option_sents, add_special_tokens=False)
        len_tokens = sum(len(ids) for ids in tokenized_relevant_sents['input_ids'])
        if len_tokens + total_length_relevant_sents <= max_length:
            option_sentences_add.append(option_sents)
    
    relevant_sents = relevant_sents_path + option_sentences_add
    flat_relevant_sents = [item for sublist in relevant_sents for item in sublist]
    set_relevant_sents = set(flat_relevant_sents)
    to_include = set()
    current_length = total_length_relevant_sents
    seen_strings = set(flat_relevant_sents)  # To track duplicates and included strings

    # Check each item in all_sents for inclusion
    for a_sublist in all_sents:
        for a_item in a_sublist:
            item_tokenized_length = len(tokenizer(a_item, add_special_tokens=False))
            if a_item in set_relevant_sents or (a_item not in seen_strings and current_length + item_tokenized_length <= max_length):
                to_include.add(a_item)
                seen_strings.add(a_item)
                if a_item not in set_relevant_sents:  # Only add to current_length if not already accounted for
                    current_length += item_tokenized_length

    combined_list = []
    all_sents_set = set()
    # Ensure each item is only added once
    for a_sublist in all_sents:
        add_list = []
        for a_item in a_sublist:
            if a_item in to_include and a_item not in add_list and a_item not in all_sents_set:
               add_list.append(a_item)
               all_sents_set.add(a_item)
        if len(add_list) > 0:
            combined_list.append(add_list)
    return combined_list

def get_all_context(query_path, wikidata_text_sentencized):
    all_context = []
    for entity in query_path:
        if entity in wikidata_text_sentencized:
            all_context.append(wikidata_text_sentencized[entity])
    return all_context
        
def get_random_entities(query_path, wikidata_util):
    random_entities = []
    for i in range(len(query_path)):
        parent_ent = query_path[i]
        pos_ents = list(wikidata_util[parent_ent].keys())
        random_entities.append(pos_ents)
    return random_entities


def get_query_data(graph_algos, source, id2name, graph_text_edge, graph_text_sentencized, tokenizer, distractor_query=False, k=5, shuffle_context=True, max_context_length=30000):
    while True:
        distractor=None
        node_distracted=None
        query_results = graph_algos.generate_random_query(k, return_path=True, source=source) # allow sampling with replacement
        if query_results is None:
            return None
        all_distractors = []
        if distractor_query:
            distractor_tuple = graph_algos.get_best_distractor(query_results[1], query_results[3])
            if distractor_tuple is None:
                continue
            distractor, node_distracted = distractor_tuple
            all_distractors = graph_algos.get_best_distractor(query_results[1], query_results[3], do_choose=False)
            
        query_inf, _, correct_ids, ids_path = query_results
        query, entity_alias, k_num = query_inf
        path = [id2name[ids_path[i]] for i in range(len(ids_path))]
        if 'country of' in query:
            query = query.replace('country of', 'country location of') # to avoid ambiguity
        true_ids_path = ids_path.copy()
        # if not distractor_query:
        #     random_distractor_parent = random.choice(list(graph_text_edge.keys()))
        #     try:
        #         random_distractor = random.choice(list(graph_text_edge[random_distractor_parent].keys()))
        #     except:
        #         random_distractor = None
        #     if random_distractor not in true_ids_path and random_distractor is not None:
        #         distractor = random_distractor
        #         node_distracted = random_distractor_parent
        options = generate_answer_options(true_ids_path[-1], all_distractors, list(reversed(true_ids_path)), get_random_entities(list(reversed(true_ids_path)), graph_algos.graph), graph_algos.graph)
        relevant_context_list = form_context_list(true_ids_path, graph_text_edge, graph_algos.graph, entity_top_alias=id2name)
        
        relevant_options_context_list = []
        for ent, parent_ent in options:
            relevant_text = graph_text_edge[parent_ent][ent]
            # if type(relevant_text) == list:
            #     relevant_text = ' '.join(relevant_text)
            relevant_options_context_list.append(relevant_text)
        
        all_context = get_all_context(true_ids_path + [ent for ent, _ in options], graph_text_sentencized)
        context_list = create_context_list(all_context, relevant_context_list, relevant_options_context_list, tokenizer, max_length=max_context_length)
        if len(context_list) == 0:
            print(f"path ids true: {true_ids_path}, query: {query}, options: {options}")
            raise ValueError("Tokenizer exceeded")
        if distractor is not None:
            ids_path.append(distractor)
            distractor_text = graph_text_edge[node_distracted][distractor]
            context_list.append(distractor_text)
        if shuffle_context:
            random.shuffle(context_list)
        context = '\n'.join([' '.join(context_part) for context_part in context_list])
        if id2name[true_ids_path[0]].lower() != entity_alias.lower():
            context += f" {id2name[true_ids_path[0]]} is also known as {entity_alias}."
        rel_path = [graph_algos.get_relation_for_vertex(true_ids_path[i], true_ids_path[i+1]) for i in range(len(true_ids_path)-1)]
        rel_info = [id2name[rel] for rel in rel_path]
        rel_aliases_used = query.split('->')[1:-1]
        # print(f"rel_info: {rel_info}, rel_aliases_used: {rel_aliases_used}")
        assert len(rel_info) == len(rel_aliases_used)
        rel_context = ' '.join([f"{rel_aliases_used[i]} means the same as {rel_info[i]}" for i in range(len(rel_info))])
        context += f"\n{rel_context}"
        answer_options = [ent for ent, _ in options]
        random.shuffle(answer_options)
        return {'query':query, 'correct_answers':[id2name[correct_id] for correct_id in correct_ids], 'path_id':true_ids_path, 
                'path_en':path, 'context':context, 'correct_ids':correct_ids, 'distractor':distractor, 'answer_options': answer_options, 
                'correct_ans_num': answer_options.index(true_ids_path[-1])+1}