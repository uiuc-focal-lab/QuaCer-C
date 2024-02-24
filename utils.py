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
from vertexai.language_models import ChatModel, InputOutputTextPair
from vertexai.preview.generative_models import GenerativeModel, ChatSession
import google.generativeai as genai
from transformers import pipeline
import torch
import gc
from unidecode import unidecode

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

class GraphAlgos():
    def __init__(self, graph: dict) -> None:
        all_vertices = set(graph.keys()) | {neighbor for neighbors in graph.values() for neighbor in neighbors.keys()}
        self.graph = {vertex: graph.get(vertex, {}) for vertex in all_vertices}
    def bfs(self, start):
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
        #bfs search for k depth
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
    
    def generate_query_for_path(self, path):
        query = "What is "
        for i in range(len(path) - 1, 1, -1):
            query += f"the {self.get_relation_for_vertex(path[i-1], path[i])} of "
        query += f"the {self.get_relation_for_vertex(path[0], path[1])} of {path[0]}?"
        return query
    
    def generate_query_for_vertices(self, start, end, k=5, path=None):
        #print(f"Generating query for {start} -> {end}")
        if path is None:
            path = self.get_path_for_vertices(start, end, k)
        if path is None:
            return None
        return self.generate_query_for_path(path), len(path)
    
    def sample_random_vertex(self, vertex_list=None):
        if vertex_list is None:
            vertex_list = list(self.graph.keys())
        return random.choice(vertex_list)
    
    def generate_random_path(self, path_len=25, source=None):
        path = None
        i = 0
        while path is None and i < 500:
            if source is None:
                start = self.sample_random_vertex()
            else:
                start = source
            path = self.dfs_path(start, path_len)
            i += 1
        return path
    
    def generate_random_query(self, k=5, return_path=False, source=None):
        path_len = random.randint(1, k)
        path = self.generate_random_path(path_len, source)
        if path is None:
            return None
        # print(f"Query Path: {str(path[start_idx:path.index(end)+1])}")
        start = path[0]
        end = path[-1]
        if return_path:
            return self.generate_query_for_vertices(start, end, k, path), start, end, path
        return self.generate_query_for_vertices(start, end, k, path), start, end, None
    
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
                        if neighbor not in subgraph:
                            subgraph[neighbor] = {}
                        subgraph[neighbor][current_vertex] = weight  # Include reverse edge for undirected graph
                        
        return subgraph
    def get_best_vertices(self, num=1000):
        graph_copy = self.graph.copy()
        vertices_out = sort_vertices_by_outdegree(graph_copy)
        return vertices_out[:num]

class GeminiChecker():
    def __init__(self) -> None:
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
        self.chat = self.model.start_chat()
        self.generation_config = {'temperature':0}
        self.count = 0

    def reset_chat(self):
        self.chat = self.model.start_chat()
        self.count = 0

    def checker(self, prompt: str, return_response=False):
        """
        input in form: "Question: What is David Beckham's Nationality? Ground Truth: England. Model Answer: British"
        returns 1 for yes 0 for no and response if return response = True
        """
        self.count += 1
        if self.count > 50 == 0:
            self.reset_chat()
        response = self.chat.send_message(
            f"""context: You are a helpful assistant. Your inputs will consist of a question and a correct answer, and a answer from a model. Your response should be a yes if the model's answer means the correct answer, else answer no.
            user:{prompt}""", generation_config=self.generation_config
        )
        response_text = response.text.lower()
        answer = 1 if 'yes' in response_text else 0
        if return_response:
            return answer, response
        return answer, None

    def form_prompt(self, question, correct_ans, model_ans):
        return f"Question: {question} Ground Truth: {correct_ans}. Model Answer: {model_ans}"
    
    def raw_checker(self, question, correct_ans, model_ans, return_response=False):
        prompt = self.form_prompt(question, correct_ans, model_ans)
        return self.checker(prompt, return_response)

class MistralChecker():
    def __init__(self, model_path, device='cuda:1') -> None:
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
            {"role": "user", "content": "context: You are an honest and fair correct answer evaluator. Your inputs will consist of a question and a correct answer, and a answer from a model. Your response should start with a single word either a yes if the model's answer means the same as correct answer technically and semantically, else the starting word should be no."},
            {"role": "assistant", "content": "Okay, I will ensure technical and semantic correctness of model answers."},
            {"role": "user", "content": prompt}
        ]
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=1, do_sample=False, pad_token_id=self.tokenizer.eos_token_id, temperature=0.0)
        decoded = self.tokenizer.batch_decode(generated_ids[:, model_inputs.size(1):])
        decoded = str(decoded[0]).strip().lower().strip()

        correct_ans = int('yes' in decoded)
        del model_inputs, generated_ids
        if return_response:
            return correct_ans, decoded
        return correct_ans, None

    def raw_checker(self, question, correct_ans, model_ans, return_response=False):
        prompt = self.form_prompt(question, correct_ans, model_ans)
        return self.checker(prompt, return_response)

def get_alias(id, aliases):
    aliases_choices = aliases[id]
    return random.choice(aliases_choices)
    
def form_alias_question(question, path, entity_aliases, relation_aliases, name2id, graph_algos):
    entity_replace = path[0]
    entity_id = name2id[entity_replace]
    entity_alias = get_alias(entity_id, entity_aliases)
    entity_alias = unidecode(entity_alias)
    question = question.replace(entity_replace, entity_alias)
    for i in range(len(path)-1):
        relation_name = graph_algos.get_relation_for_vertex(path[i], path[i+1])
        relation_name = unidecode(relation_name).lower()
        relation_id = name2id[relation_name]
        relation_alias = get_alias(relation_id, relation_aliases)
        relation_alias = unidecode(relation_alias)
        question = question.replace(relation_name, relation_alias)

    return question, entity_alias

def load_aliases(path):
    possible_entities = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            line = [x.strip() for x in line]
            possible_entities[line[0]] = line[1:]
    return possible_entities