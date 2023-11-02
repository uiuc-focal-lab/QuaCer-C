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
from collections import deque
import warnings
import os
import wikipediaapi
from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration
import torch

class DBPediaReader():
    def __init__(self, dbpedia_url='https://dbpedia.org/sparql', max_threads=20, request_timeout=5):
        self.dbpedia_url = dbpedia_url
        self.visited = set()
        self.graph = {}
        self.entity_leaf = None
        self.max_depth = 0
        self.max_threads = max_threads
        self.executor = ThreadPoolExecutor(max_workers=self.max_threads)
        self.job_queue = None
        self.futures = []  # Define the futures list to keep track of submitted tasks
        self.thread_lock = threading.Lock()
        self.depth_reached = 0
        self.REQUEST_TIMEOUT = request_timeout
    def exploreWikidata(self, entity, cur_depth, item_labels=None, property_labels=None):
        try:
            if cur_depth >= self.max_depth or self.visited is None or item_labels is None or property_labels is None:
                self.depth_reached = cur_depth
                self.entity_leaf = entity
                return
            time.sleep(1.0)
            relations = self.queryWikidataForRelations(entity)
            for relatedEntity, propertyRelation, itemLabel, propertyLabel in relations:
                item_labels[relatedEntity] = itemLabel
                property_labels[propertyRelation] = propertyLabel
                if entity not in self.graph:
                    self.graph[entity] = {}
                self.graph[entity][relatedEntity] = propertyRelation
                if cur_depth + 1 > self.depth_reached:
                    self.depth_reached = cur_depth + 1
                    self.entity_leaf = entity
                if relatedEntity not in self.visited:
                    with self.thread_lock:
                        self.visited.add(relatedEntity)
                    self.job_queue.put((relatedEntity, cur_depth + 1, item_labels, property_labels))
        except Exception as e:
            print(f"Exception: {e}")
            raise(e)
    def queryWikidataForRelations(self, entity):
    # The updated SPARQL query based on your input
    #print(entity)
        sparql = f"""
        SELECT ?relatedEntity ?propertyRelation ?relatedEntityLabel ?propertyLabel
        WHERE {{
            <http://dbpedia.org/resource/{entity}> ?propertyRelation ?relatedEntity .
            ?relatedEntity rdfs:label ?relatedEntityLabel .
            ?propertyRelation rdfs:label ?propertyLabel .
            FILTER (lang(?relatedEntityLabel) = "en")
            FILTER (lang(?propertyLabel) = "en")
            FILTER (?propertyRelation != <http://dbpedia.org/ontology/wikiPageWikiLink>) 
        }}
        LIMIT 10
        """ # filtered out wikipedia page links, which are not correct next entities
        try:
            data = requests.get(self.dbpedia_url, params={'query': sparql, 'format': 'json'}, timeout=self.REQUEST_TIMEOUT).json()
            relatedEntities = [
                (
                    binding['relatedEntity']['value'].split('/')[-1], 
                    binding['propertyRelation']['value'].split('/')[-1],
                    binding['relatedEntityLabel']['value'],
                    binding['propertyLabel']['value']
                )
                for binding in data['results']['bindings']
            ]
            #time.sleep(1.0)  # To avoid hitting the server too frequently
            return list(random.sample(relatedEntities, min(3, len(relatedEntities)))) # return a sample of all related entries, instead of all entries

        except Exception as e:
            print(f"Query for entity {entity} timed out." + str(e)[:200])
            return []
    def run(self, start_entity, start_label, max_depth=5, save_file=None) -> dict:
        self.max_depth = max_depth
        self.visited = set([start_entity])
        self.graph = {}
        self.entity_leaf = None
        self.depth_reached = 0
        self.job_queue = queue.Queue()
        self.futures = []
        item_labels = {start_entity: start_label}
        property_labels = {}
        self.job_queue.put((start_entity, 0, item_labels, property_labels))
        start_time = time.time()
        while True:
            if self.job_queue.empty() and all(f.done() for f in self.futures):
                    break  # Exit the loop if the job queue is empty and all futures are done
            if not self.job_queue.empty() or [f for f in self.futures if not f.done()]:
                incomplete_futures = [f for f in self.futures if not f.done()]
                if len(incomplete_futures) < self.max_threads and not self.job_queue.empty():
                    args = self.job_queue.get()
                    self.futures.append(self.executor.submit(self.exploreWikidata, *args))
        self.executor.shutdown(wait=True)
        print(f"Num Visited entities: {len(self.visited)}")
        print(f"Time Taken: {time.time() - start_time} seconds")
        graph = copy.deepcopy(self.graph)
        self.graph = {}
        if save_file is not None:
            with open(save_file, 'w') as f:
                json.dump(graph, f)
        return graph

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
        for neighbor, relation in self.graph[start_vertex].items():
            if neighbor == target_vertex:
                return relation
        return None

class WikiText():
    def __init__(self, document_dir) -> None:
        self.document_dir = document_dir
        self.wiki_en = wikipediaapi.Wikipedia(user_agent='Knowledge Graph Project', language='en',
                                               extract_format=wikipediaapi.ExtractFormat.WIKI) #wikipedia english to extract text
        #check if document_dir exists
        if not os.path.exists(document_dir):
            os.makedirs(document_dir)
    
    def fetch_wikipedia_content(self, wiki_id):
        """Fetch the content of a Wikipedia page given its URL."""
        print(f"Fetching {wiki_id}")
        page_wiki = self.wiki_en.page(wiki_id)
        if not page_wiki.exists():
            warnings.warn(f"Page {wiki_id} does not exist.")
            return None
        return page_wiki.text
    
    def save_wiki_text(self, entity, new_doc_dir=None):
        #entity_url = dbpedia_id_to_wikipedia_url(entity)
        content = self.fetch_wikipedia_content(entity)
        if content is None:
            warnings.warn(f"Could not fetch content for {entity}")
            return
        if new_doc_dir is None:
            save_to_txt(content, f"{self.document_dir}/{entity}.txt")
            return
        save_to_txt(content, f"{new_doc_dir}/{entity}.txt")
        
class RAGModel():
    def __init__(self, path, passage_path, index_path, retrieve_sep=False, cuda=True) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.retriever = RagRetriever.from_pretrained(
        path, index_name="custom", passages_path=passage_path, index_path=index_path
        )
        self.retrieve_sep = retrieve_sep
        if not retrieve_sep:
            self.model = RagSequenceForGeneration.from_pretrained(path)
        else:
            self.model = RagSequenceForGeneration.from_pretrained(path, use_dummy_dataset=True)
        self.cuda = cuda
        if cuda:
            self.model.cuda()

    def run_rag(self, question: str):
        inputs = self.tokenizer(question, return_tensors="pt")
        input_ids = inputs["input_ids"]
        context = None
        if self.cuda:
            input_ids = input_ids.cuda()
        if not self.retrieve_sep:
            outputs = self.model.generate(input_ids)
        else:
            question_hidden_states = self.model.question_encoder(input_ids=input_ids)[0]

            docs_dict = self.retriever(input_ids.cpu().detach().numpy(),
                                        question_hidden_states.cpu().detach().numpy(), return_tensors="pt")
            doc_scores = torch.bmm(question_hidden_states.unsqueeze(1),
                                    docs_dict["retrieved_doc_embeds"].float().transpose(1, 2).cuda()).squeeze(1)
            
            outputs = self.model.generate(context_input_ids=docs_dict["context_input_ids"].cuda(),
                        context_attention_mask=docs_dict["context_attention_mask"].cuda(),
                        doc_scores=doc_scores.cuda())
            context = self.tokenizer.batch_decode(docs_dict["context_input_ids"], skip_special_tokens=True)
        
        output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output, context

# Helper functions
def dbpedia_id_to_wikipedia_url(dbpedia_id):
        """Convert a DBpedia ID to its corresponding Wikipedia URL."""
        return "https://en.wikipedia.org/wiki/"+dbpedia_id

def save_to_txt(content, filename):
    """Save a string content to a .txt file."""
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

def docs_to_csv(docs_dir, csv_filename):
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w') as csv_file:
            csv_file.write("filename,content\n")
    for file in os.listdir(docs_dir):
        if file.endswith(".txt"):
            with open(f"{docs_dir}/{file}", 'r') as f:
                content = f.read()
                with open(csv_filename, 'a') as csv_file:
                    csv_file.write(f"{file},{content}\n")
    
def load_graph(file_name):
    with open(file_name, 'r') as f:
        graph = json.load(f)
    return graph