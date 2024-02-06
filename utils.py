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
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain.chains import LLMMathChain
from langchain.utilities import SerpAPIWrapper
import torch
from langchain.docstore import Wikipedia
from langchain.agents.react.base import DocstoreExplorer
import os
from vertexai.language_models import ChatModel, InputOutputTextPair
from vertexai.preview.generative_models import GenerativeModel, ChatSession
import google.generativeai as genai
from transformers import pipeline

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
        # print(entity)
        sparql = f"""
        SELECT ?relatedEntity ?propertyRelation ?relatedEntityLabel ?propertyLabel
        WHERE {{
            <http://dbpedia.org/resource/{entity}> ?propertyRelation ?relatedEntity .
            ?relatedEntity rdfs:label ?relatedEntityLabel .
            ?propertyRelation rdfs:label ?propertyLabel .
            FILTER (lang(?relatedEntityLabel) = "en")
            FILTER (lang(?propertyLabel) = "en")
            FILTER (?propertyRelation != <http://dbpedia.org/ontology/wikiPageWikiLink>)
            FILTER (?propertyRelation != <http://dbpedia.org/property/label>)
            FILTER (!REGEX(STR(?propertyRelation), "http://dbpedia.org/property/.+Info"))
            FILTER(?propertyRelation != <http://dbpedia.org/property/isoCode>)
           FILTER(?propertyRelation != <http://dbpedia.org/property/subdivisionType>)
           FILTER(?propertyRelation != <http://dbpedia.org/property/subdivisionName>)
           FILTER (!REGEX(STR(?propertyRelation), "http://dbpedia.org/property/.+subdivision"))
           FILTER (!REGEX(STR(?propertyRelation), "http://dbpedia.org/property/.+blank"))
           {{
                SELECT ?propertyRelation (COUNT(?related) AS ?count)
                WHERE {{
                    <http://dbpedia.org/resource/{entity}> ?propertyRelation ?related .
                }}
                GROUP BY ?propertyRelation
                HAVING (COUNT(?related) = 1)
            }}
        }}
        LIMIT 15
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
            #return list(random.sample(relatedEntities, min(15, len(relatedEntities)))) # return a sample of all related entries, instead of all entries
            return list(relatedEntities) #return all entities
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
        half_hours = 0
        while True:
            if self.job_queue.empty() and all(f.done() for f in self.futures):
                    break  # Exit the loop if the job queue is empty and all futures are done
            cur_time = time.time()
            if round((cur_time - start_time)) >= (half_hours)*1800:
                print(f"In process Num Visited entities: {len(self.visited)}, storing temp graph")
                if save_file is not None:
                    [save_file_name, ext] = save_file.split('.')
                    with open(save_file_name+str(round((cur_time - start_time)))+'.'+str(ext), 'w') as f:
                        json.dump(self.graph, f)
                half_hours += 1
                time.sleep(1)
            if not self.job_queue.empty() or [f for f in self.futures if not f.done()]:
                incomplete_futures = [f for f in self.futures if not f.done()]
                if len(incomplete_futures) < self.max_threads and not self.job_queue.empty():
                    args = self.job_queue.get()
                    self.futures.append(self.executor.submit(self.exploreWikidata, *args))
        self.executor.shutdown(wait=True)
        print(f"Num Visited entities: {len(self.visited)}")
        print(f"Time Taken: {time.time() - start_time} seconds")
        if save_file is not None:
            with open(save_file, 'w') as f:
                json.dump(self.graph, f)
        graph = copy.deepcopy(self.graph)
        self.graph = {}
        
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
    
    def generate_random_path(self, path_len=25):
        path = None
        i = 0
        while path is None and i < 500:
            start = self.sample_random_vertex()
            path = self.dfs_path(start, path_len)
            i += 1
        return path
    
    def generate_random_query(self, k=5, return_path=False):
        path_len = random.randint(1, k)
        path = self.generate_random_path(path_len)
        if path is None:
            return None
        # print(f"Query Path: {str(path[start_idx:path.index(end)+1])}")
        start = path[0]
        end = path[-1]
        if return_path:
            return self.generate_query_for_vertices(start, end, k, path), start, end, path
        return self.generate_query_for_vertices(start, end, k, path), start, end, None
    
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

def docs_to_contriever_docs(docs_dir, tsv_filename):
    if not os.path.exists(tsv_filename):
        with open(tsv_filename, 'w') as tsv_file:
            tsv_file.write("id\ttext\ttitle\n")
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap  = 50
    )
    i = 0
    to_write = ""
    for file in os.listdir(docs_dir):
        if file.endswith(".txt"):
            print(file)
            with open(f"{docs_dir}/{file}", 'r') as f:
                content = f.read()
            chunks = content.split("\n\n")
            # lines = chunks[0].split("\n")
            # title = lines[0]
            for chunk in chunks:
                chunk = chunk.strip()
                chunk = chunk.replace("\n", ";")
                chunk = chunk.replace("\t", " ")
                title = file[:-4]
                title.replace('_', ' ')
                to_write += f"{str(i)}\t{chunk}\t{title}\n"
                i += 1
    with open(tsv_filename, 'a') as tsv_file:
        tsv_file.write(to_write)
    
def load_graph(file_name):
    with open(file_name, 'r') as f:
        graph = json.load(f)
    return graph

class MultiRAGWiki():
    def __init__(self, model_name=None, documents_dir=None, pipeline_type='text-generation', model_temp=0.7, cuda_device=1, openai=False) -> None:
        self.model_name = model_name
        self.documents_dir = documents_dir
        self.doc_db = None
        self.llm = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=40)
        self.tokenizer = None
        self.pipeline_type = pipeline_type
        self.model_temp = model_temp
        self.tools = None
        self.agent = None
        self.cuda_device = cuda_device
        self.embeddings = None
        self.retrieval_qa = None
        self.openai = openai
        if self.openai == False:
            assert model_name is not None
        self.initialize_llm()
        self.initialize_retriever()
    def initialize_llm(self):
        if not self.openai:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding=True, truncation=True, max_length=512)
            # Define a question-answering pipeline using the model and tokenizer
            question_answerer = pipeline(
                self.pipeline_type, 
                model=self.model_name, 
                tokenizer=self.tokenizer,
                device=self.cuda_device,
                torch_dtype=torch.float16,
            )
            self.llm = HuggingFacePipeline(
                pipeline=question_answerer,
                model_kwargs={"temperature": self.model_temp, "max_length": 512},
            )
        else:
            self.llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")
    def initialize_retriever(self):
        self.tools = None
        self.retrieval_qa = None
        self.doc_db = None
        print('initializing retriever')
        self.retrieval_qa = DocstoreExplorer(Wikipedia())
        self.tools = [Tool(
        name="Search",
        func=self.retrieval_qa.search,
        description="useful for when you need to ask with search",
    ),
    Tool(
        name="Lookup",
        func=self.retrieval_qa.lookup,
        description="useful for when you need to ask with lookup",
    ),]
        # self.tools = []
        self.agent = initialize_agent(self.tools, self.llm, agent=AgentType.REACT_DOCSTORE, verbose=True, handle_parsing_errors=True)
    
    def run(self, query):
        return self.agent.run(query)

class MultiRAGOneStore():
    def __init__(self, model_name=None, documents_dir=None, pipeline_type='text-generation', model_temp=0.7, cuda_device=1, openai=False, llm=None) -> None:
        self.model_name = model_name
        self.documents_dir = documents_dir
        self.doc_db = None
        self.llm = llm
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=40)
        self.tokenizer = None
        self.pipeline_type = pipeline_type
        self.model_temp = model_temp
        self.tools = None
        self.agent = None
        self.cuda_device = cuda_device
        self.embeddings = None
        self.retrieval_qa = None
        self.openai = openai
        if self.openai == False and self.llm is None:
            assert(self.model_name is not None)
        assert(self.documents_dir is not None)
        if self.llm is None:
            self.initialize_llm()
        self.initialize_retriever()
    def initialize_llm(self):
        if not self.openai:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding=True, truncation=True, max_length=1024)
            # Define a question-answering pipeline using the model and tokenizer
            question_answerer = pipeline(
                self.pipeline_type, 
                model=self.model_name, 
                tokenizer=self.tokenizer,
                device=self.cuda_device,
                torch_dtype=torch.float16,
            )

            self.llm = HuggingFacePipeline(
                pipeline=question_answerer,
                model_kwargs={"do_sample": False, "max_length": 512},
            )
        else:
            self.llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")
    def initialize_retriever(self):
        self.tools = None
        self.retrieval_qa = None
        self.doc_db = None
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cuda'})
        print('initializing retriever')
        all_texts = []
        for file in os.listdir(self.documents_dir):
            doc_path = os.path.join(self.documents_dir, file)
            loader = TextLoader(doc_path)
            doc = loader.load()
            texts = self.text_splitter.split_documents(doc)
            all_texts.extend(texts)
        self.doc_db = Chroma(persist_directory='chroma_imddbtiny', embedding_function=self.embeddings, collection_name='all')
        #to handle Chroma memory issues
        if len(all_texts) > 41665:
            for chunk in chunk_list(all_texts, 40000):
                self.doc_db.add_documents(chunk)
        else:
            self.doc_db.add_documents(all_texts)
        # self.doc_db = Chroma.from_documents(all_texts, self.embeddings, collection_name='all')
        
        self.retrieval_qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type='stuff', retriever=self.doc_db.as_retriever())
        self.tools = [Tool(
                name="Intermediate Answer",
                func=self.retrieval_qa.run,
                description=f"useful for when you need to answer questions about anything",
            )]
        # self.tools = []
        self.agent = initialize_agent(self.tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)
    
    def run(self, query):
        return self.agent.run(query)

class MultiRAGMultiStore():
    def __init__(self, model_name=None, documents_dir=None, pipeline_type='text-generation', model_temp=0.7, cuda_device=1, openai=False, llm=None) -> None:
        self.model_name = model_name
        self.documents_dir = documents_dir
        self.doc_dbs = []
        self.files = []
        self.llm = llm
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=40)
        self.tokenizer = None
        self.pipeline_type = pipeline_type
        self.openai = openai
        self.model_temp = model_temp
        self.retrieval_qas = []
        self.tools = []
        self.agent = None
        self.cuda_device = cuda_device
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cuda'})
        if self.openai == False and self.llm is None:
            assert(self.model_name is not None)
        assert(self.documents_dir is not None)
        if self.llm is None:
            self.initialize_llm()
        self.initialize_retriever()
    def initialize_llm(self):
        if not self.openai:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding=True, truncation=True, max_length=512)
            # Define a question-answering pipeline using the model and tokenizer
            question_answerer = pipeline(
                self.pipeline_type, 
                model=self.model_name, 
                tokenizer=self.tokenizer,
                device=self.cuda_device
            )

            self.llm = HuggingFacePipeline(
                pipeline=question_answerer,
                model_kwargs={"temperature": self.model_temp, "max_length": 512},
            )
        else:
            self.llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")
    def initialize_retriever(self):
        self.tools = []
        self.retrieval_qas = []
        self.doc_dbs = []
        self.files = []
        print('initializing retriever')
        for file in os.listdir(self.documents_dir):
            doc_path = os.path.join(self.documents_dir, file)
            loader = TextLoader(doc_path)
            doc = loader.load()
            texts = self.text_splitter.split_documents(doc)
            collection_name = file[:-4]
            collection_name = collection_name.replace(',', '_')
            collection_name = collection_name.replace('.', '_')
            collection_name = collection_name.replace('(', '')
            collection_name = collection_name.replace(')', '')
            if len(collection_name) > 60:
                collection_name = collection_name[:60]
            doc_db = Chroma.from_documents(texts, self.embeddings, collection_name=collection_name)
            self.doc_dbs.append(doc_db)
            self.files.append(file[:-4])
        
        for i in range(len(self.doc_dbs)):
            self.retrieval_qas.append(RetrievalQA.from_chain_type(llm=self.llm, chain_type='stuff', retriever=self.doc_dbs[i].as_retriever()))
            self.files[i] = self.files[i].replace('_', ' ')
            self.tools.append(Tool(
                    name=self.files[i],
                    func=self.retrieval_qas[i].run,
                    description=f"useful for when you need to answer questions about {self.files[i]}",
                ))
        # self.tools = []
        self.agent = initialize_agent(self.tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)
    
    def run(self, query):
        return self.agent.run(query)

def run_query(multi_rag, query):
    limit = 5
    for i in range(5):
        try:
            response = multi_rag.run(query)
            print(response)
            break
        except Exception as e:
            response = str(e)
            if not response.startswith("Could not parse"):
                raise e
            response = response.removeprefix("Could not parse").removesuffix("`")
            response_split = response.split(":")
            response = ' '.join(response_split[1:])
            query = response

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

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:min(i + n, len(lst))]