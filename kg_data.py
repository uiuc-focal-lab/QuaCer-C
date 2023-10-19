import requests
from concurrent.futures import ThreadPoolExecutor
import time
import queue
import concurrent
import threading
import random

from collections import deque

def bfs_shortest_path(graph, start, end):
    visited = set()
    queue = deque([(start, [start])])
    while queue:
        node, path = queue.popleft()
        if node == end:
            return len(path) - 1  # return the length of the path
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, {}):  # use dict.get to handle missing keys
                queue.append((neighbor, path + [neighbor]))

def graph_diameter(graph):
    max_diameter = 0
    all_nodes = set(graph.keys())  # nodes with outgoing edges
    for node in graph.values():  # add nodes with incoming edges
        all_nodes.update(node.keys())
    all_nodes_list = list(all_nodes)
    all_nodes = set(random.sample(all_nodes_list, min(100, len(all_nodes_list))))
    for node in all_nodes:
        for target_node in all_nodes:
            if node != target_node:
                path_length = bfs_shortest_path(graph, node, target_node)
                if path_length is not None:
                    max_diameter = max(max_diameter, path_length)
    return max_diameter

REQUEST_TIMEOUT = 5
MIN_DEPTH = 100
entity_min_depth = ''
max_threads = 20
executor = ThreadPoolExecutor(max_workers=max_threads)
job_queue = queue.Queue()
futures = []  # Define the futures list to keep track of submitted tasks
thread_lock = threading.Lock()
visited = set()
def exploreWikidata(entity, graph, depth=3, item_labels=None, property_labels=None):
    global MIN_DEPTH, entity_min_depth, job_queue, futures, executor, visited
    try:
        if depth == 0 or visited is None or item_labels is None or property_labels is None:
            MIN_DEPTH = depth
            entity_min_depth = entity
            return
        time.sleep(1.0)
        relations = queryWikidataForRelations(entity)
        #print(f"Exploring {entity} at depth {depth} and length of relations {len(relations)}")
        for relatedEntity, propertyRelation, itemLabel, propertyLabel in relations:
            item_labels[relatedEntity] = itemLabel
            property_labels[propertyRelation] = propertyLabel
            
            #print(f"Entity {entity} ({item_labels.get(entity)}) has relation {property_labels.get(propertyRelation)} with {relatedEntity} ({itemLabel})")
            if entity not in graph:
                graph[entity] = {}
            graph[entity][relatedEntity] = propertyRelation
            if depth < MIN_DEPTH:
                MIN_DEPTH = depth
                entity_min_depth = entity
            if relatedEntity not in visited:
                with thread_lock:
                    visited.add(relatedEntity)
                job_queue.put((relatedEntity, graph, depth - 1, item_labels, property_labels))
            #print(f"Job queue1 size: {job_queue.qsize()}")
    except Exception as e:
        print(f"Exception: {e}")
        raise(e)

def queryWikidataForRelations(entity):
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
    """

    url = 'https://dbpedia.org/sparql'
    try:
        data = requests.get(url, params={'query': sparql, 'format': 'json'}, timeout=REQUEST_TIMEOUT).json()
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
        return list(random.sample(relatedEntities, min(3, len(relatedEntities))))

    except Exception as e:
        print(f"Query for entity {entity} timed out." + str(e)[:200])
        return []
    
def DFS(entity, graph, visited, current_path, all_paths, item_labels, property_labels):
    visited.add(entity)
    is_leaf = True

    for next_entity, property_id in graph.get(entity, {}).items():
        if next_entity not in visited:
            is_leaf = False
            current_relation = (entity, property_id, next_entity, item_labels[next_entity])
            current_path.append(current_relation)
            DFS(next_entity, graph, visited, current_path, all_paths, item_labels, property_labels)
            current_path.pop()  # Backtrack

    if is_leaf:
        all_paths.append(list(current_path))

def main():
    global MIN_DEPTH, entity_min_depth, job_queue, futures, executor, visited
    num_depth = 10
    start_time = time.time()
    print()
    startEntity = "Albert_Einstein"  # Starting from the entity representing a cat
    visited = set([startEntity])  # Dictionary to store entity labels (for more readable printout)
    graph = {}
    item_labels = {startEntity: "Albert Einstein"}
    property_labels = {}

    job_queue.put((startEntity, graph, num_depth, item_labels, property_labels))
    
    while True:
        if job_queue.empty() and all(f.done() for f in futures):
                break  # Exit the loop if the job queue is empty and all futures are done
        if not job_queue.empty() or [f for f in futures if not f.done()]:
            incomplete_futures = [f for f in futures if not f.done()]
            if len(incomplete_futures) < max_threads and not job_queue.empty():
                args = job_queue.get()
                futures.append(executor.submit(exploreWikidata, *args))
    print(f"job queue size: {job_queue.qsize()}")
    #print(f"Number of futures: {len(futures)}", futures)
    executor.shutdown(wait=True)
    #print(f"Graph: {graph}")
    print(f"Num Visited entities: {len(visited)}")
    print(f"Graph diameter: {graph_diameter(graph)}")
    #print(f"Item labels: {item_labels}")
    #print(f"Property labels: {property_labels}")
    print(f"Minimum depth: {MIN_DEPTH}")
    print(f"Entity at minimum depth: {entity_min_depth}")

    path = []
    vis = set()
    all_paths = []
    DFS(startEntity, graph, vis, path, all_paths, item_labels, property_labels)
    #print("All paths: ")
    #print(all_paths)
    maxi = max(all_paths, key=len)
    print(maxi)
    end_tiem = time.time()
    print(f"Time taken: {end_tiem - start_time}")

main()