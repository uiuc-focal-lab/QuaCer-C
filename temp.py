#DBPedia
import requests
from concurrent.futures import ThreadPoolExecutor
import time

REQUEST_TIMEOUT = 5
MIN_DEPTH = 100
entity_min_depth = ''
def exploreWikidata(entity, graph, depth=3, visited=None, item_labels=None, property_labels=None):
    global MIN_DEPTH, entity_min_depth
    if depth == 0 or visited is None or item_labels is None or property_labels is None:
        MIN_DEPTH = depth
        entity_min_depth = entity
        return

    relations = queryWikidataForRelations(entity)
    visited.add(entity)

    # Use ThreadPoolExecutor for parallel processing
    time.sleep(1.0)  # To avoid hitting the server too frequently
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for relatedEntity, propertyRelation, itemLabel, propertyLabel in relations:
            item_labels[relatedEntity] = itemLabel
            property_labels[propertyRelation] = propertyLabel
            
            print(f"Entity {entity} ({item_labels.get(entity)}) has relation {property_labels.get(propertyRelation)} with {relatedEntity} ({itemLabel})")
            if entity not in graph:
                graph[entity] = {}
            graph[entity][relatedEntity] = propertyRelation
            if depth < MIN_DEPTH:
                MIN_DEPTH = depth
                entity_min_depth = entity
            futures.append(executor.submit(exploreWikidata, relatedEntity, graph, depth - 1, visited, item_labels, property_labels))
    #         exploreWikidata(relatedEntity, graph, depth - 1, visited, item_labels, property_labels)
    # for relatedEntity, propertyRelation, itemLabel, propertyLabel in relations:
    #     item_labels[relatedEntity] = itemLabel
    #     property_labels[propertyRelation] = propertyLabel
        
    #     #print(f"Entity {entity} ({item_labels.get(entity)}) has relation {property_labels.get(propertyRelation)} with {relatedEntity} ({itemLabel})")
    #     if entity not in graph:
    #         graph[entity] = {}
    #     graph[entity][relatedEntity] = propertyRelation
    #     if depth < MIN_DEPTH:
    #         MIN_DEPTH = depth
    #         entity_min_depth = entity
    #     futures.append(executor.submit(exploreWikidata, relatedEntity, graph, depth - 1, visited, item_labels, property_labels))
    #     #exploreWikidata(relatedEntity, graph, depth - 1, visited, item_labels, property_labels)
    #     #Wait for all threads to complete
        for future in futures:
            future.result()
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
    }}
    LIMIT 5
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
        time.sleep(1.0)  # To avoid hitting the server too frequently
        return relatedEntities

    except Exception as e:
        print(f"Query for entity {entity} timed out." + str(e)[:200])
        return []

startEntity = "Albert_Einstein"  # Starting from the entity representing a cat
visited_entities = set([startEntity])  # Dictionary to store entity labels (for more readable printout)
graph = {}
item_labels = {startEntity: "Albert Einstein"}
property_labels = {}
exploreWikidata(startEntity, graph, 1, visited_entities, item_labels, property_labels)