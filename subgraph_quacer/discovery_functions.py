from subgraph_utils import CustomQueryResult
import random

def least_side_effects_discoverer(graph, id2name):
    """Enhanced version with better edge selection"""
    MAX_BUFFER = 5
    found_queries = []
    
    # Random disease sampling
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    
    for disease in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        disease_data = graph[disease]
        drugs = [node for node, rel in disease_data.items() if rel == 'RIDR36']
        
        if len(drugs) > 1:
            drug_side_effects = {}
            essential_edges = []
            context_edges = []
            
            for drug in drugs:
                edge = (disease, drug)
                context_edges.append(edge)
            
            for drug in drugs:
                side_effects = [node for node, rel in graph[drug].items() 
                              if rel == 'RIDR16']
                drug_side_effects[drug] = side_effects
                
                for se in side_effects:
                    edge = (drug, se)
                    context_edges.append(edge)
            
            min_count = min(len(se) for se in drug_side_effects.values())
            min_drugs = [d for d, se in drug_side_effects.items() 
                        if len(se) == min_count]
            
            if len(min_drugs) == 1:
                chosen_drug = min_drugs[0]
                
                essential_edges = [(disease, chosen_drug)]
                essential_edges.extend([
                    (chosen_drug, se) 
                    for se in drug_side_effects[chosen_drug]
                ])
                
                question = f"Which drug used to treat {id2name[disease]} has the least number of side effects?"
                
                found_queries.append({
                    'question': question,
                    'chosen_answer': chosen_drug,
                    'essential_edges': essential_edges,
                    'context_edges': context_edges
                })
    
    if found_queries:
        chosen_query = random.choice(found_queries)
        all_context_edges = []
        for query in found_queries:
            all_context_edges.extend(query['context_edges'])
            
        all_context_edges = list(dict.fromkeys(all_context_edges)) #remove duplicates
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            all_context_edges
        )
    return None