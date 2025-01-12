from subgraph_utils import CustomQueryResult
import random

def driver(graph,id2name,cert_num = 1):
    # driver function to call the specific template
    cert_func = {1:discoverer_1,2:discoverer_2,3:discoverer_3,4:discoverer_4,5:discoverer_5,6:discoverer_6}
    found_queries = cert_func[cert_num](graph,id2name,MAX_BUFFER=5)
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

def discoverer_1(graph, id2name,MAX_BUFFER = 5):
    """Enhanced version with better edge selection"""
    found_queries = []
    
    # Random disease sampling
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    questions_aliases = ['Which drug used to treat {0} has the least number of side effects?',
                         'Which medication has the fewest side effects for {0}?',
'Suppose person has {0}. Which medication could cause the minimum number of side effects?',
'What treatment for {0} reports the lowest instances of adverse effects?',
'Which drug for {0} is associated with the fewest side effects?',
'Which medication for {0} has the fewest number of documented side effects?'
]
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
                
                question = random.choice(questions_aliases).format(id2name[disease])
                
                found_queries.append({
                    'question': question,
                    'chosen_answer': chosen_drug,
                    'essential_edges': essential_edges,
                    'context_edges': context_edges
                })
    return found_queries

def discoverer_2(graph, id2name,MAX_BUFFER = 5):
    
    found_queries = []
    
    # Random disease sampling
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    questions_aliases = ['Which drug is a contraindication for {0} and indication for {1}, when patient has both diseases?',
                         'For a patient with both diseases {0} and {1}, which medications are contraindicated for {0} but indicated for {1}?',
'What drugs should be avoided in {0} but are recommended for {1}?',
'What medication presents a clinical challenge due to its contraindication for {0} and indication for {1} in a patient with both?',
'Which medication is not recommended for {0} but is recommended for {1} when a patient has both conditions?',
'Suggest treatments of {1}, which are harmful for {0} for a patient having both {1} and {0}.'
]
    for disease1 in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        disease_data = graph[disease1]
        drugs = [node for node, rel in disease_data.items() if rel == 'RIDR36']
        if len(drugs) == 0:
            continue
        drug=random.choice(drugs)
        contra_diseases = [node for node,rel in graph[drug].items() if rel == 'RIDR5']
        if len(contra_diseases) == 0:
            continue
        disease0 = random.choice(contra_diseases)
        question = random.choice(questions_aliases).format(id2name[disease0], id2name[disease1])
        answer = drug
        essential_edges = [(disease0,drug), (disease1,drug)]
        
        found_queries.append({
            'question': question,
            'chosen_answer': answer,
            'essential_edges': essential_edges,
            'context_edges': essential_edges
        })
    
    return found_queries

def discoverer_3(graph, id2name,MAX_BUFFER = 5):
    
    found_queries = []
    
    # Random disease sampling
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    questions_aliases = ['Which drug indicates {0} and is atleast an off-label use drug for {1}?',
                         'Which drug indicates {0} and is an off-label use drug or indication for {1}?',
'Is there a treatment for {0}, which is also effective as for {1}, atleast as an off-label treatment?',
'What medication is a treatment for {0} and is also used off-label, or potentially on-label (indicated), for {1}?',
'Which drug has an on-label indication for {0} but is also employed, at a minimum off-label, for the treatment of {1}?',
'Name a drug that is officially indicated for {0} and sees at least off-label use for {1}.'
]
    for disease0 in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
        disease_data = graph[disease0]
        drugs = [node for node, rel in disease_data.items() if rel == 'RIDR36']
        if len(drugs) == 0:
            continue
        drug=random.choice(drugs)
        atlease_off_label_diseases = [node for node,rel in graph[drug].items() if rel == 'RIDR6' or rel == 'RIDR7']
        if len(atlease_off_label_diseases) == 0:
            continue
        disease1 = random.choice(atlease_off_label_diseases)
        question = random.choice(questions_aliases).format(id2name[disease0], id2name[disease1])
        answer = drug
        essential_edges = [(disease0,drug), (disease1,drug)]
        
        found_queries.append({
            'question': question,
            'chosen_answer': answer,
            'essential_edges': essential_edges,
            'context_edges': essential_edges
        })
    
    return found_queries

def discoverer_4(graph, id2name,MAX_BUFFER = 5):
    
    found_queries = []
    
    # Random disease sampling
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    questions_aliases = ['Which drug indicates {0} and indicates {1}?',
                         'What drug indicates both {0} and {1}?',
'Which medication is a recognized treatment for both {0} and {1}?',
'Which drug serves as a treatment option for both {0} and {1} based on its approved uses?',
'Is there a common treatment for both {0} and {1}?',
'There is a patient with {0} and {1}. Can you suggest a single treatment officially prescribed to cure both?'
]
    for disease0 in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        disease_data = graph[disease0]
        drugs = [node for node, rel in disease_data.items() if rel == 'RIDR36']
        if len(drugs) == 0:
            continue
        drug=random.choice(drugs)
        indica_diseases = [node for node,rel in graph[drug].items() if rel == 'RIDR6']
        if len(indica_diseases) == 0:
            continue
        disease1 = random.choice(indica_diseases)
        question = random.choice(questions_aliases).format(id2name[disease0], id2name[disease1])
        answer = drug
        essential_edges = [(disease0,drug), (disease1,drug)]
        
        found_queries.append({
            'question': question,
            'chosen_answer': answer,
            'essential_edges': essential_edges,
            'context_edges': essential_edges
        })
    
    return found_queries

def discoverer_5(graph, id2name,MAX_BUFFER = 5):
    
    found_queries = []
    
    # Random disease sampling
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    questions_aliases = ['Which drug indicates {0} and interacts synergistically with the treatment of {1}?',
                         'Which medication benefits both {0} and supports the medicine used in the treatment of {1}?',
'Name a drug that treats {0} and interacts positively with {1}\'s treatment.',
'What drug is used for treating {0} and enhances the effects of treatment for {1}?',
'Can you identify a medication that addresses {0} and boosts therapy outcomes for {1}?',
'What drug both indicates {0} and enhances {1}\'s treatment efficacy?'
]
    for disease0 in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        disease_data = graph[disease0]
        drugs = [node for node, rel in disease_data.items() if rel == 'RIDR36']
        if len(drugs) == 0:
            continue
        drug = random.choice(drugs)
        syn_drugs = [node for node,rel in graph[drug].items() if rel == 'RIDR8']
        if len(syn_drugs) == 0:
            continue
        syn_drug = random.choice(syn_drugs)
        indica_syn_drug = [node for node,rel in graph[syn_drug].items() if rel == 'RIDR6']
        if len(indica_syn_drug) == 0:
            continue
        disease1 = random.choice(indica_syn_drug)
        question = random.choice(questions_aliases).format(id2name[disease0], id2name[disease1])
        answer = drug
        essential_edges = [(disease0,drug), (drug,syn_drug), (syn_drug, disease1)]
        
        found_queries.append({
            'question': question,
            'chosen_answer': answer,
            'essential_edges': essential_edges,
            'context_edges': essential_edges
        })
    
    return found_queries

def discoverer_6(graph, id2name,MAX_BUFFER = 5):
    
    found_queries = []
    
    # Random disease sampling
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    questions_aliases = ['Which drug targets {0} associated with {1}?',
                         'Identify a drug that interacts with {0} connected to {1}.',
'Which pharmaceutical targets {0} relevant to {1}?',
'What drug targets the {0} linked to {1}?',
'Name a drug that modulates {0} involved in {1}.',
'Can you identify a drug that focuses on {0} linked with {1}?'
]
    for disease0 in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        disease_data = graph[disease0]
        genes = [node1 for node1, rel in disease_data.items() if rel == 'RIDR38']
        if len(genes) > 0:
            gene = random.choice(genes)
        else:
            continue
        drugs = [node1 for node1, rel in graph[gene].items() if rel=='RIDR33']
        
        if len(drugs) > 0:
            drug = random.choice(drugs)
        else:
            continue
        question = random.choice(questions_aliases).format(id2name[gene], id2name[disease0])
        answer = drug
        essential_edges = [(disease0,gene), (gene,drug)]
            
        found_queries.append({
            'question': question,
            'chosen_answer': answer,
            'essential_edges': essential_edges,
            'context_edges': essential_edges
        })
    
    return found_queries