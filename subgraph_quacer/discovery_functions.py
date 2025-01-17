from subgraph_utils import CustomQueryResult
import random

def get_non_essential_edges(graph, node, essential_rel_types):
    """Get edges with relation types not used in the essential context"""
    edges = []
    if node in graph:
        for target, rel in graph[node].items():
            if rel not in essential_rel_types:
                edges.append((node, target))
    return edges

def get_essential_edges(graph, node, essential_rel_types):
    """Get edges with relation types used in the essential context"""
    edges = []
    distractor_nodes = []
    if node in graph:
        for target, rel in graph[node].items():
            if rel in essential_rel_types:
                edges.append((node, target))
    random.shuffle(edges)
    for parent, child in edges:
        distractor_nodes.append((child, parent)) # for option generation so answer, parent format
    return edges, distractor_nodes

def build_context_edges(graph, essential_edges, context_nodes, distractor_setting=False, max_context_edges=250):
    """Build context edges based on setting"""
    essential_rel_types = set()
    for n1, n2 in essential_edges:
        if n1 in graph and n2 in graph[n1]:
            essential_rel_types.add(graph[n1][n2])
            essential_rel_types.add(graph[n2][n1]) # undirected graph
    
    context_edges = essential_edges.copy()
    nodes_process = list(context_nodes.copy())
    necessary_nodes = set()
    for n1, n2 in essential_edges:
        necessary_nodes.add(n1)
        necessary_nodes.add(n2)
    necessary_nodes = list(necessary_nodes)
    # Add non-essential context based on setting
    distractor_nodes = []
    if distractor_setting:
        # Include all edges for distractor setting
        random.shuffle(necessary_nodes)
        for node in necessary_nodes:
            if len(context_edges) >= max_context_edges:
                break
            essential_edges, dist_nodes = get_essential_edges(graph, node, essential_rel_types)
            max_edge_collect = min(int((max_context_edges - len(context_edges))//(len(necessary_nodes)*2)), len(essential_edges))
            essential_edges = essential_edges[:max_edge_collect]
            dist_nodes = dist_nodes[:max_edge_collect]
            context_edges.extend(essential_edges)
            context_edges = list(set(context_edges))
            distractor_nodes.extend(dist_nodes)
        random.shuffle(nodes_process)
        for node in nodes_process:
            if node in necessary_nodes:
                continue
            if len(context_edges) >= max_context_edges:
                    break
            essential_edges, dist_nodes = get_essential_edges(graph, node, essential_rel_types)
            max_edge_collect = min(int((max_context_edges - len(context_edges))//(len(necessary_nodes)*2)), len(essential_edges))
            essential_edges = essential_edges[:max_edge_collect]
            dist_nodes = dist_nodes[:max_edge_collect]
            context_edges.extend(essential_edges)
            context_edges = list(set(context_edges))
            distractor_nodes.extend(dist_nodes)
        if len(distractor_nodes) < 2:
            print("Distractor nodes less than 2")
            print("Context nodes: ", context_nodes)
            print("Essential edges: ", essential_edges)
            print("Distractor setting: ", distractor_setting)
    random.shuffle(nodes_process)
    for node in nodes_process:
        if len(context_edges) >= max_context_edges:
            break
        non_essential_edges = get_non_essential_edges(graph, node, essential_rel_types)
        non_essential_edges = non_essential_edges[:max_context_edges - len(context_edges)]
        context_edges.extend(non_essential_edges)
        context_edges = list(set(context_edges))
            
    return list(set(context_edges)), list(set(distractor_nodes))  # Remove duplicates

def least_side_effects_discoverer(graph, id2name, **kwargs):
    """Enhanced version with better edge selection"""
    MAX_BUFFER = 5
    found_queries = []
    
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    
    questions = [
        'Which drug used to treat {0} has the least number of side effects?',
        'Which medication has the fewest side effects for {0}?',
        'Suppose person has {0}. Which medication could cause the minimum number of side effects?',
        'What treatment for {0} reports the lowest instances of adverse effects?',
        'Which drug for {0} is associated with the fewest side effects?',
        'Which medication for {0} has the fewest number of documented side effects?'
    ]

    for disease in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        drugs = [node for node, rel in graph[disease].items() if rel == 'RIDR36']
        if len(drugs) <= 1:
            continue
            
        drug_side_effects = {}
        essential_edges = []
        context_nodes = {disease}
        
        for drug in drugs:
            side_effects = [node for node, rel in graph[drug].items() if rel == 'RIDR16']
            drug_side_effects[drug] = side_effects
            context_nodes.add(drug)
            context_nodes.update(side_effects)
        
        min_count = min(len(se) for se in drug_side_effects.values())
        min_drugs = [d for d, se in drug_side_effects.items() if len(se) == min_count]
        
        if len(min_drugs) >= 1:
            chosen_drug = random.choice(min_drugs)
            other_correct = [d for d in min_drugs if d != chosen_drug]
            essential_edges = [(disease, chosen_drug)]
            essential_edges.extend([(chosen_drug, se) for se in drug_side_effects[chosen_drug]])
            
            question = random.choice(questions).format(id2name[disease])
            other_drugs = [d for d, se in drug_side_effects.items() if len(se) > min_count]
            sorted(other_drugs, key=lambda x: len(drug_side_effects[x]))
            for drug in other_drugs[:5]:
                essential_edges.append((disease, drug))
                essential_edges.extend([(drug, se) for se in drug_side_effects[drug]])
                if len(essential_edges) >= 100:
                    break
            found_queries.append({
                'question': question,
                'chosen_answer': chosen_drug,
                'essential_edges': essential_edges,
                'context_nodes': context_nodes,
                'other_correct_answers': other_correct,
                'distractor_rel': 'RIDR36'
            })
    
    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def contraindication_indication_discoverer(graph, id2name, **kwargs):
    MAX_BUFFER = 5
    found_queries = []
    
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    
    questions = [
        'Which drug is a contraindication for {0} and indication for {1}, when patient has both diseases?',
        'For a patient with both diseases {0} and {1}, which medications are contraindicated for {0} but indicated for {1}?',
        'What drugs should be avoided in {0} but are recommended for {1}?',
        'What medication presents a clinical challenge due to its contraindication for {0} and indication for {1} in a patient with both?',
        'Which medication is not recommended for {0} but is recommended for {1} when a patient has both conditions?',
        'Suggest treatments of {1}, which are harmful for {0} for a patient having both {1} and {0}.'
    ]

    for drug in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
        if not id2name[drug].startswith('(drug)'):
            continue
        
        indications = [node for node, rel in graph[drug].items() if rel == 'RIDR6']
        if not indications:
            continue
        contraindicated_diseases = [node for node, rel in graph[drug].items() if rel == 'RIDR5']
        if not contraindicated_diseases:
            continue
        
        disease0 = random.choice(contraindicated_diseases)
        disease1 = random.choice(indications)
        
        question = random.choice(questions).format(id2name[disease0], id2name[disease1])
        
        other_correct = []
        for candidate_drug, rel_id in graph[disease0].items():
            if candidate_drug == drug:
                continue
            if rel_id == 'RIDR35':
                if disease1 in graph[candidate_drug] and graph[candidate_drug][disease1] == 'RIDR6':
                    other_correct.append(candidate_drug)
                    
        essential_edges = [(disease0, drug), (disease1, drug)]
        context_nodes = {disease0, disease1, drug}
        
        found_queries.append({
            'question': question,
            'chosen_answer': drug,
            'essential_edges': essential_edges,
            'context_nodes': context_nodes,
            'other_correct_answers': other_correct
        })

    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def off_label_discoverer(graph, id2name, **kwargs):
    MAX_BUFFER = 5
    found_queries = []
    
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    
    questions = [
        'Which drug indicates {0} and is atleast an off-label use drug for {1}?',
        'Which drug indicates {0} and is an off-label use drug or indication for {1}?',
        'Is there a treatment for {0}, which is also effective as for {1}, atleast as an off-label treatment?',
        'What medication is a treatment for {0} and is also used off-label, or potentially on-label (indicated), for {1}?',
        'Which drug has an on-label indication for {0} but is also employed, at a minimum off-label, for the treatment of {1}?',
        'Name a drug that is officially indicated for {0} and sees at least off-label use for {1}.'
    ]

    for disease0 in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        drugs = [node for node, rel in graph[disease0].items() if rel == 'RIDR36']
        if not drugs:
            continue
            
        drug = random.choice(drugs)
        off_label_diseases = [node for node, rel in graph[drug].items() 
                            if rel in ['RIDR6', 'RIDR7']]
        if not off_label_diseases:
            continue
            
        disease1 = random.choice(off_label_diseases)
        question = random.choice(questions).format(id2name[disease0], id2name[disease1])
        
        other_correct = []
        for candidate_drug in drugs:
            if candidate_drug == drug:
                continue
            if disease1 in graph[candidate_drug] and graph[candidate_drug][disease1] in ['RIDR6', 'RIDR7']:
                if disease0 in graph[candidate_drug] and graph[candidate_drug][disease0] == 'RIDR6':
                    other_correct.append(candidate_drug)
        essential_edges = [(disease0, drug), (disease1, drug)]
        context_nodes = {disease0, disease1, drug}
        
        found_queries.append({
            'question': question,
            'chosen_answer': drug,
            'essential_edges': essential_edges,
            'context_nodes': context_nodes,
            'other_correct_answers': other_correct
        })

    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def dual_indication_discoverer(graph, id2name, **kwargs):
    MAX_BUFFER = 5
    found_queries = []
    
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    
    questions = [
        'Which drug indicates {0} and indicates {1}?',
        'What drug indicates both {0} and {1}?',
        'Which medication is a recognized treatment for both {0} and {1}?',
        'Which drug serves as a treatment option for both {0} and {1} based on its approved uses?',
        'Is there a common treatment for both {0} and {1}?',
        'There is a patient with {0} and {1}. Can you suggest a single treatment officially prescribed to cure both?'
    ]

    for disease0 in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        drugs = [node for node, rel in graph[disease0].items() if rel == 'RIDR36']
        if not drugs:
            continue
            
        drug = random.choice(drugs)
        indica_diseases = [node for node, rel in graph[drug].items() if rel == 'RIDR6']
        if not indica_diseases:
            continue
            
        disease1 = random.choice(indica_diseases)
        question = random.choice(questions).format(id2name[disease0], id2name[disease1])
        
        other_correct = []
        for candidate_drug in drugs:
            if candidate_drug == drug:
                continue
            if disease1 in graph[candidate_drug] and graph[candidate_drug][disease1] == 'RIDR6':
                if disease0 in graph[candidate_drug] and graph[candidate_drug][disease0] == 'RIDR6':
                    other_correct.append(candidate_drug)
        
        essential_edges = [(disease0, drug), (disease1, drug)]
        context_nodes = {disease0, disease1, drug}
        
        found_queries.append({
            'question': question,
            'chosen_answer': drug,
            'essential_edges': essential_edges,
            'context_nodes': context_nodes,
            'other_correct_answers': other_correct
        })

    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def synergistic_discoverer(graph, id2name, **kwargs):
    MAX_BUFFER = 5
    found_queries = []
    
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    
    questions = [
        'Which drug indicates {0} and interacts synergistically with the treatment of {1}?',
        'Which medication benefits both {0} and supports the medicine used in the treatment of {1}?',
        'Name a drug that treats {0} and interacts positively with {1}\'s treatment.',
        'What drug is used for treating {0} and enhances the effects of treatment for {1}?',
        'Can you identify a medication that addresses {0} and boosts therapy outcomes for {1}?',
        'What drug both indicates {0} and enhances {1}\'s treatment efficacy?'
    ]

    for disease0 in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        drugs = [node for node, rel in graph[disease0].items() if rel == 'RIDR36']
        if not drugs:
            continue
            
        drug = random.choice(drugs)
        syn_drugs = [node for node, rel in graph[drug].items() if rel == 'RIDR8']
        if not syn_drugs:
            continue
            
        syn_drug = random.choice(syn_drugs)
        indica_syn_drug = [node for node, rel in graph[syn_drug].items() if rel == 'RIDR6']
        if not indica_syn_drug:
            continue
            
        disease1 = random.choice(indica_syn_drug)
        question = random.choice(questions).format(id2name[disease0], id2name[disease1])
        
        other_correct = []
        for candidate_drug in drugs:
            if candidate_drug == drug:
                continue
            if disease0 in graph[candidate_drug] and graph[candidate_drug][disease0] == 'RIDR6':
                syner_drugs = [node for node, rel in graph[candidate_drug].items() if rel == 'RIDR8']
                for syner_drug in syner_drugs:
                    if disease1 in graph[syner_drug] and graph[syner_drug][disease1] == 'RIDR6':
                        other_correct.append(candidate_drug)
                        break
                    
        essential_edges = [(disease0, drug), (drug, syn_drug), (syn_drug, disease1)]
        context_nodes = {disease0, disease1, drug, syn_drug}
        
        found_queries.append({
            'question': question,
            'chosen_answer': drug,
            'essential_edges': essential_edges,
            'context_nodes': context_nodes,
            'other_correct_answers':other_correct
        })

    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def gene_target_discoverer(graph, id2name, **kwargs):
    MAX_BUFFER = 5
    found_queries = []
    
    disease_nodes = list(graph.keys())
    random.shuffle(disease_nodes)
    
    questions = [
        'Which drug targets {0} associated with {1}?',
        'Identify a drug that interacts with {0} connected to {1}.',
        'Which pharmaceutical targets {0} relevant to {1}?',
        'What drug targets the {0} linked to {1}?',
        'Name a drug that modulates {0} involved in {1}.',
        'Can you identify a drug that focuses on {0} linked with {1}?'
    ]

    for disease0 in disease_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        genes = [node for node, rel in graph[disease0].items() if rel == 'RIDR38']
        if not genes:
            continue
            
        gene = random.choice(genes)
        drugs = [node for node, rel in graph[gene].items() if rel == 'RIDR33']
        if not drugs:
            continue
            
        drug = random.choice(drugs)
        question = random.choice(questions).format(id2name[gene], id2name[disease0])
        
        other_correct = []
        for candidate_drug in drugs:
            if candidate_drug == drug:
                continue
            if gene in graph[candidate_drug] and graph[candidate_drug][gene] == 'RIDR3':
                other_correct.append(candidate_drug)
        
        essential_edges = [(disease0, gene), (gene, drug)]
        context_nodes = {disease0, gene, drug}
        
        found_queries.append({
            'question': question,
            'chosen_answer': drug,
            'essential_edges': essential_edges,
            'context_nodes': context_nodes,
            'other_correct_answers': other_correct
        })
        
    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def phenotype_drug_contraindication_discoverer(graph, id2name, **kwargs):
    """Discovery function for finding diseases sharing phenotypes where one disease's drug contraindicates the other"""
    MAX_BUFFER = 5
    found_queries = []
    
    # Find phenotypes and their associated diseases
    phenotype_nodes = [node for node in graph.keys() 
                      if id2name[node].startswith('(effect/phenotype)')]
    random.shuffle(phenotype_nodes)
    
    questions = [
        'Which disease is contraindicated by drug indication {0} for {1}, which both show {2}?',
        'What condition is deemed incompatible with {0} indicated for {1}, given their overlapping {2}?',
        'Identify the disease contraindicated by the administration of {0} for {1}, having common effect {2}.',
        'Which disease is incompatible with {0} prescribed for {1}, despite the mutual presence of {2}?',
        'What disease is excluded by {0}\'s use in {1}, having shared {2}?',
        'Which illness conflicts with {0}\'s indication for {1}, as both display {2}?'
    ]
    
    for phenotype in phenotype_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        # Get diseases with this phenotype
        diseases = [node for node, rel in graph[phenotype].items() if rel == 'RIDR13']
        if len(diseases) < 2:
            continue
        
        random.shuffle(diseases)
        # Check each pair of diseases
        for i in range(len(diseases)):
            disease1 = diseases[i]
            for j in range(i+1, len(diseases)):
                disease2 = diseases[j]
                
                # Get drugs indicated for disease1
                drugs1 = [node for node, rel in graph[disease1].items() if rel == 'RIDR36']
                random.shuffle(drugs1)
                
                for drug in drugs1:
                    # Check if drug contraindicates disease2
                    contraindications = [node for node, rel in graph[drug].items() 
                                      if rel == 'RIDR5']
                    if disease2 in contraindications:
                        question = random.choice(questions).format(
                            id2name[drug],
                            id2name[disease1],
                            id2name[phenotype]
                        )
                        
                        essential_edges = [
                            (phenotype, disease1),
                            (phenotype, disease2),
                            (disease1, drug),
                            (drug, disease2)
                        ]
                        context_nodes = {phenotype, disease1, disease2, drug}
                        other_correct = []
                        for candidate_disea in contraindications:
                            if candidate_disea == disease2:
                                continue
                            if phenotype in graph[candidate_disea] and graph[candidate_disea][phenotype] == 'RIDR12':
                                other_correct.append(candidate_disea)
                        
                        found_queries.append({
                            'question': question,
                            'chosen_answer': disease2,
                            'essential_edges': essential_edges,
                            'context_nodes': context_nodes,
                            'other_correct_answers': other_correct
                        })
                        break  # Found a valid query for this disease pair
                
                if found_queries:  # If we found a query, break the inner loop
                    break
            if found_queries:  # If we found a query, break the outer loop
                break
                    
    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def drug_contraindication_discoverer(graph, id2name, **kwargs):
    """Discovery function for finding diseases that are treated by one drug but contraindicated by another"""
    MAX_BUFFER = 5
    found_queries = []
    
    drug_nodes = [node for node in graph.keys() 
                  if id2name[node].startswith('(drug)')]
    random.shuffle(drug_nodes)
    
    questions = [
        'Which disease is treated with {0} but contraindicated with {1}?',
        'Which disease is indicated by {0} but not treated by {1}?',
        'What disease is managed with {0} but cannot be treated with {1}?',
        'Identify the disease that is treated by {0} but conflicts with {1}.',
        'What illness is treated with {0} but is incompatible with {1}?',
        'Which disease benefits from {0} but is contraindicated when using {1}?'
    ]
    
    for drug1 in drug_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        # Get diseases indicated by drug1
        indications = [node for node, rel in graph[drug1].items() if rel == 'RIDR6']
        if not indications:
            continue
        
        random.shuffle(indications)
        
        # Get other drugs that contraindicate these diseases
        for disease in indications:
            contraindicated_by = []
            for drug2, drug2_data in graph.items():
                if drug2 == drug1:
                    continue
                if disease in [node for node, rel in drug2_data.items() if rel == 'RIDR5']:
                    contraindicated_by.append(drug2)
                    
            if contraindicated_by:
                drug2 = random.choice(contraindicated_by)
                question = random.choice(questions).format(
                    id2name[drug1],
                    id2name[drug2]
                )
                
                essential_edges = [
                    (drug1, disease),
                    (drug2, disease)
                ]
                context_nodes = {drug1, drug2, disease}
                
                other_correct = []
                for candidate_disea in indications:
                    if candidate_disea == disease:
                        continue
                    if drug2 in graph[candidate_disea] and graph[candidate_disea][drug2] == 'RIDR35':
                        other_correct.append(candidate_disea)
                        
                found_queries.append({
                    'question': question,
                    'chosen_answer': disease,
                    'essential_edges': essential_edges,
                    'context_nodes': context_nodes,
                    'other_correct_answers': other_correct
                })
                
    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def exposure_drug_discoverer(graph, id2name, **kwargs):
    """Find drugs that treat diseases caused by specific exposures"""
    MAX_BUFFER = 5
    found_queries = []
    
    questions = [
        'Which drug can be used to treat a disease caused by exposure of {0}?',
        'What medication is effective for treating a disease resulting from exposure to {0}?',
        'What drug can manage a condition originating from {0}?',
        'Which drug is indicated for illnesses caused by {0}?',
        'Which pharmaceutical is prescribed for conditions caused by {0}?',
        'Name a drug that treats diseases associated with {0}.'
    ]

    exposure_nodes = [node for node in graph.keys() 
                     if id2name[node].startswith('(exposure)')]
    random.shuffle(exposure_nodes)
    
    for exposure in exposure_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        # Get diseases caused by this exposure
        exposure_diseases = [node for node, rel in graph[exposure].items() 
                           if rel == 'RIDR24']
        
        for disease in exposure_diseases:
            # Get drugs that treat these diseases
            drugs = [node for node, rel in graph[disease].items() 
                    if rel == 'RIDR36']
            
            if drugs:
                drug = random.choice(drugs)
                question = random.choice(questions).format(id2name[exposure])
                
                essential_edges = [
                    (exposure, disease),
                    (disease, drug)
                ]
                context_nodes = {exposure, disease, drug}
                
                other_correct = []
                for new_drug in drugs:
                    if new_drug == drug:
                        continue
                    for pos_disea, rel_id in graph[new_drug].items():
                        if rel_id == 'RIDR6':
                            if exposure in graph[pos_disea] and graph[pos_disea][exposure] == 'RIDR42':
                                other_correct.append(new_drug)
                                break
                
                found_queries.append({
                    'question': question,
                    'chosen_answer': drug,
                    'essential_edges': essential_edges,
                    'context_nodes': context_nodes,
                    'other_correct_answers': other_correct
                })
                break
    
    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def enzyme_drug_disease_discoverer(graph, id2name, **kwargs):
    """Find diseases treated by drugs that are catalyzed by specific enzymes"""
    MAX_BUFFER = 5
    found_queries = []
    
    questions = [
        'Which disease is treated by a drug that is catalyzed by the {0}?',
        'What disease is managed by a drug activated by {0}?',
        'Identify the disease treated with a medication processed by {0}.',
        'Name the disease treated by a medication catalyzed through {0}.',
        'What disease is treated by a medication dependent on {0} activity?',
        'What illness is addressed by a drug whose activation depends on {0}?'
    ]

    enzyme_nodes = [node for node in graph.keys() 
                   if id2name[node].startswith('(gene/enzyme)')]
    random.shuffle(enzyme_nodes)
    
    for enzyme in enzyme_nodes:
        if len(found_queries) >= MAX_BUFFER:
            break
            
        # Find drugs catalyzed by this enzyme
        catalyzed_drugs = []
        for drug, drug_data in graph.items():
            if not id2name[drug].startswith('(drug)'):
                continue
                
            if enzyme in [node for node, rel in drug_data.items() 
                         if rel in ['RIDR2', 'RIDR3', 'RIDR4']]:
                # Check if drug treats any diseases
                diseases = [node for node, rel in drug_data.items() 
                          if rel == 'RIDR6']
                if diseases:
                    catalyzed_drugs.append((drug, diseases))
        
        if catalyzed_drugs:
            drug, diseases = random.choice(catalyzed_drugs)
            disease = random.choice(diseases)
            question = random.choice(questions).format(id2name[enzyme])
            
            essential_edges = [
                (enzyme, drug),
                (drug, disease)
            ]
            context_nodes = {enzyme, drug, disease}
            
            other_correct = []
            for candidate_drug, candidate_diseases in catalyzed_drugs:
                if candidate_drug == drug:
                    continue
                if disease in candidate_diseases:
                    other_correct.append(candidate_drug)
                    
            found_queries.append({
                'question': question,
                'chosen_answer': disease,
                'essential_edges': essential_edges,
                'context_nodes': context_nodes,
                'other_correct_answers': other_correct
            })
    
    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None

def phenotype_group_disease_discoverer(graph, id2name, **kwargs):
    """
    Find diseases that uniquely match a set of phenotypes, but now we also keep track
    of other diseases if the group somehow matches more than one. 
    (Though 'unique' means probably only one disease is truly correct.)
    """
    MAX_BUFFER = 5
    found_queries = []
    
    questions = [
        'Which disease can be a diagnosis for the following phenotypes - {0}?'
    ]
    
    # Build disease-phenotype mapping
    disease_phenotypes = {}
    for phenotype, phenotype_data in graph.items():
        if not id2name[phenotype].startswith('(effect/phenotype)'):
            continue
            
        diseases = [node for node, rel in phenotype_data.items() if rel == 'RIDR13']
        for disease in diseases:
            if disease not in disease_phenotypes:
                disease_phenotypes[disease] = set()
            disease_phenotypes[disease].add(phenotype)
    
    # Group diseases by their phenotype sets
    phenotype_groups = {}
    for disease, phenotypes in disease_phenotypes.items():
        phenotype_tuple = tuple(sorted(phenotypes))
        phenotype_groups.setdefault(phenotype_tuple, []).append(disease)
    
    # We want groups that identify exactly one disease, but let's see if any group identifies multiple
    # If the group has multiple diseases, they're all correct answers (?). 
    # The question says "Which disease can be a diagnosis for the following phenotypes..."
    # We'll allow multiple diseases, storing them in other_correct_answers.
    random_groups = list(phenotype_groups.items())
    random.shuffle(random_groups)
    
    for phenos, diseases in random_groups:
        if len(found_queries) >= MAX_BUFFER:
            break
        
        if not diseases:
            continue
        # pick one disease
        chosen_disease = random.choice(diseases)
        other_diseases = [d for d in diseases if d != chosen_disease]
        
        # Format phenotype list nicely
        if len(phenos) == 1:
            phenotype_list = id2name[list(phenos)[0]]
        else:
            pheno_names = [id2name[p] for p in phenos]
            if len(pheno_names) > 1:
                phenotype_list = ', '.join(pheno_names[:-1]) + ' and ' + pheno_names[-1]
            else:
                phenotype_list = pheno_names[0]
        
        question = random.choice(questions).format(phenotype_list)
        
        essential_edges = [(p, chosen_disease) for p in phenos]
        context_nodes = set(phenos) | {chosen_disease}
        
        found_queries.append({
            'question': question,
            'chosen_answer': chosen_disease,
            'essential_edges': essential_edges,
            'context_nodes': context_nodes,
            'other_correct_answers': other_diseases
        })
    
    if found_queries:
        chosen_query = random.choice(found_queries)
        context_edges, distractor_nodes = build_context_edges(
            graph,
            chosen_query['essential_edges'],
            chosen_query['context_nodes'],
            kwargs.get('distractor_setting', False)
        )
        
        return CustomQueryResult(
            chosen_query['question'],
            chosen_query['chosen_answer'],
            chosen_query['essential_edges'],
            context_edges,
            other_correct_answers=chosen_query['other_correct_answers'],
            distractor_nodes=distractor_nodes
        )
    return None