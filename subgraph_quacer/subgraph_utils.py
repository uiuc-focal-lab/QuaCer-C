from utils import *
import numpy as np
import os
import json

class CustomQueryResult:
    def __init__(self, question, chosen_answer, essential_context_edges, all_context_edges, other_correct_answers=None, distractor_nodes=[]):
        """
        Args:
            question (str): The natural language question
            chosen_answer (str): The ID of the selected correct answer node
            context_edges (list): List of tuples (node1_id, node2_id) specifying relevant edges
            other_correct_answers (list, optional): List of other valid answer node IDs
        """
        self.question = question
        self.chosen_answer = chosen_answer
        self.context_edges = all_context_edges
        self.essential_context_edges = essential_context_edges
        for edge in self.essential_context_edges:
            if edge not in self.context_edges:
                self.context_edges.append(edge)
                
        self.distractor_nodes = distractor_nodes
        self.other_correct_answers = other_correct_answers if other_correct_answers else []

def get_entity_type(entity, id2name):
    entity_type = id2name[entity]
    idx = entity_type.index(')')
    entity_type = entity_type[:idx+1]
    return entity_type

def generate_options_with_exclusions(correct_answer, excluded_answers, context_nodes, 
                                   random_entities, graph, answer_parent, essential_edge_rels, 
                                   id2name=None, min_num_options=4, distractor_query=False, distractor_nodes=[]):
    """
    Generate answer options while excluding certain nodes.
    
    Args:
        correct_answer: The chosen correct answer node ID
        excluded_answers: List of node IDs to exclude from options
        context_nodes: List of relevant context node IDs
        random_entities: List of potential random entities
        graph: The graph structure
        min_num_options: Minimum number of options to generate, this is treated as a hard constraint and not as minimmum
    """
    options = [(correct_answer, answer_parent)]  # Start with correct answer
    if id2name is not None:
        correct_option_type = get_entity_type(correct_answer, id2name)
    else:
        correct_option_type = 'any'
    
    # Filter out excluded answers
    safe_entities = []
    for entities in random_entities:
        safe_entities.append([e for e in entities if e not in excluded_answers])
    
    safe_distractors_entities = [(e, parent) for e, parent in distractor_nodes if e not in excluded_answers]
    amazing_entities = []
    for ent, answer_parent in safe_distractors_entities:
            if correct_option_type == 'any' or get_entity_type(ent, id2name) == correct_option_type:
                amazing_entities.append((ent, answer_parent))
    random.shuffle(amazing_entities)
    good_random_entities = []

    for i in range(1, len(safe_entities)):
        parent_ent = context_nodes[i]
        for pos_ent in safe_entities[i]:
            if i > 0:
                if pos_ent == context_nodes[i-1]:
                    continue
            if parent_ent in graph and pos_ent in graph[parent_ent]:
                if correct_option_type == 'any' or get_entity_type(pos_ent, id2name) == correct_option_type:
                    if not distractor_query and graph[parent_ent][pos_ent] in essential_edge_rels:
                        continue
                    good_random_entities.append((pos_ent, parent_ent))
    
    options.extend(good_random_entities)
    
    if len(options) < min_num_options:
        random_options = []
        for i in range(1, len(safe_entities)):
            parent_ent = context_nodes[i]
            pos_ents = [(ent, parent_ent) for ent in safe_entities[i]
                       if (ent, parent_ent) not in options]
            if pos_ents:
                random_options.extend(random.sample(pos_ents, 
                                   min(min_num_options - len(options), len(pos_ents))))
        options.extend(random_options)
    
    # Ensure minimum number of options
    if len(options) < min_num_options:
        origin_vertex = context_nodes[-1]
        pos_ents = [(ent, origin_vertex) for ent in safe_entities[-1] 
                   if (ent, origin_vertex) not in options]
        if pos_ents:
            options_needed = min(min_num_options - len(options), len(pos_ents))
            options.extend(random.sample(pos_ents, options_needed))
    incorrect_options = options[1:]
    incorrect_options = list(set(incorrect_options))
    random.shuffle(incorrect_options)
    if distractor_query and len(amazing_entities) > 0:
        # print(len(amazing_entities), 'haha', amazing_entities)
        amazing_distractor = random.choice(amazing_entities)
        amazing_entities = [other_distractor for other_distractor in amazing_entities if other_distractor != amazing_distractor]
        incorrect_options = [other_opt for other_opt in incorrect_options if other_opt != amazing_distractor]
        incorrect_options = amazing_entities + incorrect_options
        incorrect_options = list(set(incorrect_options))
        options = [options[0]] + [amazing_distractor] + incorrect_options[:min_num_options-2]
    elif distractor_query:
        print('No Amazing Distractor', len(amazing_entities))
        return []
    else:
        options = [options[0]] + incorrect_options[:min_num_options-1]
    random.shuffle(options)
    return options

def get_random_entities_filter(context_nodes, graph, essential_context_edges, distractor_query):
    """Get random entities that are not in the context nodes"""
    random_entities = []
    rels_avoid = set()
    for node1, node2 in essential_context_edges:
        rels_avoid.add(graph[node1][node2])
        
    for node in context_nodes:
        if node in graph and not distractor_query:
            random_entities.append([ent for ent in graph[node] if ent not in context_nodes and graph[node][ent] not in rels_avoid])
        elif node in graph:
            random_entities.append([ent for ent in graph[node] if ent not in context_nodes])
        else:
            random_entities.append([])
    
    return random_entities

class CustomQueryGenerator:
    def __init__(self, graph_algos, discovery_funcs):
        self.graph_algos = graph_algos
        self.discovery_funcs = discovery_funcs

    def get_single_query(self, discovery_func_idx=None, **kwargs):
        """Get a single random query from a random discovery function"""
        if discovery_func_idx is None:
            func = random.choice(self.discovery_funcs)
        else:
            func = self.discovery_funcs[discovery_func_idx]
            
        return func(self.graph_algos.graph, **kwargs)

    def generate_query_data(self, query_result, id2name, graph_text_edge, 
                          tokenizer, shuffle_context=False, 
                          max_context_length=30000, distractor_query=False):
        """Generate full query data from context edges"""

        context_list = []
        context_nodes = set()
        answer_parent = None
        essential_edge_rels = set()
        for node1, node2 in query_result.essential_context_edges:
            context_nodes.add(node1)
            context_nodes.add(node2)
            if node2 == query_result.chosen_answer:
                answer_parent = node1
            if node1 in graph_text_edge and node2 in graph_text_edge[node1]:
                context_list.extend(graph_text_edge[node1][node2])
            essential_edge_rels.add(self.graph_algos.graph[node1][node2])
                
        options = generate_options_with_exclusions(
            query_result.chosen_answer,
            query_result.other_correct_answers,
            list(context_nodes),
            get_random_entities_filter(list(context_nodes), self.graph_algos.graph, query_result.essential_context_edges, distractor_query),
            self.graph_algos.graph,
            answer_parent,
            essential_edge_rels,
            id2name=id2name,
            distractor_query=distractor_query,
            distractor_nodes=query_result.distractor_nodes
        )
        if len(options) == 0:
            return None
        all_context = []
        nodes_to_check = list(context_nodes) + [ent for ent, _ in options]
        for node1, node2 in query_result.context_edges:
            all_context.extend(graph_text_edge[node1][node2])
        
        relevant_options_context_list = []
        for ent, parent_ent in options:
            relevant_text = graph_text_edge[parent_ent][ent]
            relevant_options_context_list.append(relevant_text)
            all_context.extend(relevant_text)
            

        final_context_list = create_context_list(
            [all_context],
            [context_list],
            [relevant_options_context_list],
            tokenizer,
            max_length=max_context_length
        )
        
        if shuffle_context:
            random.shuffle(final_context_list)
        
        context = '\n'.join(['. '.join(context_part) for context_part in final_context_list])
            
        # Format answer options
        answer_options = [ent for ent, _ in options]
        
        return {
            'query': query_result.question,
            'correct_answers': [id2name[query_result.chosen_answer]],
            'path_id': list(context_nodes),
            'path_en': [id2name[node] for node in context_nodes],
            'context': context,
            'correct_ids': [query_result.chosen_answer],
            'answer_options': answer_options,
            'correct_ans_num': answer_options.index(query_result.chosen_answer) + 1,
            'other_correct_answers': [id2name[ans] for ans in query_result.other_correct_answers]
        }