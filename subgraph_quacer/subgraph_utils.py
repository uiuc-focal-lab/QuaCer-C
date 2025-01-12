from utils import *
import numpy as np
import os
import json

class CustomQueryResult:
    def __init__(self, question, chosen_answer, essential_context_edges, all_context_edges, other_correct_answers=None):
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
                
        self.distractor_context_edges = []
        self.other_correct_answers = other_correct_answers if other_correct_answers else []

def generate_options_with_exclusions(correct_answer, excluded_answers, context_nodes, 
                                   random_entities, graph, distractor_nodes, min_num_options=4):
    """
    Generate answer options while excluding certain nodes.
    
    Args:
        correct_answer: The chosen correct answer node ID
        excluded_answers: List of node IDs to exclude from options
        context_nodes: List of relevant context node IDs
        random_entities: List of potential random entities
        graph: The graph structure
    """
    options = [(correct_answer, context_nodes[1])]  # Start with correct answer
    
    # Filter out excluded answers
    safe_entities = []
    for entities in random_entities:
        safe_entities.append([e for e in entities if e not in excluded_answers])
    
    rel_to_find = graph[context_nodes[1]][correct_answer]
    good_random_entities = []

    for i in range(1, len(safe_entities)):
        parent_ent = context_nodes[i]
        for pos_ent in safe_entities[i]:
            if i > 0:
                if pos_ent == context_nodes[i-1]:
                    continue
            if parent_ent in graph and pos_ent in graph[parent_ent] and graph[parent_ent][pos_ent] == rel_to_find:
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
            
    random.shuffle(options)
    return options

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
        
        for node1, node2 in query_result.essential_context_edges:
            context_nodes.add(node1)
            context_nodes.add(node2)
            if node1 in graph_text_edge and node2 in graph_text_edge[node1]:
                context_list.extend(graph_text_edge[node1][node2])

        distractor_nodes = []
        if distractor_query:
            distractor_nodes = [nodes[0] for nodes in query_result.distractor_context_edges]
                
        options = generate_options_with_exclusions(
            query_result.chosen_answer,
            query_result.other_correct_answers,
            list(context_nodes),
            get_random_entities(list(context_nodes), self.graph_algos.graph),
            self.graph_algos.graph,
            distractor_nodes
        )

        all_context = []
        nodes_to_check = list(context_nodes) + [ent for ent, _ in options]
        for node1, node2 in query_result.context_edges:
            all_context.extend(graph_text_edge[node1][node2])
        
        relevant_options_context_list = []
        for ent, parent_ent in options:
            relevant_text = graph_text_edge[parent_ent][ent]
            relevant_options_context_list.append(relevant_text)
            

        final_context_list = create_context_list(
            [all_context],
            [context_list],
            relevant_options_context_list,
            tokenizer,
            max_length=max_context_length
        )
        
        if shuffle_context:
            random.shuffle(final_context_list)
        
        context = '\n'.join([' '.join(context_part) for context_part in final_context_list])
            
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