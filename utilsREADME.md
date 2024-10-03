# Utilities for Knowledge Graph Reasoning and LLM Evaluation

This document describes the core utilities provided in `utils.py` for working with knowledge graphs, generating questions, and evaluating LLM responses.

**Core Classes and Functions:**

### `GraphAlgos`

The `GraphAlgos` class provides methods for manipulating and querying knowledge graphs.  Key functionalities include:

- **Initialization:**
    ```python
    graph_algos = GraphAlgos(graph, entity_aliases, relation_aliases)
    ```
    - `graph`: The knowledge graph as a dictionary.  See `wikidata_util.json` for the expected format.
    - `entity_aliases`: Dictionary of entity aliases.
    - `relation_aliases`: Dictionary of relation aliases.

- **Graph Traversal and Querying:**
    - `bfs(start)`: Breadth-first search.
    - `compute_diameter()`: Computes graph diameter.
    - `dfs(start, visited=None)`: Depth-first search.
    - `dfs_path(start, length, path=None)`: Finds a path of a specified length.
    - `get_vertices()`: Returns all vertices in the graph.
    - `get_relations()`: Returns all relations in the graph.
    - `get_relation_for_vertex(start_vertex, target_vertex)`: Returns the relation between two vertices.
    - `get_path_for_vertices(start, end, k=5)`: Finds a path between two vertices (limited depth).
    - `get_queries_for_relpath(rel_path, start_vertex)`: Finds paths following a specific sequence of relations.
    - `generate_query_for_path(path)`: Generates a query string from a path.
    - `generate_query_for_vertices(start, end, k=5, path=None)`: Generates a query for a path between vertices.
    - `sample_random_vertex(vertex_list=None)`: Samples a random vertex.

- **Random Path and Query Generation:**
    - `generate_random_path(path_len=25, source=None)`: Generates a random path (attempts uniqueness).
    - `generate_random_query(k=5, return_path=False, source=None)`: Generates a random query from a random path.

- **Distractor Generation:**
    - `get_best_distractor(start_vertex, path, do_choose=True)`: Finds a suitable distractor node for a given path.

- **Subgraph Creation:**
    - `create_subgraph_within_radius(start_vertex, k)`: Creates a subgraph within a given radius.
    - `get_best_vertices(num=1000, method='outdegree', **kwargs)`:  Gets "best" vertices based on various criteria (primarily used internally during experiment setup).

### Prompt and Context Generation Utilities

- `get_alias(id, aliases)`: Retrieves a random alias for a given ID.
- `form_alias_question(question, path, entity_aliases, relation_aliases, entity_name2id, relation_name2id, graph_algos)`: [Not actively used] Replaces entities and relations in a question with aliases.
- `load_aliases(path)`: Loads aliases from a file.
- `form_context_list(query_path, wikidata_text_edge, wikidata_util, entity_top_alias)`: Forms a list of context sentences for a given path.
- `dumb_checker(model_answer, correct_answer_num)`: Checks if the model's answer matches the expected answer number format.
- `create_context_list(all_sents, relevant_sents_path, relevant_sents_opts, tokenizer, max_length=15000)`: Creates a trimmed context list for LLMs.
- `get_all_context(query_path, wikidata_text_sentencized)`: Retrieves all context for entities in a path.
- `get_random_entities(query_path, wikidata_util)`: Gets random entities related to the path (for answer option generation).

### Final Query Generation Utility
- `get_query_data(graph_algos, source, id2name, graph_text_edge, graph_text_sentencized, tokenizer, distractor_query=False, k=5, shuffle_context=True, max_context_length=30000)`:  Combines many of the above utilities to generate all necessary data for a single query (prompt, context, answer options, etc.)

Parameters:
- `graph_algos`: GraphAlgos instance
- `source`: Source vertex for the query
- `id2name`: Mapping of IDs to names
- `graph_text_edge`: Edge context information
- `graph_text_sentencized`: Sentencized text for entities
- `tokenizer`: Tokenizer for the target LLM
- `distractor_query`: Whether to include distractor information
- `k`: Maximum path length
- `shuffle_context`: Whether to shuffle the context
- `max_context_length`: Maximum context length for the LLM

Returns: A dictionary containing query information, context, and answer options.

## Experiment Utilities

Key functions in `experiment_utils.py` for running experiments.

### `load_experiment_setup(args, load_model, GPU_MAP)`

Loads necessary data and models for experiments.

### `run_experiment(args, load_model, query_model_func, GPU_MAP, model_context_length, BATCH_NUM=1, INPUT_DEVICE='cuda:0')`

Main function for running certification experiments.

Parameters:
- `args`: Parsed command-line arguments
- `load_model`: Function to load the LLM
- `query_model_func`: Function to query the LLM
- `GPU_MAP`: GPU memory mapping
- `model_context_length`: Maximum context length for the LLM
- `BATCH_NUM`: Batch size for queries
- `INPUT_DEVICE`: GPU device for input processing

Returns: Experiment results and statistics.