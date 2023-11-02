# DBPedia Graph Explorer and Algorithms

This project consists of two main classes, `DBPediaReader` and `GraphAlgos`. `DBPediaReader` is used to explore and build a graph from DBPedia entities and their relations, while `GraphAlgos` provides methods for analyzing and traversing the graph.

## Dependencies

The project requires the following Python libraries:
- `requests`
- `concurrent.futures`
- `queue`
- `threading`
- `random`
- `copy`
- `json`
- `collections`

## Usage

### DBPediaReader

`DBPediaReader` explores DBPedia entities and their relations to build a graph, with the option to specify a maximum depth for exploration.

```python
reader = DBPediaReader()
graph = reader.run(start_entity='Apple_Inc', start_label='Apple Inc', max_depth=5, save_file='graph.json')
```
### GraphAlgos

`GraphAlgos` provides methods for analyzing and traversing the graph, including Breadth-First Search (BFS), Depth-First Search (DFS), computing the diameter of the graph, and finding a path of a specific length.

```python
graph = load_graph('graph.json')
algos = GraphAlgos(graph)
vertices = algos.get_vertices()
relations = algos.get_relations()
relation = algos.get_relation_for_vertex('Apple_Inc', 'IPhone')
diameter = algos.compute_diameter()
visited = algos.dfs('Apple_Inc')
path = algos.dfs_path('Apple_Inc', 3)
```

### DBPediaReader

- ```__init__(self, dbpedia_url='https://dbpedia.org/sparql', max_threads=20, request_timeout=5)```: Initializes a new `DBPediaReader` instance.
- ```exploreWikidata(self, entity, cur_depth, item_labels=None, property_labels=None)```: Explores the relations of a DBPedia entity.
- ```queryWikidataForRelations(self, entity)```: Queries DBPedia for relations of a specified entity.
- ```run(self, start_entity, start_label, max_depth=5, save_file=None) -> dict```: Runs the exploration starting from a specified entity.

### GraphAlgos

- ```__init__(self, graph: dict) -> None```: Initializes a new `GraphAlgos` instance.
- ```bfs(self, start)```: Performs a Breadth-First Search from a specified vertex.
- ```compute_diameter(self)```: Computes the diameter of the graph.
- ```dfs(self, start, visited=None)```: Performs a Depth-First Search from a specified vertex.
- ```dfs_path(self, start, length, path=None)```: Finds a path of a specified length.
- ```get_vertices(self)```: Returns a list of all vertices in the graph.
- ```get_relations(self)```: Returns a list of all relations in the graph.
- ```get_relation_for_vertex(self, start_vertex, target_vertex)```: Returns the relation between two specified vertices.

### `WikiText` Class:
1. **Constructor (`__init__`):**
   - Initializes the `WikiText` object.
   - Accepts `document_dir` as a parameter which specifies the directory where the documents will be saved.
   - Initializes `wiki_en` object for interacting with Wikipedia using the `wikipediaapi` library.
   - Checks if `document_dir` exists, if not, creates it.
2. **Method `fetch_wikipedia_content`:**
   - Fetches the content of a Wikipedia page given its ID (`wiki_id`).
   - Logs the fetching process.
   - Checks if the page exists; if not, issues a warning and returns `None`.
   - If the page exists, returns the text content of the page.
3. **Method `save_wiki_text`:**
   - Fetches the Wikipedia content of a given `entity` and saves it to a text file.
   - Checks if the content is `None`, issues a warning if true, and returns.
   - Saves the content to a text file either in `document_dir` or `new_doc_dir` if specified.

### 'Preparing Documents':
- Use the make_knowledge_dataset.py file to prepare the documents for the RAG model. from https://github.com/huggingface/transformers/blob/main/examples/research_projects/rag/README.md
- To use: ```python examples/research_projects/rag/use_own_knowledge_dataset.py \
    --csv_path path/to/my_csv \
    --output_dir path/to/my_knowledge_dataset \ ```
- The csv file should have two columns: filename and content. The filename column should contain the name of the document and the content column should contain the text content of the document. This can be created using the `docs_to_csv` function in `utils.py`.

### `RagModel` Class:
1. **Constructor (`__init__`):**
   - Initializes the `RAGModel` object.
   - Accepts several parameters including `path`, `passage_path`, `index_path`, `retrieve_sep`, and `cuda`.
   - Loads the tokenizer and retriever from the specified `path`.
   - Initializes the RAG model for sequence generation.
   - Checks if CUDA is available for GPU acceleration and if so, moves the model to the GPU.
2. **Method `run_rag`:**
   - Accepts a `question` string as input and processes it to generate a response.
   - Tokenizes the `question` using the tokenizer initialized in the constructor and prepares the input tensors.
   - If the retrieval and generation steps are combined (controlled by the `retrieve_sep` flag), it generates a response directly using the RAG model.
   - If the retrieval and generation steps are separate, it first retrieves relevant document embeddings using the retriever, computes similarity scores between the question and documents, and then generates a response using the RAG model along with the retrieved documents.
   - Returns the generated response and, in the case of separate retrieval and generation, also returns the context (retrieved documents) used for generation.

### Utility

1. **Function `dbpedia_id_to_wikipedia_url`:**
   - Converts a DBpedia ID to its corresponding Wikipedia URL.
   - Returns the constructed Wikipedia URL.

2. **Function `save_to_txt`:**
   - This function takes in two arguments: `content` and `filename`.
   - It saves the string `content` to a text file specified by `filename`.
   - The text file is saved with UTF-8 encoding to ensure that all characters are correctly encoded.

3. **Function `docs_to_csv`:**
   - This function accepts two arguments: `docs_dir` and `csv_filename`.
   - It iterates through the documents in the `docs_dir` directory, extracting the content of each `.txt` file.
   - It then writes the filename and content of each document to a CSV file specified by `csv_filename`.
   - The CSV file is structured with two columns: "filename" and "content".

4. **Function `load_graph`:**
    - Accepts a single argument: `file_name`, which specifies the name of the JSON file to be read.
    - Reads a JSON file specified by `file_name` and loads the graph data from the file.
    - Returns the loaded graph data as a dictionary.