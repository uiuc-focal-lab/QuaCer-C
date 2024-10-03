## Overview

The data processing pipeline involves:
1. Downloading raw Wikidata5m files
2. Preprocessing the dataset
3. Creating the knowledge graph structure
4. Generating supporting files for experiments

## Step 1: Download Raw Data

Download the following files from the Wikidata5m project[Wikidata5m](https://deepgraphlearning.github.io/project/wikidata5m):
- wikidata5m_all_triplet.txt
- wikidata5m_text.txt
- wikidata5m_entity.txt
- wikidata5m_relation.txt

Place these files in the directory of `wikidata_make.py`.

## Step 2: Preprocess Wikidata5m

Run the preprocessing script:

```bash
python wikidata_make.py
```

This script performs the following operations:

1. Filters out ambiguous relations (e.g., 'instance of', 'subclass of')
2. Extracts relevant information for edges
3. Converts Unicode characters to ASCII for consistency

## Step 3: Create Knowledge Graph

The `create_knowledge_graph()` function in `wikidata_make.py`:

1. Builds the graph structure from processed triplets
2. Associates text paragraphs with nodes
3. Creates edge contexts based on relevant sentences

## Step 4: Generate Supporting Files

The script generates several JSON files:

- `wikidata_util.json`: The main knowledge graph structure
- `wikidata_text_edge.json`: Edge context information
- `wikidata_name_id.json`: Mapping between entity names and IDs
- `wikidata_sentencized.json`: Sentencized text for entities

## Key Functions

### `preprocess_wikidata()`

Preprocesses the Wikidata5m dataset:
- Filters relations
- Extracts relevant information for edges
- Converts text to ASCII

### `create_knowledge_graph()`

Creates the knowledge graph structure:
- Builds graph from triplets
- Associates text with nodes
- Creates edge contexts

### `generate_supporting_files()`

Generates necessary JSON files for experiments.

## Usage

To regenerate the processed data:

```bash
python wikidata_make.py
```

Note: This process is computationally intensive, and runs on CPU only.

## Output

Processed files are saved in the `wikidata_graphs` directory:
- wikidata_util.json
- wikidata_text_edge.json
- wikidata_name_id.json
- wikidata_sentencized.json

These files are used by the experiment scripts to generate queries and evaluate LLM performance.