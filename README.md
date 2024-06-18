# Quantitative Certification of Knowledge Comprehension in LLMs

This repository contains code and resources for certifying knowledge comprehension abilities of Large Language Models (LLMs) during in-context learning, based on the paper **"Quantitative Certification of Knowledge Comprehension in LLMs."**

[![arXiv](https://img.shields.io/badge/arXiv-2402.15929-b31b1b.svg)](https://arxiv.org/abs/2402.15929)

## Overview

This work introduces a method to **certify the comprehension of knowledge by LLMs using a quantitative approach.**  We leverage a knowledge graph to generate questions, evaluate LLM responses against ground truth, and provide a measure of comprehension based on their performance.

**Key Features:**

- **Knowledge Graph Driven:** Utilizes a filtered Wikidata5m knowledge graph for realistic and diverse question generation.
- **Quantitative Certification:** Provides a measurable and interpretable score of knowledge comprehension.
- **Flexible and Extensible:** Easily adaptable to different LLMs and knowledge domains.

**Certification Process:**

1. **Knowledge Graph:** We use a preprocessed Wikidata5m knowledge graph for generating questions.
2. **Pivot Node Certificates:** We select a pivot node to provide a certificate for by selecting queries based on this node.
3. **Path Selection & Prompt Construction:** Random paths are selected from the subgraph of the pivot node to create challenging reasoning based questions.
4. **Response Validation & Certification:** LLM responses are evaluated for correctness, and a confidence interval is calculated.

![Certification Process](image.png)
*Overview of our knowledge comprehension certifier. (a) A knowledge graph G pivoted on
some node, in this case the ’Paul Sophus Epstein’. (b) A randomly chosen path originating at the
pivot node from the various possibilities in G. (c) A prompt created by our prompt constructor using
the selected path and context from the Wikidata5m corpus of the entities (nodes) involved, along
with a distractor context and the final query. (d) The target LLM’s output to the prompt, validated
using the response checker. (e) Certifier obtains bounds on the probability of correct response and
we iterate till the range of the bounds falls below a threshold. (f) The final bounds contained in the
certificate.*

## Getting Started

### Prerequisites


- **Python 3.7+**
- **Required Packages:**
  ```bash
  pip install -r requirements.txt
  ```
- **GPU:** For running the experiments.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/uiuc-focal-lab/QuaCer-C.git
   cd QuaCer-C
   ```

2. **Download Wikidata5m Filtered Data:**
- You can get the related files here: [(google drive)](https://drive.google.com/drive/folders/1q3ELIwexfTiW1mVSJTlQez_6Gvp5Pd9X?usp=sharing)

- **[Optional] Generate Knowledge Graph Files:** 
    - If you prefer to generate the knowledge graph files yourself, you can use the `wikidata_make.py` script. Refer to the script for detailed instructions.
- **File Descriptions:**
    - **`wikidata_util.json`:** Stores the filtered knowledge graph with Wikidata IDs in the following format:
        ```
        graph = {
            'vertex1': {'vertex2': 'relation1', 'vertex3': 'relation2'},
            ...
        }
        ```
    - **`wikidata5m_text_edge.json`:** Stores the context for each edge in the knowledge graph:
        ```
        graph = {
            'vertex1': {'vertex2': ['sent_1', 'sent_2', ...], 'vertex3': ['sent_4', 'sent_5', ...]},
            ...
        }
        ```
    - **`wikidata5m_name_id_uni.json`:** Stores the dictionary mapping between English entity/relation names and Wikidata IDs for easy access:
        ```
        graph = {
            'vertex1': 'vertex1id', 
            'relation1': 'relation1id',
            ...
        }
        ```
- Please refer to `wikidata_make.py` to see how these files were processed.

## Running Experiments

### 1. Answer Checker Server

This experiment requires a separate answer checker server (e.g., Mistral or Gemini). We recommend using our provided `server.py` for easy setup.

   ```bash
   python server.py [--checker_llm_device device] 
   ```

   - **Optional:** Replace `device` with your desired device (e.g., 'cuda:0').
   - The server will listen on port 12345 (configurable). 

### 2. Main Experiments

Use the provided experiment scripts to evaluate different LLMs:

   ```bash
   python vicuna_experiment.py \
       --qa_llm mistralai/Mistral-7B-Instruct-v0.2 \
       --qa_graph_path data/wikidata_util.json \
       --context_graph_edge_path data/wikidata_text_edge.json \
       --results_dir results/mistral_distractorfull/ \
       --entity_aliases_path data/wikidata5m_entity.txt \
       --id2name_path data/wikidata_name_id.json \
       --relation_aliases_path data/wikidata5m_relation.txt \
       --distractor_query \ 
       --num_queries 1000 \
       --host localhost \
       --port 12345
   ```
   
   - **Adjust arguments:** Modify the script arguments (see "Argument Descriptions" below) to use different data paths, and experimental settings.
   - **Other LLM scripts:**  Refer to the `mistral_experiment.py` and `llama_experiment.py` scripts for evaluating other models.

### Argument Descriptions

- **--qa_llm (str):**  Path or identifier of the question-answering LLM.
- **--qa_graph_path (str):** Path to the knowledge graph JSON file.
- **--context_graph_edge_path (str):**  Path to the context descriptions JSON file.
- **--results_dir (str):**  Directory to save experiment results.
- **--entity_aliases_path (str):** Path to the entity aliases text file.
- **--id2name_path (str):** Path to the Wikidata ID to name mapping JSON file.
- **--relation_aliases_path (str):** Path to the relation aliases text file.
- **--distractor_query:** Flag to enable distractor queries.
- **--num_queries (int):** Number of queries to generate.
- **--host (str):** Hostname of the checker server.
- **--port (int):** Port of the checker server.

## Results

- Experiment results are saved as pickle files (.pkl) in the specified `results_dir`. 
- Each file corresponds to a subgraph's pivot entity and contains detailed information for further analysis.

## Evaluating the Checker

- The `checker_mistral_evaluate.ipynb` notebook provides a detailed evaluation of the MistralChecker.
- Refer to the appendix of the paper for more information on evaluating the checker's performance.

## Citation

If you use this code or the findings from our paper, please cite:

```bibtex
@misc{chaudhary2024quacerc,
      title={QuaCer-C: Quantitative Certification of Knowledge Comprehension in LLMs}, 
      author={Isha Chaudhary and Vedaant V. Jain and Gagandeep Singh},
      year={2024},
      eprint={2402.15929},
      archivePrefix={arXiv},
      primaryClass={id='cs.AI' full_name='Artificial Intelligence' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers all areas of AI except Vision, Robotics, Machine Learning, Multiagent Systems, and Computation and Language (Natural Language Processing), which have separate subject areas. In particular, includes Expert Systems, Theorem Proving (although this may overlap with Logic in Computer Science), Knowledge Representation, Planning, and Uncertainty in AI. Roughly includes material in ACM Subject Classes I.2.0, I.2.1, I.2.3, I.2.4, I.2.8, and I.2.11.'}
}
``` 
