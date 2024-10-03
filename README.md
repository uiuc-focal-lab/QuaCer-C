# Quantitative Certification of Knowledge Comprehension in LLMs

This repository contains code and resources for certifying knowledge comprehension abilities of Large Language Models (LLMs) during in-context learning, based on the paper **"Quantitative Certification of Knowledge Comprehension in LLMs."**

[![arXiv](https://img.shields.io/badge/arXiv-2402.15929-b31b1b.svg)](https://arxiv.org/abs/2402.15929)

## Overview

This work introduces a method to **certify the comprehension of knowledge by LLMs using a quantitative approach.**  We leverage a knowledge graph to generate questions, evaluate LLM responses against ground truth, and provide a measure of comprehension based on their performance.

**Key Features:**

- **Formal Probabilistic Guarantees:** QuaCer-C provides high-confidence bounds on the probability of correct LLM responses.
- **Knowledge Graph Specifications:**  Uses knowledge graphs to define distributions of multi-hop reasoning problems, enabling precise and scalable prompt generation.
- **Model-Agnostic:**  Compatible with both open and closed-source LLMs via API access.
- **Quantitative Certificates:** Delivers certificates as tight bounds on the probability of correctness, offering a nuanced measure of LLM capability.

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
- **GPU:** For running the experiments with Open-Source LLMs.
- **API Keys:** Required for accessing closed-source LLMs like Gemini and GPT-4.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/uiuc-focal-lab/QuaCer-C.git
   cd QuaCer-C
   ```

2. **Download Wikidata5m Filtered Data:**
- Download the preprocessed Wikidata5m data from: [(google drive)](https://drive.google.com/drive/folders/1q3ELIwexfTiW1mVSJTlQez_6Gvp5Pd9X?usp=sharing)
- Please also download entity.txt and relation.txt from [(Wikidata5m raw)](https://deepgraphlearning.github.io/project/wikidata5m)

- **[Optional] Generate Knowledge Graph Files:** 
    - If you prefer to generate the knowledge graph files yourself, you can use the `wikidata_make.py` script. Refer to `preprocessREADME.md` for detailed instructions.
- **File Descriptions:**
    - wikidata_util.json: Filtered knowledge graph.

    - wikidata_text_edge.json: Context for each edge in the knowledge graph.

    - wikidata_name_id.json: Mapping between Wikidata IDs and entity/relation names.

    - wikidata_sentencized.json: Sentencized text for each Wikidata entity.

    - wikidata5m_entity.txt: Wikidata entity aliases.

    - wikidata5m_relation.txt: Wikidata relation aliases.

- Please refer to `wikidata_make.py` to see how these files were processed.

## Running Experiments
### 1. Main Experiments

The primary experiments are run using the `experiment_utils.py` script along with model-specific experiment files.  Examples are provided for several LLMs:

```bash
python {model}_experiment.py \
    --qa_llm {model_name} \
    --qa_graph_path data/wikidata_util.json \
    ... (other arguments - see below) 
```

Replace `{model}_experiment.py` and `{model_name}` with the appropriate values. The provided experiment scripts are:

- `gemini_experiment.py`: For Gemini models.
- `gpt_experiment.py`: For GPT models.
- `mistral_experiment.py`: For Mistral models.
- `llama_experiment.py`: For Llama models (including Vicuna).
- `phi_experiment.py`: For Phi models.

### Important Arguments


- `--qa_llm`: Name or path of the LLM to be certified.
- `--quant_type`: (Optional) Quantization type for open-source models (e.g., '8_bit', '4_bit').
- `--distractor_query`: (Optional) Enable distractor queries (recommended).
- `--shuffle_context`: (Optional) Shuffle context in prompts.
- `--num_queries`: Number of queries to generate per certificate.
- `--num_certificates`: Total number of certificates to generate (each for a different subgraph).
- `--results_dir`: Directory to store results.
- `--k`: Maximum path length for multi-hop reasoning.

Please refer to mainly the `experiment_utils.py` file and somewhat to the individual experiment files for model-specific arguments and default settings. You can always run `python file.py --help`

### [Optional] Creating a Custom Model Experiment File

You can create your own `model_experiment.py` file to adapt QuaCer-C to a new LLM.  Your custom file should define two main functions:

1. `get_args()`: This function should parse command-line arguments specific to your model.  It should inherit base arguments from `experiment_utils.get_base_args()` and add any model-specific arguments. The minimum args to add are 'qa-llm' and 'quant_type'. See `model_experiment_template.py` or `gemini_experiment.py` for details.


2.  `load_model(model_name, only_tokenizer=False, gpu_map=GPU_MAP, quant_type=None)`:  This function loads your LLM and its tokenizer. It should handle different quantization options if applicable. Note that the tokenizer can be None if not needed by query_model function defined below.


3. `query_model(prompts, model, tokenizer, do_sample=True, top_k=10, num_return_sequences=1, max_length=240, temperature=1.0, INPUT_DEVICE='cuda:0')`: This function takes a list of prompts and returns the LLM's responses. It should handle batching and any necessary preprocessing or postprocessing of prompts and responses.


4. `main()`: The main execution function. It calls `experiment_utils.run_experiment()` with the appropriate arguments, including your `load_model` and `query_model` functions.



A template file (`model_experiment_template.py`) is provided in the repository for reference.  You can copy and modify this template to create your custom experiment file.

### Distributed Experiments

For multi-GPU setups, modify the `GPU_MAP` and `INPUT_DEVICE` variables in the experiment scripts(see `mistral_experiment.py`) to distribute computation across available GPUs.

## Troubleshooting

- For out-of-memory errors, use a smaller model or use a combination:
 ```python
    torch.cuda.empty cache()
    gc.collect()
 ```
 in the query_model function like `phi_experiment.py`. Note that this may slow the certification.

- Modify `experiment_utils.py` for detailed logging.

## Results

Results are saved as pickle files (.pkl) in the specified `results_dir`. Each file corresponds to a subgraph's pivot entity and contains:
- Generated queries
- Model responses
- Evaluation results
- All query details(eg. path_id, answer options, etc.)

Use the provided Jupyter notebooks(`analyse_results.py`) to get the certification bounds.

## Code Organization

- `utils.py`: Core utility functions for graph algorithms, prompt generation, and response checking.
- `experiment_utils.py`: Functions for running the main certification experiments.
- `{model}_experiment.py`: Model-specific experiment scripts.
- `server.py`: Server for Mistral-based answer checking.
- `wikidata_make.py`:  [Generally not needed] Script for preprocessing the raw Wikidata5m data.  Use only if you need to regenerate the knowledge graph files from scratch.

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
