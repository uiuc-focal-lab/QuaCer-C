# Quantitative Certification of Knowledge Comprehension in LLMs

This code contains checkpoints and training code for the following work:

* **Quantitative Certification of Knowledge Comprehension in LLMs**

## Overview
- This work aims to certify the knowledge comprehension ability of LLMs during in-context learning.
- The certification is summarized in the figure below:
![hello](image.png)
*Overview of our knowledge comprehension certifier. (a) A knowledge graph G pivoted on
some node, in this case the ’Paul Sophus Epstein’. (b) A randomly chosen path originating at the
pivot node from the various possibilities in G. (c) A prompt created by our prompt constructor using
the selected path and context from the Wikidata5m corpus of the entities (nodes) involved, along
with a distractor context and the final query. (d) The target LLM’s output to the prompt, validated
using the response checker. (e) Certifier obtains bounds on the probability of correct response and
we iterate till the range of the bounds falls below a threshold. (f) The final bounds contained in the
certificate.*

- We use a distilled version of Wikidata5m as a base knowledge graph for our experiments.

- For each model architecture we test(Mistral, Vicuna, Llama), we provide a separate python script to run experiments on the model. The scripts contain code to generate queries from the knowledge graph and get LLM response as well as check the answer. The utils.py file provides some common functionality and the functions are described in detail in utilsREADME.md file.

### Setup

**Dependencies**

* Python 3.x
* transformers
* numpy
* statsmodels
* unidecode
* genai (if using Gemini checker)
* torch
* fastchat
* accelerate
* llama2 repository and model weights
* Environment variable `GOOGLE_API_KEY` (if using Gemini checker)

**Environment Setup**

1. **Install Packages:** `pip install transformers numpy statsmodels unidecode genai torch fastchat accelerate`
2. For **LLaMA2**: follow instructions at https://github.com/facebookresearch/llama. Clone the repository as llama_local for use with existing scripts or modify the import statements in the llama2 script accordingly.
3. Add environment variable 'LLAMA_PATH' which is the folder where the llama2 weights and tokenzier and stored, you can also pass in the full path if required to run the scripts.
4. **API Key:** If using the Gemini checker, set the `GOOGLE_API_KEY` environment variable. We use only the MistralChecker for our experiments.

### Wikidata5m Filtered:
- You can get the related files here: insert_link [(google drive)](https://drive.google.com/drive/folders/1q3ELIwexfTiW1mVSJTlQez_6Gvp5Pd9X?usp=sharing)
- You can also generated the kg files using the wikidata5m.ipynb notebook.
- *wikidata5m_en_util_unidecoded.json*: stores the filtered knowledge graph with english names in the format: 
`
    graph = {'vertex1': {'vertex2': 'relation1', 'vertex3': 'relation2'},
    ...}
`
- *wikidata5m_en_unidecoded.json*: stores the filtered knowledge graph with english names in the format: 
`
    graph = {'vertex1': {'relation1': 'vertex2', 'relation2': 'vertex3'},
    ...}
`
- *wikidata5m_unidecoded.json*: stores the filtered knowledge graph with wikidata ids in the format: 
`
    graph = {'vertex1': {'relation1': 'vertex2', 'relation2': 'vertex3'},
    ...}
`
- *wikidata5m_en_text.json*: stores the context for each entity with its english name as key: 
`
    graph = {'vertex1': 'abstract_v1', 'vertex2':'abstract_v2'
    ...}
`
- *wikidata5m_name_id_uni.json*: stores the dictionary between english entity/relation names and wikidata ids for easy access: 
`
    graph = {'vertex1': 'vertex1id', 'relation1', 'relation1id',
    ...}
`
- The notebook that took the raw kg wikidata5m to get the filtered versions is wikidata5m.ipynb.

### Experiment With Vicuna
**How to Run the Experiment**

```bash
python vicuna_experiment.py --qa_llm [QA Model path on huggingface] --checker_llm [Checker Model path on huggingface] --num_queries [Number of Max Queries] --[Other Optional Arguments]
```

**Example**:
```bash
python vicuna_texperiment.py --qa_llm lmsys/vicuna-7b-v1.5 --checker_llm mistralai/Mistral-7B-Instruct-v0.2 --num_queries 1000
```

**Experiment Results**
The script saves results as pickle files (.pkl) with the entity ID of the pivot of the subgraph. These contain detailed information for further analysis.

**Key Functions**

* **`load_model()`** Loads the QA and (optionally) the checker language models.
* **`simple_checker()`** A basic answer-checking function using string matching and aliases.
* **`check_answer()`** Comprehensive correctness checker that can use either the `simple_checker` or an external checker model.
* **`query_vicuna_model()`** Handles question answering using the specified LLM.
* **`experiment_pipeline()`** The core function that executes the experimental loop.

##### Argument Descriptions

* --qa_llm (str): Path or name of the QA language model.
* --checker_llm (str): Path or name of the checker language model (Mistral or Gemini).
* --qa_graph_path (str): Path to the JSON file containing the knowledge graph for question answering.
* --context_graph_path (str): Path to the JSON file containing textual descriptions for graph entities.
* --qa_llm_device (str): Device to use for the QA model ('cuda:1', etc.)
* --checker_llm_device (str): Device for the checker model.
* --results_path (str):  Where to save experimental results. 
* --entity_aliases_path (str), --id2name_path (str),

* --relation_aliases_path (str): Paths to the respective data files.
* --num_queries (int): The total number of questions to generate in the experiment. 
* --gpu_map (dict): Specify desired memory allocation for each device.

### Experiment with Mistral

**How to Run the Experiment**

```bash
python mistral_experiment.py --num_queries [Number of Max Queries] --[Other Optional Arguments]
```

**Example**:
```bash
python mistral_experiment.py --num_queries 1000 
```

**Experiment Results**
The script saves results as pickle files (.pkl) with the entity ID of the pivot of the subgraph (e.g., `exp7b_Q12345.pkl`). These files contain detailed information for further analysis.

**Key Functions**

* **`get_args()`** Parses command-line arguments for experiment configuration.
* **`simple_checker()`** A basic answer-checking function using string matching and aliases.
* **`check_answer()`** Comprehensive correctness checker that can use either the `simple_checker` or an external checker model (like Mistral).
* **`experiment_pipeline()`** The core function that executes the experimental loop.  This includes:
    * Generating questions from the knowledge graph.
    * Using the specified QA model to get answers.
    * Evaluating answer correctness using the checker. 

**Argument Descriptions**

* **--qa_llm (str):** Path or name of the QA  language model (default: `mistralai/Mistral-7B-Instruct-v0.2`).
* **--checker_llm (str):** Path or name of the checker language model (default: `mistralai/Mistral-7B-Instruct-v0.2`).
* **--qa_graph_path (str):** Path to the JSON file containing the knowledge graph for question answering.
* **--context_graph_path (str):** Path to the JSON file containing textual descriptions for graph entities.
* **--qa_llm_device (str):** Device to use for the QA model ('cuda:1', etc.). Default is 'cuda:1'.
* **--checker_llm_device (str):** Device to use for the checker model (default: 'cuda:3').
* **--results_path (str):** Where to save experimental results. 
* **--entity_aliases_path (str), --id2name_path (str), --relation_aliases_path (str):** Paths to the respective data files.
* **--num_queries (int):** The total number of questions to generate in the experiment.

### Experiment with llama2

**Prerequisites**

* **Checker Server:** This experiment requires a separate answer checker server (e.g., Mistral or Gemini) already running.  You can use `server.py` (if provided) to set this up.
* **LLaMA2 Model:** The code assumes you have the LLaMA model files downloaded.

**How to Run the Experiment**

1. **Start the Checker Server:** Follow the instructions for your checker server to start it on your local machine. Note the host and port it's running on.

2. **Run the Experiment Script:** 
   ```bash
   torchrun --nproc_per_node [num_nodes] llama_experiment.py --host [checker_host] --port [checker_port] --num_queries [Number of Max Queries] --[Other Optional Arguments]
   ```
   * Replace `[checker_host]` and `[checker_port]` with the actual values where your checker server is reachable. 

**Example:**
```bash
torchrun --nproc_per_node 1 llama_experiment.py --host localhost --port 12345 --num_queries 1000
```

**Experiment Results**
The script saves results as pickle files (.pkl) with the entity ID of the pivot of the subgraph (e.g., `exp_Q12453.pkl`). These files contain detailed information for further analysis.

**Key Functions**

* **`get_args()`** Parses command-line arguments for experiment configuration.
* **`simple_checker()`** A basic answer-checking function using string matching and aliases.
* **`check_answer()`** Sends a question, reference answer, and model-generated answer to the external checker server to get a correctness evaluation.
* **`build_llama()`**  Loads the LLaMA model.
* **`query_llama_model()`** Handles question answering using the LLaMA model.
* **`experiment_pipeline()`** The core function that executes the experimental loop, generating questions and getting answers from LLaMA.

**Argument Descriptions**

* **--qa_llm_path (str):** Path to the LLaMA model files.
* **--tokenizer_path (str):** Path to the LLaMA tokenizer.
* **--qa_graph_path (str):** Path to the JSON file containing the knowledge graph for question answering.
* **--context_graph_path (str):** Path to the JSON file containing textual descriptions for graph entities.
* **--results_path (str):** Where to save experimental results. 
* **--entity_aliases_path (str), --id2name_path (str), --relation_aliases_path (str):** Paths to the respective data files.
* **--num_queries (int):** The total number of questions to generate in the experiment. 
* **--host (str):** The hostname where your checker server is running (default: `localhost`).
* **--port (int):** The port where your checker server is listening (default: `12345`).

Absolutely! Here's a small README for `server.py`:

**server.py for llama2 experiments**

* **Purpose**

    This script sets up a server that acts as an answer correctness checker. It's designed to serve as a means to check for answers for llama2 responses. This is designed to run the MistralChecker.

* **How to Run**

    1. **Start the Server:**
    ```bash 
    python server.py [--checker_llm_device device]
    ```
    * **Optional:** Replace `device` with the desired device for your checker model (e.g., 'cuda:3').

* **Usage**

    * The server will listen on the specified port (default: 12345).
    * Your main experiment script should send requests to this server containing:
    * `question`: The question being asked.
    * `correct_answer`: The ground truth answer.
    * `model_answer`:  The answer generated by your question-answering model.
    * This is already done by the llama_experiment.py script.

* **Response**

    The server will process the request using the checker model and return a JSON object containing the original data along with a `result` field indicating whether the model's answer is considered correct.

* **Key Functions**

    * **`process_request()`** Handles incoming requests, calls the checker, and prepares the response.
    * **`check_answer()`**  The function that interfaces with your Mistral (or other) checker model.
    * **`main()`** Sets up the socket server and handles the listening loop.

### Evaluating Checker
*    Our work on evaluating the MistralChecker is present in the checker_mistral_evaluate.ipynb file. FOr details reagarding evaluating see the appendix of the paper.