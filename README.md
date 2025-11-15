# Hybrid GNN-QUBO for Knowledge Graph Alignment

### Project Proposal

#### Objective:

Design a hybrid pipeline that aligns two Knowledge Graphs (KGs) by framing the task as a global combinatorial optimization problem, which is solvable on a quantum annealer through the QUBO framework.

#### Problem:

Aligning KGs is difficult.

Current GNN methods:

- Require a large set of pre-aligned "seed" entities for supervised training (they are data-hungry).
- Rely on a greedy nearest-neighbor search for alignment, which is fast but not guaranteed to find the best global solution and can get stuck in a local optimum.

#### Our Hypothesis:

A global optimization approach (QUBO) can find a more accurate and structurally-consistent alignment from unsupervised GNN embeddings than a local, greedy search can.

#### Our Pipeline:

**1. Knowledge Extraction (NLP)**

- Use SciBERT to parse raw text (Wikipedia, arXiv) in the quantum computing domain.
- Output: Two separate KGs (`kg_wiki`, `kg_arxiv`), complete with entities (e.g., Qubit) and relation labels (e.g., "is_used_in").

**2. Unsupervised Representation (GAE)**

- Train two separate Graph Autoencoders (GAEs), one on each KG, using a self-supervised task (e.g., link prediction).

  - This will make it learn the unique structure and role of each entity within its own graph.
- Output: Two sets of entity embeddings. NOTE: These two embedding spaces are not aligned with each other.

**3. QUBO Formulation (Global Optimization)**

- Formulate the alignment task as a QUBO ($H_{total}=H_{node}+H_{structure}+H_{constraint}$). This finds the best "pattern match" between the two unaligned spaces.
- $H_{node}$ (Linear): A weak hint. The similarity between the GAE entity embeddings from Phase 2.
- $H_{structure}$ (Quadratic): The global pattern matcher. The similarity between the SciBERT relation embeddings (e.g., sim("builds", "produces")). This rewards preserving structural relationships.
- $H_{constraint}$ (Quadratic): A penalty to enforce a flexible "at-most-one" mapping. This is a critical design choice that allows entities to remain unaligned if no good match is found.

$$
H_{total} = \underbrace{\sum_{i,a}{-S(i, a) \cdot x_{i,a}}}_{H_{\text{node}}} + \underbrace{\sum_{i,j,a,b}{-w_{ij,ab} \cdot x_{i,a} \cdot x_{j,b}}}_{H_{\text{structure}}} + \underbrace{\sum_{i} P_{1} \sum_{a=1}^M \sum_{b=a+1}^M x_{i,a} x_{i,b}}_{\text{Constraint}\ 1} + \underbrace{\sum_{a} P_{2} \sum_{i=1}^N \sum_{j=i+1}^N x_{i,a} x_{j,a}}_{\text{Constraint}\ 2}
$$

- Where:
  - $x_{i,a}$: A binary variable (1 or 0) that is 1 if we align entity $i$ from KG1 with entity $a$ from KG2.
  - $S(i,a)$: The similarity score between entity $i$ and $a$, derived from the GAE embeddings.
  - $w_{ij,ab}$: The structural similarity weight, derived from the SciBERT relation embeddings.
  - $P_1, P_2$: Large positive penalty constants to enforce the constraints.
  - Constraint 1: Enforces that each entity $i$ in KG1 maps to at most one entity in KG2. If $i$ matches zero entities, the penalty is 0. If it matches one, the penalty is 0. If it matches two or more, the penalty is high.
  - Constraint 2: Enforces that each entity $a$ in KG2 is mapped to by at most one entity from KG1. This allows entities to remain unaligned, making the formulation more robust to realistic KGs that do not have perfect 1-to-1 overlap.
- Output: A single Q matrix representing the entire optimization problem.

**4. Solve & Benchmark**

- Method A (Ours): Solve the QUBO on a D-Wave quantum annealer.
- Method B (Baseline): Apply a greedy Nearest Neighbor search to the exact same GAE embeddings from Phase 2.
- Evaluation: Compare the accuracy of both methods against a manually-created "ground truth" (F1-Score).

#### Code Structure:

```
├── main.py               # Run the full pipeline.
├── config.py             # Store all global variables.
├── requirements.txt      # Dependencies.
│
├── kg_construction/# Module 1: Build KGs from raw text.
│   ├── fetch_data.py     # Download Wikipedia/arXiv articles.
│   ├── nlp_pipeline.py   # Extract entities/relations with SciBERT.
│   └── llm_based_pipeline.py # Alternative Groq-powered KG builder (JSON triples).
│   └── build_kg.py       # Create and save the rdflib .ttl graphs.
│
├── gnn_embedding/# Module 2: Learn entity embeddings.
│   ├── dataset.py        # Load .ttl files, convert to PyG Data objects.
│   ├── model.py          # Define the Graph Autoencoder (GAE) model.
│   ├── train.py          # Train the GAE, save the model weights (.pth).
│   └── generate_embeddings.py # Load trained model, save embeddings (.pt).
│
├── qubo_alignment/# Module 3: Formulate the QUBO problem.
│   ├── weights.py        # Calculate H_node (from GAE) and H_structure (from SciBERT).
│   └── formulate.py      # Use pyqubo to build and save the final Q-matrix.
│
├── evaluation/# Module 4: Solve and benchmark.
│   ├── solvers.py        # Contains `solve_qubo()` and `solve_nearest_neighbor()`.
│   ├── metrics.py        # Calculates Precision, Recall, F1, etc.
│   └── run_benchmark.py  # Load ground truth and run both solvers, print scores.
│
├── data/# Stores raw downloaded text.
├── ground_truth/# Stores the manual "ground_truth.csv" files.
│
└── output/# Generated files.
    ├── models/# Trained GAE model weights (.pth).
    ├── kg/# .ttl graph files.
    ├── embeddings/# The final entity embeddings (.pt).
    ├── qubo/# The final Q-matrix (.pkl).
    └── results/# The final alignment CSVs.

```

### LLM-based KG extraction

The Groq-driven pipeline in `src/kg_construction/llm_based_pipeline.py` prompts a modern LLM to return JSON-formatted entities and triples and then writes the unpruned KGs. To enable it:

1. Duplicate the sample env file and add your Groq key:
  ```bash
  cp .env.example .env
  # edit .env and set GROQ_API_KEY
  ```
2. Install the updated dependencies (`groq`, `python-dotenv`).
3. Instantiate `LLMBasedPipeline` in your script or notebook:
  ```python
  from src.kg_construction.fetch_data import fetch_wiki_data, fetch_arxiv_data
  from src.kg_construction.llm_based_pipeline import LLMBasedPipeline

  wiki_data, _ = fetch_wiki_data()
  arxiv_data, _ = fetch_arxiv_data()

  LLMBasedPipeline().build_unpruned_kgs(wiki_data, arxiv_data)
  ```

The pipeline persists entities and TTL files to the same locations used by the classical NLP pipeline, so downstream steps (pruning, embedding, alignment) continue to work unchanged.
