
# Task1: Full pipeline
We name our work as KGEAR. 
## The task have several stages in sequence:
1. offline: 
    a. Train the GNN.
    b. import the freebase dataset to Virtuoso.
2. online: 
for each query from CWQ dataset, we got the topic entity.
  a. 2-hop neighbor extraction:
    search for all 2-hop neighbors of topic entity in Virtuoso.
  b. 1st step GNN-based pruninig:
    - use the PPR to coarsely filter the entity
    - use the GNN to prune the top k entity.
  c. LLM Pregeneration: Use the real LLM model qwen2.5-32B to judge
    - for prompt:
        - 3 sectoion: system prompt + retrieval prompt + query prompt.
            - system prompt contain the one-shot sample, tells llm what tasks it need to do
            - retrieval prompt contain the retrieval information, like the top K entities.
            - query prompt
        - generate one-shot prompt with a simple example.
    - what does LLM do? 
        - If the LLM judges the evidence sufficient, it generates the answer.
        - If the LLM judges the evidence insufficient, it returns N entities as bridge entities to serve as starting points for path extension.
  d. path extension:
    - Triggered only if the LLM deems evidence insufficient. path_extension() explores 1-hop neighborhoods only from the returned bridge entities.
    - The 1-hop data must be queried from Virtuoso—no fallbacks, heuristics, or simulated results.
    - Merge the newly retrieved triplets into the working set.
  e. LLM final generationL:
    - judge whether the information is sufficient to answer the questi
    - whether it's sufficient or insufficient, generate the final answer.
    - Triggered only if the LLM deems evidence insufficient. 
    - Use the LLM to generate the final answer (generate_final_answer). 
    - if the pre generation think it's sufficient and generate the answer, it will not generate again.

## test metric
1. after b., test the ground truth hit and the reasonning path coverage of the top K entitied pruned by GNN. 
reasonning path coverage: if the reasoning path have 5 triplets, 3 triplets are included in the list. The coverage if 3/5.
2. after c, record the LLM input token and output token size. Record the sufficient judgement. If pre generation think it's sufficient, record the exact match of final answer. If it thinks it's insufficient, record the N bridge entitis it find.
3. after d, record the triplets number after expansion. test the ground truth hit and the reasonning path coverage of the top K entitied pruned by GNN.
4. after f, Record the sufficient judgement. record the LLM answer, whether is exact match with the ground truth answer.
You could use some [] to record the metric. 
E.g. [GNN pruning GT Hit][GNN pruning Reasoning Path Coverage]
[Pre Generation input token][Pre Generation output toke][Sufficency judgement][Generation Exact Match][Bridge Entities]
[Path extension GT Hit][Path extension Reasoning Path Coverage]
[Final Generation Exact Match]

## concrete implemetation:
1. offline: We use the trained GNN in /home/haoyang/private/KGRAG/SubgraphRAG/retrieve/. The structre of the GNN is:
  ├── cwq_Jun15-05:26:49/
  │   ├── cpt.pth                  # ← TRAINED GNN MODEL (16.9 MB)
  │   └── retrieval_result.pth     # Inference results (81.6 MB)
  ├── data_files/cwq/
  │   ├── processed/
  │   │   └── test.pkl             # Processed test data
  │   ├── emb/gte-large-en-v1.5/
  │   │   ├── train.pth
  │   │   ├── val.pth
  │   │   └── test.pth             # Test embeddings
  │   ├── triple_scores/
  │   │   ├── train.pth
  │   │   ├── val.pth
  │   │   └── test.pth             # Triple scores
  │   ├── entity_identifiers.txt   # Entity mapping
  │   └── gpt_triples.pth
2. 1st step GNN-based pruninig:
    - top K: K =5 0
    - We already have some test set GNN pruned result in /home/haoyang/private/KGRAG/SubgraphRAG/retrieve/.
    - You could directly use the result we have extracted to replace the a b step.
3. LLM Pregeneration: 
    - Use the real LLM model qwen2.5-32B to judge
    - N = 3, record the 3 potentail different entities.
4. path extension:
    - use the virtuoso
    - N = 3.
5. LLM final generation: Use the real LLM model qwen2.5-32B to judge


# Task2 Baseline1 SubgraphRAG
I would like to test a baseline for comparison.
You could follow the /home/haoyang/private/KGRAG/SubgraphRAG/ to implement the baseline. You don't need to change too much, the focus is to run the experiments.

## The task have several stages in sequence:
1. offline: 
    a. Train the GNN.
    b. import the freebase dataset to Virtuoso.
2. online: 
for each query from CWQ dataset, we got the topic entity.
  a. 2-hop neighbor extraction:
    search for all 2-hop neighbors of topic entity in Virtuoso.
  b. 1st step GNN-based pruninig:
    - use the PPR to coarsely filter the entity
    - use the GNN to prune the top k entity.
  c. LLM final generation: Use the real LLM model qwen2.5-32B to judge
    - for prompt:
        - Use a direct prompt to judge the result.
    - what does LLM do? 
        - Given the top N triplets, generate the results based on the that.

## test metric
1. after b., test the ground truth hit and the reasonning path coverage of the top K entitied pruned by GNN. 
reasonning path coverage: if the reasoning path have 5 triplets, 3 triplets are included in the list. The coverage if 3/5.
2. after c, record the LLM input token and output token size. Record the LLM answer, whether is exact match with the ground truth answer.
You could use some [] to record the metric. 
E.g. [GNN pruning GT Hit][GNN pruning Reasoning Path Coverage]
[Final Generation input token][Final Generation output toke][Final Generation Exact Match]


## concrete implemetation:
1. offline: We use the trained GNN in /home/haoyang/private/KGRAG/SubgraphRAG/retrieve/. The structre of the GNN is:
  ├── cwq_Jun15-05:26:49/
  │   ├── cpt.pth                  # ← TRAINED GNN MODEL (16.9 MB)
  │   └── retrieval_result.pth     # Inference results (81.6 MB)
  ├── data_files/cwq/
  │   ├── processed/
  │   │   └── test.pkl             # Processed test data
  │   ├── emb/gte-large-en-v1.5/
  │   │   ├── train.pth
  │   │   ├── val.pth
  │   │   └── test.pth             # Test embeddings
  │   ├── triple_scores/
  │   │   ├── train.pth
  │   │   ├── val.pth
  │   │   └── test.pth             # Triple scores
  │   ├── entity_identifiers.txt   # Entity mapping
  │   └── gpt_triples.pth   
2. 1st step GNN-based pruninig:
    - top K: K = 100
    - !!! We already have some test set GNN pruned result in /home/haoyang/private/KGRAG/SubgraphRAG/retrieve/.
    - You could directly use the result we have extracted to replace the a b step.
3. LLM final generation: 
    - Use the real LLM model qwen2.5-32B to judge
    - the final generation prompt is in /home/haoyang/private/KGRAG/SubgraphRAG/retrieve


## Task0: Basic Pipeline testing
the basic pipeline
2. Path:
EM: 42.87%
Results: in test_v1
/home/haoyang/private/KGRAG/KGEAR_final/eval_cwq_full_v1_vllm.py
Checkpoint/progress tracking: results/test_v1/progress.json 
log: results/graphStructure/cwq100_test.log
test:  /home/haoyang/private/KGRAG/KGEAR_final/eval_cwq_full_v1_vllm.py
shell: run_cwq100_graph_structure.sh
main pipeline: /home/haoyang/private/KGRAG/KGEAR_final/eval_cwq_full_v1_vllm.py


## Task1: Explore using the graph structure
1. Using the graph sturcture rather than list structure to improve the LLM ability to detect the correct results.
2. Path:
Results: results/graphStructure/cwq100_full_pipeline.jsonl
log: results/graphStructure/cwq100_test.log
test: test_cwq100_graph_structure_pipeline.py
shell: run_cwq100_graph_structure.sh
main pipeline:  /home/haoyang/private/KGRAG/KGEAR_final/src/kgear_pipeline_vllm.py


# Task2: explore path extension
Try to improve the path extension process. Currently, it expand all 1-hop neighbor, it's inefficient and includes only 1-hop.
I want to try explore the 2-hop neighbors and use the PPR+GNN to optimize it.

## An improved pipeline
1. offline: 
    a. Train the GNN.
    b. import the freebase dataset to Virtuoso.
2. online: 
for each query from CWQ dataset, we got the topic entity.
  a. 2-hop neighbor extraction:
    search for all 2-hop neighbors of topic entity in Virtuoso.
  b. 1st step GNN-based pruninig:
    - use the PPR to coarsely filter the entity
    - use the GNN to prune the top k entity.
  c. LLM Pregeneration: Use the real LLM model qwen2.5-32B to judge
    - for prompt:
        - 3 sectoion: system prompt + retrieval prompt + query prompt.
            - system prompt contain the one-shot sample, tells llm what tasks it need to do
            - retrieval prompt contain the retrieval information, like the top K entities.
            - query prompt
        - generate one-shot prompt with a simple example.
    - what does LLM do? 
        - If the LLM judges the evidence sufficient, it generates the answer.
        - If the LLM judges the evidence insufficient, it returns N entities as bridge entities to serve as starting points for path extension.
  d. path extension:
    - Triggered only if the LLM deems evidence insufficient. path_extension() explores 2-hop neighborhoods from the returned bridge entities.
    - The 2-hop data must be queried from Virtuoso—no fallbacks, heuristics, or simulated results.
    - Do PPR + GNN to prune the path extension.
      - Use the previous GNN in step1 and steo2
      - Implement the same PPR as in Claude_preGeneration
    
  e. LLM final generationL:
    - judge whether the information is sufficient to answer the questi
    - whether it's sufficient or insufficient, generate the final answer.
    - Triggered only if the LLM deems evidence insufficient. 
    - Use the LLM to generate the final answer (generate_final_answer). 
    - if the pre generation think it's sufficient and generate the answer, it will not generate again.

# Task2: prompt improvement

# Task3: dataset exploration

# Task4:Ok
