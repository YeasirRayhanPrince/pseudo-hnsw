# Goal
[PSEUDO-HNSW]
1. The goal is to compress the trajectories in the /scratch/gilbreth/yrayhan/pseudo-hnsw/dataset/SIFT1M/01_07_26_M_16_efCons_200/sift1M_result_path_v2.csv file into a Decision Transformer that does behavior cloning.
3. Input & Output specification
  - Input to the Transformer: 
    - a query vector q
    - k (the number of nearest neighbour to find)
    - r (rank of the nearest neighbour)
  - Output of the Transformer:
    - The trajectory of hnsw (ID1, ID2, ID3)
      - ID1 represents the ID of the node (vecor embedding) in layer 1
      - ID1, ID2 can be from the same layer
      - The trajectory output length can be variable depending on the k
        - k = -1, it outputs the rth ranked knn
        - k = +ve, it outputs the k nearest neighbours of query vector v
4. Raw Traning dataset:
  - Query vector: /scratch/gilbreth/yrayhan/pseudo-hnsw/dataset/SIFT1M/01_07_26_M_16_efCons_200/sift_base_raw_vectors.csv
  - Trajectory of hnsw: /scratch/gilbreth/yrayhan/pseudo-hnsw/dataset/SIFT1M/01_07_26_M_16_efCons_200/sift1M_result_path_v2.csv
    - Format: (ID, layer)

[DONE]
- A sample code has been implemented.

[TODO]
1. Follow the procedure shown in /scratch/gilbreth/yrayhan/pseudo-hnsw/decision-transformer/atari/run_dt_atari.py with model = 'naive' to clone the trajectories in hnsw, given the input
