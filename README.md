# CluEval
Clustering Evaluation (pronounced as 'clue val') for Author Name Disambiguation  <br />
This code is to evaluate input and output clustering for author name disambiguation in Python 3. <br />
<br />
Clustering evaluation metrics for author name disambiguation <br/>

(1) cluster-f: cluster-f precision/recall/f1 <br />
(2) k-metric: k-metric precision/recall/f1 <br />
(3) split-lump: splitting & lumping error precision/recall/f1 <br />
(4) pairwise-f: paired precision/recall/f1 <br />
(5) b-cubed: b3 precision/recall/f1 <br />
(6) all: all types of clustering metric combined <br />

For more details on clustering evaluation metrics, see a paper below: <br />
Kim, J. (2019). A fast and integrative algorithm for clustering performance evaluation
    in author name disambiguation. Scientometrics, 120(2), 661-681. 
    
## How to run files
1. Download two files: 'CluEval.py' and 'metrics.py'  <br />
2. Make sure that two files are in the same directory and run 'CluEval.py' <br />
3. You can also use sample input files provided in the sample_data folder
