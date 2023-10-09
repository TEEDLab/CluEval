# CluEval
Clustering Evaluation (pronounced as 'clue val') for named entity disambiguation  <br />
This code is to evaluate clustering outputs for named entity disambiguation in Python 3. <br />
<br />

If you use this tool in your research or tool development, please cite as follows:
Kim, Jinseok and Kim, Jenna (2023). CluEval: A Python tool for evaluating clustering performance in named entity disambiguation. Software Impacts, Volume 16, 100510

Here is the link to the full text of the paper (open-access)
https://www.softwareimpacts.com/article/S2665-9638(23)00047-7/fulltext

Clustering evaluation metrics: <br/>

(1) 'cluster-f': cluster-f precision/recall/f1 <br />
(2) 'k-metric': k-metric precision/recall/f1 <br />
(3) 'split-lump': splitting & lumping error precision/recall/f1 <br />
(4) 'pairwise-f': paired precision/recall/f1 <br />
(5) 'b-cubed': b3 precision/recall/f1 <br />
(6) 'all': all types of clustering metric combined <br />
<br />
For more information on clustering evaluation metrics, please refer to the paper below: <br />
Kim, Jinseok. (2019). A fast and integrative algorithm for clustering performance evaluation
    in author name disambiguation. Scientometrics, 120(2), 661-681. 
<br />    
## How to run files
1. The code was implemented on Python 3.9.16. To run the code, make sure that numpy and pandas are installed. The version we used: numpy (1.22.4) & pandas (1.4.4). You can install them using pip. Type the following command in the terminal or command prompt:

           pip install numpy==1.22.4  
           pip install pandas==1.4.4

2. Download two files: 'CluEval.py' and 'metrics.py'. Make sure that they are located in the same directory. <br />
3. Place your input files into the directory. You can use sample files provided in the data folder <br/> 
4. Run 'CluEval.py' file. <br />

## Acknowledgements
The development of CluEval received funding from the National Science Foundation (NSF NCSES Award #1917663).
