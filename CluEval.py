#######################################################################
# Code set to evaluate input and output clustering for named entity disambiguation in Python 3
# This code set was developed under a grant from the National Science Foundation 
# (NSF NCSES Award # 1917663: Creating a Data Quality Control Framework for Producing New Personnel-Based S&E Indicators) 
# and its supplementary fund program, Research Experiences for Undergraduates (REU).
#######################################################################


# NOTE: Run this script with 'metrics.py' in the same directory


import time
import uuid
from datetime import timedelta
from metrics import *


##########################################################
##########             Set Parameters          ###########
##########################################################


### 1. Which format of input would you like to use? ###

"""
Input data are required to be prepared in a specific format
Either file or list of data can be used.

Option 1: Read in input files and convert data to list of clusters
cluster file: cluster id and instance id list
              > Instance ids in instance id list are separated by vertical bar
Each file is created in .txt and columns are separated by tab

true_cluster_file: file containing true clusters
pred_cluster_file: file containing predicted clusters

Option 2: Load data directly with the format of list containing clusters

true_cluster: a list containing true clusters of instance ids
pre_cluster: a list containing predicted clusters of instance ids

"""

''' Option 1: Convert data from files into list '''

true_cluster_file  = 'cluster_true.txt'
pred_cluster_file   = 'cluster_pred.txt'

true_cluster = file_converter(true_cluster_file)
pred_cluster = file_converter(pred_cluster_file)

''' Option 2: Load data directly to function '''
# true_cluster = NAME OF THE VARIABLE 
# pred_cluster = NAME OF THE VARIABLE 


### 2. Which clustering evaluation metric do you want to use? ###

"""
(1) cluster-f: cluster-f precision/recall/f1
(2) k-metric: k-metric precision/recall/f1
(3) split-lump: splitting & lumping error precision/recall/f1
(4) pairwise-f: paired precision/recall/f1    
(5) b-cubed: b3 precision/recall/f1  
(6) all: all types of clustering metric combined

For more details on clustering evaluation metrics, see a paper below
Kim, J. (2019). A fast and integrative algorithm for clustering performance evaluation
    in author name disambiguation. Scientometrics, 120(2), 661-681. 

"""

clustering_metric = "all" 


### 3. Would you like to assign a distinct identifier to each cluster output? ###

"""
enable_cluster_id = True
enable_cluster_id = False 


The parameter enable_cluster_id controls whether a unique identifier is 
assigned to each cluster within the namespace "550e8400-e29b-41d4-a716-44665544abcd".
This can be useful for tracking individual clusters throughout an analysis. 
To enable cluster ID, set enable_cluster_id to True. The output file includes 
IDs in the first column and cluster lists in the second column with a tab as a delimiter. 
To disable cluster ID, set enable_cluster_id to False.

The namespace used in this script is a UUID (Universally Unique Identifier) 
generated with the value '550e8400-e29b-41d4-a716-44665544abcd'. A UUID is a 128-bit
identifier that is globally unique and can be used to prevent naming conflicts in
between different systems or entities. This namespace is used to create deterministic 
UUIDs using the uuid5() function from the uuid module, which takes a namespace and 
a name as input and generates a UUID based on them. 

"""

enable_cluster_id = True
cluster_id_namespace = uuid.UUID('550e8400-e29b-41d4-a716-44665544abcd')


### 4. Would you like to have evaluation scores in a separate file? ###

"""
enable_evlaution_file = True
enable_evlaution_file = False

If enable_evaluation_file is set to True, the evaluation scores (precision, recall, f1-score)
measure by the selection of clustering metric are displayed in the screen
and a separate file containing the scores will be generated.
If enable_evaluation_file is set to False, the evaluation scores are only displayed in the acreen.

"""

enable_evlaution_file = True
evaluation_filename = "evaluation_scores_" + clustering_metric + ".txt" 


##########################################################
##########           Run Main Function         ###########
##########################################################

''' measure start time '''
start_time = time.time()


if __name__ == "__main__":
    
    cluster_eval(true_cluster,
                 pred_cluster,
                 clustering_metric     = clustering_metric,
                 enable_cluster_id     = enable_cluster_id,
                 cluster_id_namespace  = cluster_id_namespace,
                 enable_evlaution_file = enable_evlaution_file,
                 evaluation_filename   = evaluation_filename
                 )


''' measure finish time '''
elapsed_time_secs = time.time() - start_time
msg = "\nrun time: %s secs" % timedelta(seconds=round(elapsed_time_secs))
print(msg)

### The end of line ###
