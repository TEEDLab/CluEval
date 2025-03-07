# Updated: 03/07/2025

#######################################################################
# Code set to evaluate input and output clustering for named entity disambiguation
# Developed under NSF NCSES Award #1917663
# and its supplementary REU program.
#######################################################################

# NOTE: Run this script with 'metrics.py' in the same directory

import time
import uuid
from datetime import timedelta
from metrics import *

##########################################################
#                 Set Parameters                         #
##########################################################

### Step1: Designate files to import ###
# Option 1: Convert data from files into a list

true_cluster_file = 'cluster_true.txt'
pred_cluster_file = 'cluster_pred.txt'

true_cluster = file_converter(true_cluster_file)
pred_cluster = file_converter(pred_cluster_file)

# Option 2: Load data directly (uncomment below to use)

# true_cluster = NAME_OF_TRUE_CLUSTER_VARIABLE
# pred_cluster = NAME_OF_PREDICTED_CLUSTER_VARIABLE

### Step 2: Select clustering evaluation metric ###
# Options: "cluster-f", "k-metric", "split-lump", "pairwise-f", "b-cubed", "all"
# For details about each metric, see comments in metrics.py

clustering_metric = "all"

# Step 3: Decide if unique identifiers are assigned to clusters
# True: include cluster IDs; False: exclude cluster IDs
# For details about UUID, see a page at
# https://en.wikipedia.org/wiki/Universally_unique_identifier

enable_cluster_id = True
cluster_id_namespace = uuid.UUID('550e8400-e29b-41d4-a716-44665544abcd')

# Step 4: Decide if evaluation scores are saved to a separate file
# True: create an evaluation result file; False: print scores only

enable_evaluation_file = True
evaluation_filename = f"evaluation_scores_{clustering_metric}.txt"

##########################################################
#                  Main Function                         #
##########################################################

if __name__ == "__main__":
    start_time = time.time()

    cluster_eval(true_cluster,
                 pred_cluster,
                 clustering_metric=clustering_metric,
                 enable_cluster_id=enable_cluster_id,
                 cluster_id_namespace=cluster_id_namespace,
                 enable_evaluation_file=enable_evaluation_file,
                 evaluation_filename=evaluation_filename)

    elapsed_time_secs = time.time() - start_time
    print("\nRun time: %s secs" % timedelta(seconds=round(elapsed_time_secs)))

# End of script
