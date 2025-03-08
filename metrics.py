# Updated: 03/07/2025

"""
Clustering evaluation metrics for named entity disambiguation

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

import time
import math 
import uuid
import numpy as np
import pandas as pd

def cluster_eval(true_cluster, pred_cluster, clustering_metric=None, 
                 enable_cluster_id=False, cluster_id_namespace="",
                 enable_evaluation_file=False, evaluation_filename=None):
    """
    Calculate evaluation scores based on the chosen clustering metric.
    """

    # Define available metrics and corresponding functions
    metrics_functions = {
        "cluster-f": clusterf_precision_recall_fscore,
        "k-metric": kmetric_precision_recall_fscore,
        "split-lump": split_lump_error_precision_recall_fscore,
        "pairwise-f": pairwisef_precision_recall_fscore,
        "b-cubed": bcubed_precision_recall_fscore,
        "all": all_metrics_precision_recall_fscore
    }

    if clustering_metric not in metrics_functions:
        print("Error: Invalid clustering metric specified.")
        return

    # Compute scores
    score_result = metrics_functions[clustering_metric](true_cluster, pred_cluster)

    # Prepare the output string
    if clustering_metric == "all":
        output_string = f"Evaluation Scores (all metrics):\n\n{score_result}"
    else:
        precision, recall, f_score = score_result
        output_string = (
            f"Evaluation Scores measured by '{clustering_metric}':\n\n"
            f"precision | recall | f1-score\n"
            f"{precision:.4f} | {recall:.4f} | {f_score:.4f}"
        )

    print(output_string)

    # Write evaluation scores to a file, if enabled
    if enable_evaluation_file and evaluation_filename:
        with open(evaluation_filename, 'w') as eval_file:
            eval_file.write(output_string)

    # Generate clustering results file
    clustering_filename = f"clustering_results_{clustering_metric}.txt"
    with open(clustering_filename, 'w') as output_file:
        if enable_cluster_id and cluster_id_namespace:
            for index, cluster in enumerate(pred_cluster):
                instance_ids = "|".join(str(instance) for instance in cluster)
                cluster_uuid = uuid.uuid5(cluster_id_namespace, f"{clustering_metric}_{index}")
                output_file.write(f"{cluster_uuid}\t{instance_ids}\n")
        else:
            for cluster in pred_cluster:
                instance_ids = "|".join(str(instance) for instance in cluster)
                output_file.write(instance_ids + "\n")

    print(f"\n{clustering_filename} created\n")

     

def file_converter(filename):
    """
    Read in an input file and load data
    
    :param filename: text file with clustering id and instance lists
    :return: list of clusters containing integer instance ids
    """
    
    # read in data from input file
    df = pd.read_csv(filename, sep="\t", encoding='utf-8', header=None)
  
    # parse cluster lists 
    # assumes that clusters of instances are located in the last column
    df = pd.read_csv(filename, sep="\t", encoding='utf-8', header=None)
    clusters_list = df.iloc[:, -1].apply(lambda x: [int(instance) for instance in x.strip().split("|")])
    return clusters_list.tolist()


def compute_fscore(precision, recall):
    """
    Compute F-score safely, handling division by zero.
    """
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def clusterf_precision_recall_fscore(labels_true, labels_pred):
    """
    Compute the cluster-f of precision, recall and F-score.
    
    Parameters:
        labels_true (list): True clusters as a list of lists of instance IDs.
        labels_pred (list): Predicted clusters as a list of lists of instance IDs.

    Returns:
        str: Formatted string with precision, recall, and F-score for all metrics.
    """
    
    truth = labels_true
    predicted = labels_pred

    cSize = {}
    cMatch = 0

    pIndex = {}
    
    for i, pred_i in zip(range(len(predicted)),predicted):
        for p in pred_i:
            pIndex[p] = i + 1

        cSize[i + 1] = len(pred_i)        

    for true_j in truth:
        tMap = {}

        for t in true_j:
            if not pIndex[t] in tMap.keys():
                tMap[pIndex[t]] = 0
            tMap[pIndex[t]] = tMap[pIndex[t]] + 1

        for key, value in sorted(tMap.items(), key = lambda kv: kv[0]):
            if value == len(true_j) and cSize[key] == len(true_j):
                cMatch += 1   
            
    precision = cMatch/len(predicted)
    recall = cMatch/len(truth)
    f_score = compute_fscore(precision, recall)        

    return (precision, recall, f_score)


def kmetric_precision_recall_fscore(labels_true, labels_pred):
    """
    Compute the k-metric of precision, recall and F-score.
    
    Parameters:
        labels_true (list): True clusters as a list of lists of instance IDs.
        labels_pred (list): Predicted clusters as a list of lists of instance IDs.

    Returns:
        str: Formatted string with precision, recall, and F-score for all metrics.
    """
    
    truth = labels_true
    predicted = labels_pred

    cSize = {}
    cMatch = 0
    aapSum = 0 
    acpSum = 0
    pIndex = {}
    
    for i, pred_i in zip(range(len(predicted)),predicted):
        for p in pred_i:
            pIndex[p] = i + 1
        cSize[i + 1] = len(pred_i)       
    
    instSum = 0
    for true_j in truth:
        instSum += len(true_j)
        tMap = {}

        for t in true_j:
            if not pIndex[t] in tMap.keys():
                tMap[pIndex[t]] = 0
            tMap[pIndex[t]] = tMap[pIndex[t]] + 1

        for key, value in sorted(tMap.items(), key = lambda kv: kv[0]):
            if value == len(true_j) and cSize[key] == len(true_j):
                cMatch += 1   
            aapSum += pow(value,2)/len(true_j)
            acpSum += pow(value,2)/cSize[key]
    
    precision = acpSum/instSum
    recall = aapSum/instSum

    try:
        f_score = math.sqrt(recall*precision)
    except ZeroDivisionError:
        f_score = 1.0

    return (precision, recall, f_score)


def split_lump_error_precision_recall_fscore(labels_true, labels_pred):
    """
    Compute the splitting & lumping error with precision, recall and F-score.
    
    Parameters:
        labels_true (list): True clusters as a list of lists of instance IDs.
        labels_pred (list): Predicted clusters as a list of lists of instance IDs.

    Returns:
        str: Formatted string with precision, recall, and F-score for all metrics.
    """
    
    truth = labels_true
    predicted = labels_pred
    
    cSize = {}
    spSum = 0
    lmSum = 0
    instTrSum = 0
    instPrSum = 0
    pIndex = {}
    
    for i, pred_i in zip(range(len(predicted)),predicted):
        for p in pred_i:
            pIndex[p] = i + 1

        cSize[i + 1] = len(pred_i)         # hash of a cluster P_i and its size
        
    for true_j in truth:
        tMap = {}
        maxKey = 0
        maxValue = 0

        for t in true_j:
            if not pIndex[t] in tMap.keys():
                tMap[pIndex[t]] = 0
            tMap[pIndex[t]] = tMap[pIndex[t]] + 1

        for key, value in sorted(tMap.items(), key = lambda kv: kv[0]):
            if value > maxValue:
                maxValue = value
                maxKey = key
            instTrSum += len(true_j) 
            instPrSum += cSize[maxKey] 
            spSum += (len(true_j) - maxValue)  
            lmSum += (cSize[maxKey] - maxValue) 
    
    LE = lmSum/instPrSum
    SE = spSum/instTrSum

    precision = 1 - LE
    recall = 1 - SE
    f_score = compute_fscore(precision, recall)

    return (precision, recall, f_score)


def pairwisef_precision_recall_fscore(labels_true, labels_pred):
    """
    Compute the pairwise-f of precision, recall and F-score.
    
    Parameters:
        labels_true (list): True clusters as a list of lists of instance IDs.
        labels_pred (list): Predicted clusters as a list of lists of instance IDs.

    Returns:
        str: Formatted string with precision, recall, and F-score for all metrics.
    """
    
    truth = labels_true
    predicted = labels_pred
    
    pairPrSum = 0
    pairTrSum = 0
    pairIntSum = 0
    pIndex = {}
      
    for i, pred_i in zip(range(len(predicted)),predicted):
        for p in pred_i:
            pIndex[p] = i + 1
        pairPrSum += len(pred_i)*(len(pred_i) - 1)/2
        

    for true_j in truth:
        pairTrSum += len(true_j)*(len(true_j) - 1)/2
        tMap = {}

        for t in true_j:
            if not pIndex[t] in tMap.keys():
                tMap[pIndex[t]] = 0
            tMap[pIndex[t]] = tMap[pIndex[t]] + 1

        for key, value in sorted(tMap.items(), key = lambda kv: kv[0]):
            pairIntSum += value*(value - 1)/2
            
    try:
        precision = pairIntSum/pairPrSum
    except ZeroDivisionError:
        precision = 1.0
        
    try:
        recall = pairIntSum/pairTrSum
    except ZeroDivisionError:
        recall = 1.0

    f_score = compute_fscore(precision, recall)

    return (precision, recall, f_score)


def bcubed_precision_recall_fscore(labels_true, labels_pred):
    """
    Compute the b-cubed metric of precision, recall and F-score.
    
    Parameters:
        labels_true (list): True clusters as a list of lists of instance IDs.
        labels_pred (list): Predicted clusters as a list of lists of instance IDs.

    Returns:
        str: Formatted string with precision, recall, and F-score for all metrics.
    """
    
    truth = labels_true
    predicted = labels_pred

    cSize = {}
    cMatch = 0
    aapSum = 0 
    acpSum = 0
    pIndex = {}
    
    for i, pred_i in zip(range(len(predicted)),predicted):
        for p in pred_i:
            pIndex[p] = i + 1

        cSize[i + 1] = len(pred_i) # hash of a cluster P_i and its size
        
    instSum = 0 
    for true_j in truth:
        instSum += len(true_j)
        tMap = {}

        for t in true_j:
            if not pIndex[t] in tMap.keys():
                tMap[pIndex[t]] = 0
            tMap[pIndex[t]] = tMap[pIndex[t]] + 1

        for key, value in sorted(tMap.items(), key = lambda kv: kv[0]):
            if value == len(true_j) and cSize[key] == len(true_j):
                cMatch += 1     
            aapSum += pow(value,2)/len(true_j)
            acpSum += pow(value,2)/cSize[key]
    
    precision = acpSum/instSum
    recall = aapSum/instSum
    f_score = compute_fscore(precision, recall)

    return (precision, recall, f_score)
    

def all_metrics_precision_recall_fscore(labels_true, labels_pred):
    
    """
    Compute all clustering evaluation metrics: 
    Cluster-F, K-Metric, Split-Lump, Pairwise-F, and B-Cubed.

    
    Parameters:
        labels_true (list): True clusters as a list of lists of instance IDs.
        labels_pred (list): Predicted clusters as a list of lists of instance IDs.

    Returns:
        str: Formatted string with precision, recall, and F-score for all metrics.
    """
    
    ## load data
    truth = labels_true
    predicted = labels_pred
    
    ## compute clustering metrics
    cSize = {}
    cMatch = 0
    aapSum = 0 
    acpSum = 0
    spSum = 0
    lmSum = 0
    instTrSum = 0
    instPrSum = 0
    pairPrSum = 0
    pairTrSum = 0
    pairIntSum = 0
    pIndex = {}
    
    for i, pred_i in zip(range(len(predicted)),predicted):
        for p in pred_i:
            pIndex[p] = i + 1

        cSize[i + 1] = len(pred_i)        # hash of a cluster P_i and its size
        pairPrSum += len(pred_i)*(len(pred_i) - 1)/2
        
    instSum = 0
    for true_j in truth:
        instSum += len(true_j)
        pairTrSum += len(true_j)*(len(true_j) - 1)/2
        tMap = {}
        maxKey = 0
        maxValue = 0

        for t in true_j:
            if not pIndex[t] in tMap.keys():
                tMap[pIndex[t]] = 0
            tMap[pIndex[t]] = tMap[pIndex[t]] + 1

        for key, value in sorted(tMap.items(), key = lambda kv: kv[0]):
            if value == len(true_j) and cSize[key] == len(true_j):
                cMatch += 1               # the count of P_i that contains all and the only instances in T_j
            aapSum += pow(value,2)/len(true_j)
            acpSum += pow(value,2)/cSize[key]
            if value > maxValue:
                maxValue = value
                maxKey = key
            pairIntSum += value*(value - 1)/2
            instTrSum += len(true_j)      # sum of instances in the truth clusters for a unique author
            instPrSum += cSize[maxKey]    # sum of instances in the largest predicted clusters for a unique author
            spSum += (len(true_j) - maxValue)   # sum of split instances
            lmSum += (cSize[maxKey] - maxValue) # sum of lumped instances
    

    ## measure clustering performance
    # cluster-f
    pre_clusterf = cMatch/len(predicted)
    rec_clusterf = cMatch/len(truth)
    f_clusterf = compute_fscore(pre_clusterf, rec_clusterf)
        

    scores_clusterf = "cluster-f \t {:.4f}|{:.4f}|{:.4f}".format(pre_clusterf, rec_clusterf, f_clusterf)

    # k-metric
    rec_kmetric = aapSum/instSum
    pre_kmetric = acpSum/instSum

    try:
        f_kmetric = math.sqrt(rec_kmetric*pre_kmetric)
    except ZeroDivisionError:
        f_kmetric = 1.0         
    
    scores_kmetric = "k-metric \t {:.4f}|{:.4f}|{:.4f}".format(pre_kmetric, rec_kmetric, f_kmetric)

    # splitting & Lumping Error
    le = lmSum/instPrSum
    se = spSum/instTrSum
    pre_sl = 1 - le
    rec_sl = 1 - se
    f_sl = compute_fscore(pre_sl, rec_sl)
    
    scores_sl = "SE & LE \t {:.4f}|{:.4f}|{:.4f}".format(pre_sl, rec_sl, f_sl)

    # pairwise-F
    try:
        pre_pairwisef = pairIntSum/pairPrSum
    except ZeroDivisionError:
        pre_pairwisef = 1.0

    try:
        rec_pairwisef = pairIntSum/pairTrSum
    except ZeroDivisionError:
        rec_pairwisef = 1.0

    f_pairwisef = compute_fscore(pre_pairwisef, rec_pairwisef)
    
    scores_pairwisef = "pairwise-f \t {:.4f}|{:.4f}|{:.4f}".format(pre_pairwisef, rec_pairwisef, f_pairwisef)

    # b-cubed
    pre_bcubed = acpSum/instSum
    rec_bcubed = aapSum/instSum
    f_bcubed = compute_fscore(pre_bcubed, rec_bcubed)
    
    scores_bcubed = "b-cubed \t {:.4f}|{:.4f}|{:.4f}".format(pre_bcubed, rec_bcubed, f_bcubed)

    output = scores_clusterf + "\n" \
             + scores_kmetric + "\n" \
             + scores_sl + "\n" \
             + scores_pairwisef + "\n" \
             + scores_bcubed

    return output       
### The end of line ###
