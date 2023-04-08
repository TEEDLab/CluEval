"""
Clustering evaluation metrics for author name disambiguation

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

def cluster_eval(true_cluster, 
                 pred_cluster, 
                 clustering_metric = None, 
                 enable_cluster_id = False, 
                 cluster_id_namespace = "",
                 enable_evlaution_file = False,
                 evaluation_filename = None):
     
    ''' Calulate evaluation scores by the choice of clustering metric '''

    if clustering_metric == "cluster-f":
      score_list = clusterf_precision_recall_fscore(true_cluster, pred_cluster)
      output_string = "Evaluation Scores: measured by '" + \
                       clustering_metric + "'\n\nprecision|recall|f1-score"
      scores = "{:.4f}|{:.4f}|{:.4f}".format(score_list[0], 
                                             score_list[1], 
                                             score_list[2])
      print(output_string)
      print(scores)
    
    elif clustering_metric == "k-metric":
      score_list = kmetric_precision_recall_fscore(true_cluster, pred_cluster)
      output_string = "Evaluation Scores: measured by '" + \
                       clustering_metric + "'\n\nprecision|recall|f1-score"
      scores = "{:.4f}|{:.4f}|{:.4f}".format(score_list[0], 
                                             score_list[1], 
                                             score_list[2])
      print(output_string)
      print(scores) 
    
    elif clustering_metric == "split-lump":
      score_list = split_lump_error_precision_recall_fscore(true_cluster, pred_cluster)
      output_string = "Evaluation Scores: measured by '" + \
                       clustering_metric + "'\n\nprecision|recall|f1-score"
      scores = "{:.4f}|{:.4f}|{:.4f}".format(score_list[0], 
                                             score_list[1], 
                                             score_list[2])
      print(output_string)
      print(scores)

    elif clustering_metric == "pairwise-f":
      score_list = pairwisef_precision_recall_fscore(true_cluster, pred_cluster)
      output_string = "Evaluation Scores: measured by '" + \
                       clustering_metric + "'\n\nprecision|recall|f1-score"
      scores = "{:.4f}|{:.4f}|{:.4f}".format(score_list[0], 
                                             score_list[1], 
                                             score_list[2])
      print(output_string)
      print(scores)

    elif clustering_metric == "b-cubed":
      score_list = bcubed_precision_recall_fscore(true_cluster, pred_cluster)
      output_string = "Evaluation Scores: measured by '" + \
                       clustering_metric + "'\n\nprecision|recall|f1-score"
      scores = "{:.4f}|{:.4f}|{:.4f}".format(score_list[0], 
                                             score_list[1], 
                                             score_list[2])
      print(output_string)
      print(scores)

    elif clustering_metric == "all":
      scores = all_metrics_precision_recall_fscore(true_cluster, pred_cluster)
      #string1 = "Evaluation Scores: measured by '" + clustering_metric + "'\n\n"
      #string2 = "metricname \t {}|{}|{}".format("precision", "recall", "f1-score")
      #output_string = string1 + string2

      output_string = "Evaluation Scores: measured by '" + \
                       clustering_metric + \
                       "'\n\nmetricname \t precision|recall|f1-score"
                       
      print(output_string)
      print(scores)

    else:
      print("Please provide the name of clustering metric you choose")
    
    ''' Create a file containing evaluation scores '''

    if enable_evlaution_file:
        with open(evaluation_filename, 'w') as eval_file:
            eval_file.write(output_string + "\n" + scores)                


    ''' Create an output file containing predicted clusters '''

    clustering_filename = "_".join(['clustering_results', clustering_metric]) + '.txt'
    with open(clustering_filename, 'w') as output_file:
        if enable_cluster_id:
            # Generate UUID for each cluster in the cluster list
            for index, line in enumerate(pred_cluster):
                instance_ids = "|".join(str(e) for e in line)              
                name = clustering_metric + '_' + str(index)
                uuid_obj = uuid.uuid5(cluster_id_namespace, name)
                output_file.write(str(uuid_obj) + "\t" + instance_ids + "\n")                   
        else:
            # If enable_cluster_id is False, simply write cluster_list to output_file
            for line in pred_cluster:
                instance_ids = "|".join(str(e) for e in line) 
                output_file.write(instance_ids + "\n")

    print("\n" + clustering_filename + " created\n")
     

def file_converter(filename):
    """Read in input file and load data
    
    :param filename: text file with clustering id and instance lists
    :return: list of clusters containing integer instance ids
    """
    
    # read in data from input file
    df = pd.read_csv(filename, sep="\t", encoding='utf-8', header=None)
  
    # parse cluster lists 
    # assumes that clusters of instances are located in the last column
    df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: [int(instance) for instance in x.strip().split("|")])
    clusters_list = df.iloc[:, -1].tolist()
    
    return clusters_list


def clusterf_precision_recall_fscore(labels_true, labels_pred):
    """Compute the cluster-f of precision, recall and F-score.
    
    Parameters
    ----------
    :param labels_true: list containing the ground truth cluster labels.
    :param labels_pred: list containing the predicted cluster labels.

    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    
    Reference
    ---------
    Kim, J. (2019). A fast and integrative algorithm for clustering performance evaluation
    in author name disambiguation. Scientometrics, 120(2), 661-681.
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
            
    recall = cMatch/len(truth)
    precision = cMatch/len(predicted)

    try:
        f_score = 2*recall*precision/(recall + precision)
    except ZeroDivisionError:
        f_score = 1.0	

    return (precision, recall, f_score)


def kmetric_precision_recall_fscore(labels_true, labels_pred):
    """Compute the k-metric of precision, recall and F-score.
    
    Parameters
    ----------
    :param labels_true: list containing the ground truth cluster labels.
    :param labels_pred: list containing the predicted cluster labels.

    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    
    Reference
    ---------
    Kim, J. (2019). A fast and integrative algorithm for clustering performance evaluation
    in author name disambiguation. Scientometrics, 120(2), 661-681.
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
	
    recall = aapSum/instSum
    precision = acpSum/instSum

    try:
        f_score = math.sqrt(recall*precision)
    except ZeroDivisionError:
        f_score = 1.0	

    return (precision, recall, f_score)


def split_lump_error_precision_recall_fscore(labels_true, labels_pred):
    """Compute the splitting & lumping error with precision, recall and F-score.
    
    Parameters
    ----------
    :param labels_true: list containing the ground truth cluster labels.
    :param labels_pred: list containing the predicted cluster labels.

    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    
    Reference
    ---------
    Kim, J. (2019). A fast and integrative algorithm for clustering performance evaluation
    in author name disambiguation. Scientometrics, 120(2), 661-681.
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
	
    SE = spSum/instTrSum
    LE = lmSum/instPrSum
  
    recall = 1 - SE
    precision = 1 - LE

    try:
        f_score = (2*recall*precision)/(recall + precision)
    except ZeroDivisionError:
        f_score = 1.0

    return (precision, recall, f_score)


def pairwisef_precision_recall_fscore(labels_true, labels_pred):
    """Compute the pairwise-f of precision, recall and F-score.
    
    Parameters
    ----------
    :param labels_true: list containing the ground truth cluster labels.
    :param labels_pred: list containing the predicted cluster labels.

    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    
    Reference
    ---------
    Kim, J. (2019). A fast and integrative algorithm for clustering performance evaluation
    in author name disambiguation. Scientometrics, 120(2), 661-681.
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
        recall = pairIntSum/pairTrSum
    except ZeroDivisionError:
        recall = 1.0
    
    try:
        precision = pairIntSum/pairPrSum
    except ZeroDivisionError:
        precision = 1.0

    try:
        f_score = (2*recall*precision)/(recall+precision)
    except ZeroDivisionError:
        f_score = 1.0

    return (precision, recall, f_score)


def bcubed_precision_recall_fscore(labels_true, labels_pred):
    """Compute the b-cubed metric of precision, recall and F-score.
    
    Parameters
    ----------
    :param labels_true: list containing the ground truth cluster labels.
    :param labels_pred: list containing the predicted cluster labels.

    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    
    Reference
    ---------
    Kim, J. (2019). A fast and integrative algorithm for clustering performance evaluation
    in author name disambiguation. Scientometrics, 120(2), 661-681.
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
	
    recall = aapSum/instSum
    precision = acpSum/instSum
  
    try:
        f_score = 2*recall*precision/(recall + precision)
    except ZeroDivisionError:
        f_score = 1.0

    return (precision, recall, f_score)
	

def all_metrics_precision_recall_fscore(labels_true, labels_pred):
    """An integrative algorithm for clustering performance evaluation 
    in author name disambiguation
    
	  Parameters
    ----------
    :param labels_true: list containing the ground truth cluster labels.
    :param labels_pred: list containing the predicted cluster labels.

    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    
    Reference
    ---------
    Kim, J. (2019). A fast and integrative algorithm for clustering performance evaluation
    in author name disambiguation. Scientometrics, 120(2), 661-681. 

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
    rec_clusterf = cMatch/len(truth)
    pre_clusterf = cMatch/len(predicted)

    try:
        f_clusterf = 2*rec_clusterf*pre_clusterf/(rec_clusterf + pre_clusterf)
    except ZeroDivisionError:
        f_clusterf = 1.0	

    scores_clusterf = "cluster-f \t {:.4f}|{:.4f}|{:.4f}".format(pre_clusterf, 
                                                               rec_clusterf, 
                                                               f_clusterf)

    # k-metric
    rec_kmetric = aapSum/instSum
    pre_kmetric = acpSum/instSum

    try:
        f_kmetric = math.sqrt(rec_kmetric*pre_kmetric)
    except ZeroDivisionError:
        f_kmetric = 1.0			
    
    scores_kmetric = "k-metric \t {:.4f}|{:.4f}|{:.4f}".format(pre_kmetric, 
                                                             rec_kmetric, 
                                                             f_kmetric)

    # splitting & Lumping Error
    se = spSum/instTrSum
    le = lmSum/instPrSum
  
    rec_sl = 1 - se
    prec_sl = 1 - le

    try:
        f_sl = (2*rec_sl*prec_sl)/(rec_sl + prec_sl)
    except ZeroDivisionError:
        f_sl = 1.0
    
    scores_sl = "SE & LE \t {:.4f}|{:.4f}|{:.4f}".format(prec_sl, 
                                                                   rec_sl, 
                                                                   f_sl)

    # pairwise-F
    try:
        rec_pairwisef = pairIntSum/pairTrSum
    except ZeroDivisionError:
        rec_pairwisef = 1.0
    
    try:
        pre_pairwisef = pairIntSum/pairPrSum
    except ZeroDivisionError:
        pre_pairwisef = 1.0

    try:
        f_pairwisef = (2*rec_pairwisef*pre_pairwisef)/(rec_pairwisef+pre_pairwisef)
    except ZeroDivisionError:
        f_pairwisef = 1.0
    
    scores_pairwisef = "pairwise-f \t {:.4f}|{:.4f}|{:.4f}".format(pre_pairwisef, 
                                                                 rec_pairwisef, 
                                                                 f_pairwisef)

    # b-cubed
    rec_bcubed = aapSum/instSum
    pre_bcubed = acpSum/instSum
  
    try:
        f_bcubed = 2*rec_bcubed*pre_bcubed/(rec_bcubed + pre_bcubed)
    except ZeroDivisionError:
        f_bcubed = 1.0
    
    scores_bcubed = "b-cubed \t {:.4f}|{:.4f}|{:.4f}".format(pre_bcubed, 
                                                           rec_bcubed, 
                                                           f_bcubed)

    output = scores_clusterf + "\n" \
             + scores_kmetric + "\n" \
             + scores_sl + "\n" \
             + scores_pairwisef + "\n" \
             + scores_bcubed

    return output		
### The end of line ###