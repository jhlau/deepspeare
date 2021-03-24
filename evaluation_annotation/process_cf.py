"""
Author:         Jey Han Lau
Date:           Oct 17
"""

import argparse
import sys
import unicodecsv as csv
import cPickle as pickle
from collections import defaultdict
import numpy as np
import operator

#parser arguments
desc = "Process CF results to compute accuracy"
parser = argparse.ArgumentParser(description=desc)

#arguments
parser.add_argument("-r", "--result-csv", required=True, help="csv file containing CF results")
parser.add_argument("-m", "--model-pickle", help="pickle file containing model and IDs")
args = parser.parse_args()

#parameters
debug = False
golden_col = 11
judgement_col = 1
id1_col = 5
id2_col = 6
worker_col = 17

###########
#functions#
###########

def get_model_names(row, model):

    selected, unselected = "", ""
    if row[judgement_col] == "Poem 1":
        selected = row[id1_col]
        unselected = row[id2_col]
    else:
        selected = row[id2_col]
        unselected = row[id1_col]

    selected_mname = (selected if model == None else model[int(selected)])
    unselected_mname = (unselected if model == None else model[int(unselected)])

    return selected_mname, unselected_mname

def get_score(selected_mname, unselected_mname):

    if selected_mname != unselected_mname:

        if selected_mname == "real":
            return unselected_mname, 1.0
        else:
            return selected_mname, 0.0

    else:

        return None, None
######
#main#
######

def main():

    #load ids and model names
    model = None
    if args.model_pickle:
        model = pickle.load(open(args.model_pickle))
        if debug:
            print model

    #first parse to find perfect score worker (perfect score worker might be cheating)
    worker_accs = defaultdict(list)
    for row in csv.reader(open(args.result_csv), encoding="utf-8"):
        if row[golden_col] != "false":
            continue

        worker_id = (row[worker_col], row[worker_col+1])
        selected_mname, unselected_mname = get_model_names(row, model)
        key, score = get_score(selected_mname, unselected_mname)

        if key != None:
            worker_accs[worker_id].append(score)

    #remove annotations from perfect workers
    perfect_workers = set([])
    worker_meanacc = {}
    country_count = (defaultdict(int), defaultdict(int))
    for k, v in sorted(worker_accs.items()):
        worker_meanacc[k] = np.mean(v)
        if np.mean(v) == -1.0:
            perfect_workers.add(k)
            country_count[1][k[1]] += 1
        country_count[0][k[1]] += 1

    #print "Number of perfect workers =", len(perfect_workers), "/", len(worker_accs)
    if debug:
        print "\nall country count =", sorted(country_count[0].items(), key=operator.itemgetter(1), reverse=True)
        print "\nperfect country count =", sorted(country_count[1].items(), key=operator.itemgetter(1), reverse=True)
        for k, v in sorted(worker_meanacc.items(), key=operator.itemgetter(1), reverse=True):
            print k, v

    #parse results csv
    accs = defaultdict(list)
    for row in csv.reader(open(args.result_csv), encoding="utf-8"):
        if row[golden_col] != "false":
            continue

        worker_id = (row[worker_col], row[worker_col+1])

        if worker_id in perfect_workers:
            continue

        selected_mname, unselected_mname = get_model_names(row, model)
        key, score = get_score(selected_mname, unselected_mname)

        if key != None:
            accs[key].append(score)

            if debug:
                print "\n", row[0], ":", row[id1_col], "vs.", row[id2_col]
                print "Selected ID =", selected_mname, score
        
    #print mean accuracy
    for k in accs.keys():
        print "Mean Accuracy: Real vs.", k, "=", np.mean(accs[k]), "(", len(accs[k]), ")"

if __name__ == "__main__":
    main()
