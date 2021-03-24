# Instructions

This folder contains files with annotations collected from crowdworkers and an expert; see published paper for more details on these two types of evaluation.

## Expert annotations

The expert annotations are recorded in `expert_results.tsv`. The spreadsheet should be self-explanatory.

## Crowd annotations

The crowdworker annotations are recorded in `cf_results.csv`. This csv file is created by CrowdFlower.

To process the csv file to compute crowdworker's guessing accuracy, run: `python process_cf.py -r cf_results.csv`
