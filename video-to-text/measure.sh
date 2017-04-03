#!/usr/bin/sh


python2 evaluate.py

cd caption-eval
python2 create_json_references.py -i ../results/references.txt -o ../results/references.json
python2 run_evaluations.py -i ../results/predictions.txt -r ../results/references.json
