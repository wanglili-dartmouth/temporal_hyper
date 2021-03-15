#!/bin/bash

for dataset in slash
do



    edgelist=datasets/${dataset}/edgelist.tsv
    features=datasets/${dataset}/feats.csv
    labels=datasets/${dataset}/labels.csv
    output=edgelists/${dataset}

    edgelist_f=$(printf "${output}/training_edges/edgelist.tsv")

    if [ ! -f $edgelist_f  ]
    then
        python remove_edges.py --edgelist=$edgelist --output=$output
        #python remove_edges.py --edgelist=$edgelist --features=$features --labels=$labels --output=$output --seed $seed
    fi
done
