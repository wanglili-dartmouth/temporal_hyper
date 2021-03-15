#!/bin/bash

#--features ${features} --labels ${labels}\


e=5

for dataset in  dblp
do
    for seed in  10 20 30 40 50 60 70 80 90 100
	do
     for dim in  10 20
     do
        for alpha in  00
        do
            if [ $alpha -eq 100 ];
            then
                alpha=1.00
            else
                alpha=0.$alpha
            fi
            time=1
            if [ $seed -eq 100 ];
            then
                time=1.00
            else
                time=0.$seed
            fi

            data_dir=datasets/${dataset}

            embedding_dir=embeddings/${dataset}/lp_experiment
            walks_dir=walks/${dataset}/lp_experiment
            training_dir=$(printf "edgelists/${dataset}/training_edges")
            edgelist=${training_dir}/edgelist.tsv
            features=${data_dir}/feats.csv
            labels=${data_dir}/labels.csv
            embedding_f=$(printf "${embedding_dir}/alpha=${alpha}/seed=%03d/dim=%03d/%05d_embedding.csv.gz" ${seed} ${dim} ${e})

            if [ ! -f $embedding_f ]
            then

                args=$(echo --edgelist ${edgelist} \
                --embedding ${embedding_dir} --walks ${walks_dir} --seed ${seed} --dim \ ${dim} --alpha ${alpha} -e ${e}  --time ${time})  
                echo ${embedding_dir}
                python main.py ${args}
            fi
        done
    done
done
done