#!/bin/bash


e=5
exp=nc_experiment

for dataset in dblp
do
    for seed in   70
	do
     for dim in   4  8 
     do
        for alpha in   00 10 20
        do
            if [ $alpha -eq 100 ];
            then
                alpha=1.00
            else
                alpha=0.$alpha
            fi
            if [ $seed -eq 100 ];
            then
                time=1.00
            else
                time=0.$seed
            fi

            data_dir=datasets/${dataset}
            edgelist=${data_dir}/edgelist.tsv
            features=${data_dir}/feats.csv
            labels=${data_dir}/labels.csv
            embedding_dir=embeddings/${dataset}/lp_experiment

            test_results=$(printf "test_results/${dataset}/${exp}/alpha=${alpha}/dim=%03d/" ${dim})
            embedding_f=$(printf "${embedding_dir}/alpha=${alpha}/seed=%03d/dim=%03d/%05d_embedding.csv.gz" ${seed} ${dim} ${e})
            echo $embedding_f

            args=$(echo --edgelist ${edgelist} --labels ${labels} \
                --dist_fn hyperboloid \
                --embedding ${embedding_f} --seed ${seed} \
                --test-results-dir ${test_results})
            echo $args

            python evaluate_svm_nc.py ${args}
        done
    done
done
done