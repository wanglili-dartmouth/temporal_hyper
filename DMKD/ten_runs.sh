#!/bin/bash

#--features ${features} --labels ${labels}\


e=5

for dataset in  collegemsg  email  ppi dblp  arxiv 
do

     for dim in  2 4 6 8 10 20 128
     do
       for runs in 1 2 3 4 5 6 7 8 9 10
       do
        for alpha in  00
        do

            if [ $dataset = collegemsg ];
            then
                seed=20
                echo $seed
            fi
            
            if [ $dataset = email ];
            then
                seed=10
            fi
            
            
            if [ $dataset = ppi ];
            then
                seed=70
            fi
            
            
            if [ $dataset = dblp ];
            then
                seed=70
            fi
            
            if [ $dataset = arxiv ];
            then
                seed=10
            fi
            echo $seed
            
                
            if [ $alpha -eq 100 ];
            then
                alpha=1.00
            else
                alpha=0.$alpha
            fi
            time=1
            if [ $seed = 100 ];
            then
                time=1.00
            else
                time=0.$seed
            fi
            echo $time
            data_dir=datasets/${dataset}

            embedding_dir=embeddings/${dataset}/lp_experiment
            walks_dir=walks/${dataset}/lp_experiment
            training_dir=$(printf "edgelists/${dataset}/training_edges")
            edgelist=${training_dir}/edgelist.tsv
            features=${data_dir}/feats.csv
            labels=${data_dir}/labels.csv
            embedding_f=$(printf "${embedding_dir}/alpha=${alpha}/seed=%03d/dim=%03d/%05d_embedding.csv.gz" ${seed} ${dim} ${e})
            

            output=edgelists/${dataset}

            test_results=$(printf "test_results/${dataset}/lp_experiment/alpha=${alpha}/dim=%03d/" ${dim})
            embedding_f=$(printf "${embedding_dir}/alpha=${alpha}/seed=%03d/dim=%03d/%05d_embedding.csv.gz" ${seed} ${dim} ${e})
            echo $embedding_f
            
            if [ 1==1 ]
            then

                args=$(echo --edgelist ${edgelist} \
                --embedding ${embedding_dir} --walks ${walks_dir} --seed ${seed} --dim \ ${dim} --alpha ${alpha} -e ${e}  --time ${time})  
                echo ${embedding_dir}
                python main.py ${args}
                
                
                
                 args=$(echo --output ${output} --dist_fn hyperboloid \
                --embedding ${embedding_f} --seed ${seed} \
                --test-results-dir ${test_results})
                echo $args
                python evaluate_lp.py ${args}
            fi
        done
    done
done
done