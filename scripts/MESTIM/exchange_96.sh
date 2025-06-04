export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./csv_results" ]; then
    mkdir ./csv_results
fi
if [ ! -d "./results" ]; then
    mkdir ./results
fi
if [ ! -d "./test_results" ]; then
    mkdir ./test_results
fi
model_name=MESTIM

root_path_name=./data/exchange_rate
data_path_name=exchange_rate.csv
model_id_name=exchange_rate
data_name=custom

random_seed=2024
dconv=2
for seq_len in 96
do
    for pred_len in 96 
    do
        for fc_drop in 0.1 
        do
            for e_fact in 1 2
            do
                for top_k in  2 
                do
                    for n2 in 32 
                    do
                        for n1 in  256 
                        do
                            for learning_rate in 0.001 
                            do
                                for batch_size in 16 
                                do
                                    for d_state in 64 
                                    do
                                        python -u  run_longExp.py \
                                        --random_seed $random_seed \
                                        --is_training 1 \
                                        --root_path $root_path_name \
                                        --data_path $data_path_name \
                                        --model_id $model_id_name_$seq_len'_'$pred_len \
                                        --model $model_name \
                                        --data $data_name \
                                        --features M \
                                        --seq_len $seq_len \
                                        --pred_len $pred_len \
                                        --enc_in 8 \
                                        --n1 $n1 \
                                        --n2 $n2 \
                                        --dropout $fc_drop\
                                        --dconv $dconv \
                                        --d_state $d_state\
                                        --e_fact $e_fact\
                                        --des 'Exp' \
                                        --lradj 'type3'\
                                        --pct_start 0.3\
                                        --train_epochs 10\
                                        --top_k $top_k\
                                        --batch_size $batch_size \
                                        --learning_rate $learning_rate\
                                        --devices 0 \
                                        --itr 1 >logs/LongForecasting/exchange_rate/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$n1'_'$n2'_'$fc_drop'_'$d_state'_'$dconv'_'$e_fact'_'$top_k'_'$learning_rate'_'$batch_size.log 
                                    done
                                done
                            done
                        done
                    done
                done
            done 
        done           
    done
done
