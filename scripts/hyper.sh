#!/bin/bash
export LANG="en_US.UTF-8"
device=0
for method_name in min_js
do
for view_num in 15 # 9 11 13 15
do
for few_shot_type in 1 # 1 2 5 10
do
for bs in 8
do
for lr in 3e-4
do
for seed in 10 # 5 10 15 20 25
do

python main.py --seed $seed \
                --method_name $method_name \
                --view_num $view_num \
                --data_dir data/FSQP \
                --few_shot_type $few_shot_type \
                --device $device \
                --do_lower 1 \
                --output_dir outputs \
                --train_batch_size $bs \
                --learning_rate $lr

done
done
done
done
done
done