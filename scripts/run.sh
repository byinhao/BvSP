#!/bin/bash
export LANG="en_US.UTF-8"

ROOT_DIR=../
PYTHON_FILE=$ROOT_DIR/BvSP/main.py
DATA_DIR=$ROOT_DIR/data/FSQP
OUTPUT_DIR=$ROOT_DIR/BvSP/outputs
MODEL_DIR=$ROOT_DIR/pre-train/outputs
mkdir -p $OUTPUT_DIR
for method_name in min_js
do
for view_num in 3
do
for few_shot_type in 1 # 1 2 5 10
do
for seed in 5 10 15 20 25
do

python main.py --seed $seed \
                --method_name $method_name \
                --view_num $view_num \
                --few_shot_type $few_shot_type \
                --data_dir $DATA_DIR \
                --output_dir $OUTPUT_DIR \
                --model_name_or_path $MODEL_DIR

done
done
done
done