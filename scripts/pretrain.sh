#!/bin/bash
export LANG="en_US.UTF-8"
ROOT_DIR=../
PYTHON_FILE=$ROOT_DIR/pre-train/main.py
DATA_DIR=$ROOT_DIR/data/FSQP
OUTPUT_DIR=$ROOT_DIR/pre-train/outputs
mkdir -p $OUTPUT_DIR
python $PYTHON_FILE --data_dir $DATA_DIR \
                    --output_dir $OUTPUT_DIR
