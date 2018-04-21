#!/bin/bash
PROBLEM=obs_int
MODEL=transformer
HPARAMS=transformer_small

USER_DIR=$PWD
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --t2t_usr_dir=$USER_DIR \
  --tmp_dir=$TMP_DIR

# Train
t2t-trainer \
  --data_dir=$DATA_DIR \
  --eval_steps=3 \
  --hparams_set=$HPARAMS \
  --local_eval_frequency=100 \
  --model=$MODEL \
  --output_dir=$TRAIN_DIR \
  --problems=$PROBLEM \
  --t2t_usr_dir=$USER_DIR \
  --train_steps=2000 \
  --worker_gpu_memory_fraction=0.75

# Decode
t2t-decoder \
  --data_dir=$DATA_DIR \
  --decode_from_file=sample.obs \
  --decode_hparams="beam_size=4,alpha=0.6" \
  --decode_to_file=sample.int \
  --hparams_set=$HPARAMS \
  --model=$MODEL \
  --output_dir=$TRAIN_DIR \
  --problems=$PROBLEM \
  --t2t_usr_dir=$USER_DIR

# See the transformations
cat sample.obs
echo '->'
cat sample.int
