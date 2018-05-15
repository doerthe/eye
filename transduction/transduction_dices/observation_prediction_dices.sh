#!/bin/bash
PROBLEM=observation_prediction_dices
MODEL=transformer
HPARAMS=transformer_small

USER_DIR=$PWD
DATA_DIR=$HOME/t2t_data/$PROBLEM
TMP_DIR=/tmp/t2t_datagen/$PROBLEM
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
  --eval_steps=10 \
  --hparams_set=$HPARAMS \
  --local_eval_frequency=100 \
  --model=$MODEL \
  --output_dir=$TRAIN_DIR \
  --problem=$PROBLEM \
  --t2t_usr_dir=$USER_DIR \
  --train_steps=2000 \
  --worker_gpu_memory_fraction=0.75

# Decode
t2t-decoder \
  --data_dir=$DATA_DIR \
  --decode_from_file=sample_dices.observation \
  --decode_hparams="beam_size=3,alpha=0.6,return_beams=True,write_beam_scores=True" \
  --decode_to_file=sample_dices.prediction \
  --hparams_set=$HPARAMS \
  --model=$MODEL \
  --output_dir=$TRAIN_DIR \
  --problem=$PROBLEM \
  --t2t_usr_dir=$USER_DIR \
  --worker_gpu_memory_fraction=0.75