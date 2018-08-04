#!/bin/bash
PROBLEM=observation_prediction_bodies
MODEL=transformer
HPARAMS=transformer_small

USER_DIR=$PWD
DATA_DIR=/tmp/t2t_data/$PROBLEM
TMP_DIR=/tmp/t2t_datagen/$PROBLEM
TRAIN_DIR=/tmp/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --t2t_usr_dir=$USER_DIR \
  --tmp_dir=$TMP_DIR

# train with Adam for 3600 steps
t2t-trainer \
  --data_dir=$DATA_DIR \
  --eval_steps=10 \
  --eval_throttle_seconds=30 \
  --hparams_set=$HPARAMS \
  --model=$MODEL \
  --output_dir=$TRAIN_DIR \
  --problem=$PROBLEM \
  --t2t_usr_dir=$USER_DIR \
  --train_steps=3600

# train with SGD for 3600 steps
t2t-trainer \
  --data_dir=$DATA_DIR \
  --eval_steps=10 \
  --eval_throttle_seconds=30 \
  --hparams="optimizer=SGD" \
  --hparams_set=$HPARAMS \
  --model=$MODEL \
  --output_dir=$TRAIN_DIR \
  --problem=$PROBLEM \
  --t2t_usr_dir=$USER_DIR \
  --train_steps=7200

# decode
t2t-decoder \
  --data_dir=$DATA_DIR \
  --decode_from_file=test_bodies.observation \
  --decode_hparams="beam_size=1,alpha=0.6" \
  --decode_to_file=test_bodies.prediction \
  --hparams_set=$HPARAMS \
  --model=$MODEL \
  --output_dir=$TRAIN_DIR \
  --problem=$PROBLEM \
  --t2t_usr_dir=$USER_DIR
