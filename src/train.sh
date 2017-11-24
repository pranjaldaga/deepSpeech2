#!/bin/bash

clear
cur_dir=$(cd "$(dirname $0)";pwd)
# echo ${cur_dir}
export PYTHONPATH=${cur_path}:$PYTHONPATH
# echo $PYTHONPATH
# export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64/:$LD_LIBRARY_PATH

# environment variables
unset TF_CPP_MIN_VLOG_LEVEL
# export TF_CPP_MIN_VLOG_LEVEL=2

# clear
echo "-----------------------------------"
echo "Start training"

dummy=False   # True or False
nchw=False    # True or False
debug=False   # True or False
engine="tf"   # tf, cudnn

# echo $dummy

config_check=`(test "${nchw}" = "False") && (test "${engine}"x = "tf"x -o "${engine}"x = "cudnn"x) && echo 'OK'`
echo "config_check: "$config_check

if [[ ${config_check}x != "OK"x ]];then
    echo "unsupported configuration combination"
    exit -1
fi

model_dir='../models/librispeech/train'
data_dir='../data/LibriSpeech/audio/processed/'

python deepSpeech_train.py --batch_size 32 --no-shuffle --max_steps 40000 --num_rnn_layers 7 --num_hidden 1760 --num_filters 32 --initial_lr 1e-6 --train_dir $model_dir --data_dir $data_dir --debug ${debug} --nchw ${nchw} --engine ${engine} --dummy ${dummy} 

echo "Done"
 
