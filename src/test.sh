#!/bin/bash
# This script trains a deepspeech model in tensorflow with sorta-grad.
# usage ./train.sh  or  ./train.sh dummy


clear
cur_dir=$(cd "$(dirname $0)";pwd)
# echo ${cur_dir}
export PYTHONPATH=${cur_path}:/home/matrix/inteltf/:$PYTHONPATH
# echo $PYTHONPATH
#export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64/:$LD_LIBRARY_PATH

# environment variables
unset TF_CPP_MIN_VLOG_LEVEL
# export TF_CPP_MIN_VLOG_LEVEL=1

# clear
echo "-----------------------------------"
echo "Start testing"

nchw=True    # True or False
engine="tf"  # tf, cudnn

config_check=`test "${nchw}" = "False" && test "${engine}"x = "tf"x -o "${engine}"x = "cudnn"x && echo 'OK'`
echo "config_check: "$config_check

if [[ ${config_check}x != "OK"x ]];then
    echo "unsupported configuration conbimation"
    exit -1
fi

python deepSpeech_test.py --eval_data 'test' --nchw ${nchw} --engine ${engine} --run_once True
echo "Done"


