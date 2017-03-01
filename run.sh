#!/bin/bash

#source ~/tf/whl/tmp/bin/activate;
source /home/linchao/tf/whl/02.25.2017_py2/bin/activate;
hostname=`hostname`
echo "Running on $hostname"

stage=$1
case $stage in
    "train")
        ;;
    *)
        model_ckpt_path=$2
esac

gpu_id=`python gpustat.py $hostname $stage`
export CUDA_VISIBLE_DEVICES=$gpu_id;
echo "Using GPU $CUDA_VISIBLE_DEVICES"

python -m yt8m.main \
    --stage=$stage \
    --model_ckpt_path=$model_ckpt_path \
    --config_name="BaseConfig"


#rm -rf $train_dir && mkdir $train_dir
#python -m yt8m.train \
    #--train_data_pattern='/data/uts700/linchao/yt8m/data/train/train*.tfrecord' \
    #--frame_features=True \
    #--model=$model \
    #--feature_names="rgb" \
    #--feature_sizes="1024" --batch_size=256 \
    #--train_dir=$train_dir

#python -m yt8m.eval \
    #--eval_data_pattern='/data/uts700/linchao/yt8m/data/validate/validate*.tfrecord' \
    #--frame_features=True \
    #--model=$model \
    #--feature_names="rgb" \
    #--feature_sizes="1024" \
    #--train_dir=$train_dir  \
    #--run_once=True

#python inference.py \
    #--input_data_pattern='/data/uts711/linchao/yt8m/test/test/test*.tfrecord' \
    #--frame_features=True \
    #--model=FrameLevelLogisticModel \
    #--feature_names="rgb" \
    #--feature_sizes="1024" \
    #--batch_size=1280 \
    #--train_dir="/data/D2DCRC/linchao/YT/log" \
    #--output_file=predictions.csv
