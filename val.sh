#!/bin/bash

run_id=$1
model_id=$2
prefix="/data/D2DCRC/linchao/YT/log/"
#prefix="/home/linczhu/yt/log/"
dst_dir=${prefix}/${run_id}/src/
if [ ! -d "$dst_dir"  ]; then
    cd ${prefix}/${run_id}; tar xf src.tar
fi
cd $dst_dir
./run.sh eval ${prefix}/${run_id}/model.ckpt-${model_id} > ${prefix}/${run_id}/model.ckpt-$model_id.validate.log 2>&1
