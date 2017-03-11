#!/bin/bash

run_id=$1
model_id=$2
dst_dir=/data/D2DCRC/linchao/YT/log/${run_id}/src/
if [ ! -d "$dst_dir"  ]; then
    cd /data/D2DCRC/linchao/YT/log/${run_id}; tar xf src.tar
fi
cd $dst_dir
./run.sh eval /data/D2DCRC/linchao/YT/log/${run_id}/model.ckpt-${model_id} > /data/D2DCRC/linchao/YT/log/${run_id}/model.ckpt-$model_id.validate.log 2>&1
