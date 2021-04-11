#!/bin/bash

DATA_DIR="data"

dname="Thunderbird"
gpu="0"      # The GPU you want to use
nw=32
tp=0.8
pr=-1
lsstep=50

wsize=10 # window size
vsize=5 # vector size
seqlen=16 # maximum sequence length
ochannel=10 # the output channel size of TextCNN
lstmw=64 # the width of lstm
lstmd=8 # the depth of lstm
dp=0.5 # dropout

opt="SGD"
lr=1e-1       # Learning rate for model parameters
sm=0.9
ts=-1      # Number of training steps (counted as parameter updates)
ws=320      # Learning rate warm-up steps
tepoch=3
bsize=1000    # Batch size
wd=1e-2      # Weight decay

seed=42    # Seed for randomness
expname=unilog-v2-${dname}-tp${tp}-pr${pr}-opt${opt}-lr${lr}-ws${wsize}-vs${vsize}-sl${seqlen}-oc${ochannel}-lw${lstmw}-ld${lstmd}-dp${dp}-ts${ts}-ws${ws}-te${tepoch}-bs${bsize}-wd${wd}-seed${seed}

python -u run.py \
  --gpu ${gpu} \
  --n_worker ${nw} \
  --do_train \
  --evaluate_during_training \
  --do_eval \
  --test_all \
  --do_lower_case \
  --data_dir $DATA_DIR/$dname \
  --dataset_name $dname \
  --train_percent $tp \
  --positive_rate ${pr} \
  --output_dir checkpoints/${expname}/ --overwrite_output_dir \
  --logging_steps $lsstep --save_steps $lsstep \
  --window_size ${wsize} \
  --vector_size ${vsize} \
  --max_seq_len ${seqlen} \
  --textcnn_out_channels $ochannel \
  --bilstm_width $lstmw \
  --bilstm_depth $lstmd \
  --dropout $dp \
  --optimizer $opt \
  --learning_rate ${lr} --sgd_momentum ${sm} --weight_decay ${wd} \
  --max_steps ${ts} --warmup_steps ${ws} \
  --num_train_epochs ${tepoch} \
  --per_gpu_train_batch_size ${bsize} \
  --per_gpu_eval_batch_size ${bsize} \
  --seed ${seed} \
  --amp 0 \
  --expname ${expname} \
  > logs/${expname}.log
  #| tee log.txt
  #--comet \
  #2>&1 &
  #--fp16 \

# MODEL_NAME     DATASET_NAME     tp     pr     opt     lr     wsize     vsize     seqlen     ochannel     lstmw     lstmd     dp     ts     ws     tepoch     bsize     wd     seed

# unilog     BGL     0.8     0.2     SGD     1e-1     10     5     16     10     64     10     0.5     -1     320     10     1000     1e-2     42

# unilog     HDFS     0.8     -1     SGD     1e-1     50     20     50     10     64     8     0.5     -1     320     10     1000     1e-2     42

# unilog     OpenStack     0.8     0.2     SGD     1e-1     10     5     16     10     64     8     0.5     -1     320     10     1000     1e-2     42

# unilog     Thunderbird     0.8     0.2     SGD     1e-1     10     5     16     10     64     8     0.5     -1     320     10     1000     1e-2     42
