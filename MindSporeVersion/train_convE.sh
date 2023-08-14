#/usr/bin/sh

time1="log//"`date "+%d_%m_%Y_%H:%M:%S"`".log"
echo $time1
# CUDA_VISIBLE_DEVICES=1 python run.py \
#     -score_func conve \
#     -opn sub \
#     -ker_sz 5 \
#     -batch 128

# CUDA_VISIBLE_DEVICES=0 python run.py \
#     -score_func conve \
#     -opn mult \
#     -batch 128
# FB15k-237
# WN18
# CUDA_VISIBLE_DEVICES=0 python run.py \
#     -data WN9 \
#     -score_func conve \
#     -opn mult \
#     -batch 2

CUDA_VISIBLE_DEVICES=0 python run.py \
    -data FB15k-237 \
    -score_func conve \
    -opn mult \
    -batch 64 \
    -lr 0.00001 \
    -epoch 500 \
    -rnn_layers 1 \
    -rnn_model lstm