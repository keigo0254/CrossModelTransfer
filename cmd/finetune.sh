#!bin/bash
# usage: nohup bash cmd/finetune.sh > log/finetune.log &
function finetune()
{
    python src/finetune.py \
        --train_datasets $train_datasets \
        --eval_datasets $eval_datasets \
        --model_architecture $model_architecture \
        --pretrained $pretrained \
        --finetuning_type $finetuning_type \
        --lr $lr \
        --wd $wd \
        --rank $rank \
        --alpha $alpha \
        --batch_size $batch_size \
        --grad_accum_steps $grad_accum_steps \
        --seed $seed \
        --save \
        --wandb
}

train_datasets="Cars","DTD","EuroSAT","GTSRB","MNIST","RESISC45","SUN397","SVHN"
eval_datasets="Cars","DTD","EuroSAT","GTSRB","MNIST","RESISC45","SUN397","SVHN"
model_architecture=ViT-B-32
pretrained=openai
finetuning_type=full # full, linear, lora
lr=1e-05
wd=0.1
rank=16
alpha=32
batch_size=128
grad_accum_steps=1 # 1 for ViT-B-32, 2 for ViT-B-16 8 for ViT-L-14
seed=2025
finetune $train_datasets $eval_datasets $model_architecture $pretrained $finetuning_type $lr $wd $rank $alpha $batch_size $grad_accum_steps $seed
