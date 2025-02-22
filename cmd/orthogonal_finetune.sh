#!bin/bash
# usage: nohup bash cmd/orthogonal_finetune.sh > log/orthogonal_finetune.log &
function orthogonal_finetune()
{
    python src/orthogonal_finetune.py \
        --train_datasets $train_datasets \
        --eval_datasets $eval_datasets \
        --model_architecture $model_architecture \
        --pretrained $pretrained \
        --pretrained_to_transfer $pretrained_to_transfer \
        --finetuning_type $finetuning_type \
        --adjust_type $adjust_type \
        --dataset_type $dataset_type \
        --lr $lr \
        --wd $wd \
        --rank $rank \
        --alpha $alpha \
        --batch_size $batch_size \
        --grad_accum_steps $grad_accum_steps \
        --seed $seed \
        --lamb $lamb \
        --epochs $epochs \
        --num_images $num_images \
        --num_augments $num_augments \
        --save \
        --wandb
}

train_datasets="Cars","DTD","EuroSAT","GTSRB","MNIST","RESISC45","SUN397","SVHN"
eval_datasets="Cars","DTD","EuroSAT","GTSRB","MNIST","RESISC45","SUN397","SVHN"
model_architecture=ViT-B-32
pretrained=laion400m_e32
pretrained_to_transfer=openai
finetuning_type=full # full, linear, lora
adjust_type=fro
dataset_type=cycle
lr=1e-05
wd=0.1
rank=16
alpha=32
batch_size=128
grad_accum_steps=1 # 1 for ViT-B-32, 2 for ViT-B-16 8 for ViT-L-14
lamb=0.25
epochs=100
num_images=100
num_augments=10
seed=2025
orthogonal_finetune $train_datasets $eval_datasets $model_architecture $pretrained $pretrained_to_transfer $finetuning_type $lr $wd $rank $alpha $batch_size $grad_accum_steps $seed $lamb $adjust_type $dataset_type $epochs $num_images $num_augments

train_datasets="EuroSAT","GTSRB","MNIST","RESISC45","SUN397","SVHN"
eval_datasets="Cars","DTD","EuroSAT","GTSRB","MNIST","RESISC45","SUN397","SVHN"
model_architecture=ViT-B-32
pretrained=laion400m_e32
pretrained_to_transfer=openai
finetuning_type=linear # full, linear, lora
adjust_type=fro
dataset_type=cycle
lr=1e-05
wd=0.1
rank=16
alpha=32
batch_size=128
grad_accum_steps=1 # 1 for ViT-B-32, 2 for ViT-B-16 8 for ViT-L-14
lamb=1.0
epochs=100
num_images=100
num_augments=10
seed=2025
orthogonal_finetune $train_datasets $eval_datasets $model_architecture $pretrained $pretrained_to_transfer $finetuning_type $lr $wd $rank $alpha $batch_size $grad_accum_steps $seed $lamb $adjust_type $dataset_type $epochs $num_images $num_augments

train_datasets="Cars","DTD","EuroSAT","GTSRB","MNIST","RESISC45","SUN397","SVHN"
eval_datasets="Cars","DTD","EuroSAT","GTSRB","MNIST","RESISC45","SUN397","SVHN"
model_architecture=ViT-B-32
pretrained=laion400m_e32
pretrained_to_transfer=openai
finetuning_type=lora # full, linear, lora
adjust_type=fro
dataset_type=cycle
lr=1e-05
wd=0.1
rank=16
alpha=32
batch_size=128
grad_accum_steps=1 # 1 for ViT-B-32, 2 for ViT-B-16 8 for ViT-L-14
lamb=1.0
epochs=100
num_images=100
num_augments=10
seed=2025
orthogonal_finetune $train_datasets $eval_datasets $model_architecture $pretrained $pretrained_to_transfer $finetuning_type $lr $wd $rank $alpha $batch_size $grad_accum_steps $seed $lamb $adjust_type $dataset_type $epochs $num_images $num_augments
