# Task Arithmetic [Ilharco+, [ICLR'23](https://openreview.net/forum?id=6t0Kwf8-jrj)]

## Clone Repository
```code
git clone https://github.com/kawakera-lab/TaskArithmetic.git    # HTTPS
git clone git@github.com:kawakera-lab/TaskArithmetic.git        # SSH
```

## Download Datasets
- [RESISC45](https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&id=5C5E061130630A68%21107&cid=5C5E061130630A68&parId=root&parQt=sharedby&o=OneUp): Download to /home/user/
- [ImageNet-1k](https://image-net.org/index.php): Download and extract to /home/user/dataset/
    ```code
    cd /home/user/dataset
    mkdir ILSVRC2012_img_train
    tar -xvf ILSVRC2012_img_train.tar -C ILSVRC2012_img_train/
    tar -zxvf ILSVRC2012_devkit_t12.tar.gz
    ```
- Stanford Cars, DTD, EuroSAT, SUN397: Run download_sample.sh. However, you need to enter your own Kaggle API key and rename the file to download.sh before running.
    ```code
    bash download.sh
    ```
- Organize dataset directory structure.
    ```code
    cd ~/TaskArithmetic         # Move to repository directory
    python split_dataset.py     # Some users may need to run python3 split_dataset.py
    ```

## Environment Setup
To ensure experiment reproducibility, we use Docker for environment setup.
```code
cd ~/TaskArithmetic
bash docker.sh build    # Create image, first time only
bash docker.sh shell    # Start container
```

## Fine-Tuning
```code
nohup bash cmd/finetune.sh > log/finetune.log
```

## Task Arithmetic
```code
nohup bash cmd/arithmetic.sh > log/arithmetic.log
```

## Watch Features
```code
nohup bash cmd/feature.sh > log/feature.log
```

## Cross-Model-Tuning of Task Vectors
```code
nohup bash cmd/orthogonal_finetuning.sh > log/orthogonal_finetuning.log
```

## おすすめ論文 (外に公開する場合は消す)
### タスク算術系
* [Editing Models with Task Arithmetic](https://openreview.net/forum?id=6t0Kwf8-jrj)
* [Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d28077e5ff52034cd35b4aa15320caea-Abstract-Conference.html)
* [Fine-Tuning Attention Modules Only: Enhancing Weight Disentanglement in Task Arithmetic](https://openreview.net/forum?id=dj0TktJcVI)
* [Mastering Task Arithmetic: τJp as a Key Indicator for Weight Disentanglement](https://openreview.net/forum?id=1VwWi6zbxs)
* [Model Merging by Uncertainty-Based Gradient Matching](https://openreview.net/forum?id=D7KJmfEDQP)

### アライメント系
* [Git Re-Basin: Merging Models modulo Permutation Symmetries](https://openreview.net/forum?id=CQsmMYmlP5T)
* [ZipIt! Merging Models from Different Tasks without Training](https://openreview.net/forum?id=LEYUkvdUhq)
* [Foldable SuperNets: Scalable Merging of Transformers with Different Initializations and Tasks](https://arxiv.org/abs/2410.01483)
* [Transformer Fusion with Optimal Transport](https://openreview.net/forum?id=LjeqMvQpen)

### サーベイ論文
* [Deep Model Fusion: A Survey](https://arxiv.org/abs/2309.15698)

## Original Codes
https://github.com/mlfoundations/task_vectors

https://github.com/gortizji/tangent_task_arithmetic

https://github.com/kyrie-23/linear_task_arithmetic
