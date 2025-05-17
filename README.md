<div align="left">
<h1>Cross-Model Transfer of Task Vectors via Few-Shot Orthogonal Alignment</h1>

<i>Kazuhiko Kawamoto</i>, <i>Atsuhiro Endo</i>, <i>Hiroshi Kera</i>

_url_

**Abstract**

Task arithmetic enables efficient model editing by representing task-specific changes as vectors in parameter space. Task arithmetic typically assumes that the source and target models are initialized from the same pre-trained parameters. This assumption limits its applicability in cross-model transfer settings, where models are independently pre-trained on different datasets. To address this challenge, we propose a method based on few-shot orthogonal alignment, which aligns task vectors to the parameter space of a differently pre-trained target model. These transformations preserve key properties of task vectors, such as norm and rank, and are learned using only a small number of labeled examples. We evaluate the method using two Vision Transformers pre-trained on YFCC100M and LAION400M,andtest on eight classification datasets. Experimental results show that our method improves transfer accuracy over direct task vector application and achieves performance comparable to few-shot fine-tuning, while maintaining the modularity and reusability of task vectors.

</div>

### Clone Repository
```code
git clone https://github.com/kawakera-lab/CrossModelTransfer.git    # HTTPS
git clone git@github.com:kawakera-lab/CrossModelTransfer.git        # SSH
```

### Download Datasets
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

### Environment Setup
To ensure experiment reproducibility, we use Docker for environment setup.
```code
cd ~/TaskArithmetic
bash docker.sh build    # Create image, first time only
bash docker.sh shell    # Start container
```

### Fine-Tuning
```code
nohup bash cmd/finetune.sh > log/finetune.log
```

### Task Arithmetic
```code
nohup bash cmd/arithmetic.sh > log/arithmetic.log
```

### Watch Features
```code
nohup bash cmd/feature.sh > log/feature.log
```

### Cross-Model Transfer of Task Vectors
```code
nohup bash cmd/orthogonal_finetuning.sh > log/orthogonal_finetuning.log
```
