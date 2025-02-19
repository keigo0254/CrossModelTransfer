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

## Original Codes
https://github.com/mlfoundations/task_vectors

https://github.com/gortizji/tangent_task_arithmetic

https://github.com/kyrie-23/linear_task_arithmetic
