# Task Arithmetic [Ilharco+, [ICLR'23](https://openreview.net/forum?id=6t0Kwf8-jrj)]

## リポジトリのクローン
```code
git clone https://github.com/kawakera-lab/TaskArithmetic.git    # HTTPS
git clone git@github.com:kawakera-lab/TaskArithmetic.git        # SSH
```

## データセットのダウンロード
- [RESISC45](https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&id=5C5E061130630A68%21107&cid=5C5E061130630A68&parId=root&parQt=sharedby&o=OneUp): /home/user/にダウンロード．
- [ImageNet-1k](https://image-net.org/index.php): /home/user/dataset/にダウンロード&解凍．
    ```code
    cd /home/user/dataset
    mkdir ILSVRC2012_img_train
    tar -xvf ILSVRC2012_img_train.tar -C ILSVRC2012_img_train/
    tar -zxvf ILSVRC2012_devkit_t12.tar.gz
    ```
- Stanford Cars, DTD, EuroSAT, SUN397: download_sample.shを実行．ただし，各自のKaggle APIキーを入力し，ファイル名をdownload.shにしてから実行すること．
    ```code
    bash download.sh
    ```
- データセットのディレクトリ構造を整える．
    ```code
    cd ~/TaskArithmetic         # リポジトリのディレクトリに移動
    python split_dataset.py     # 人によってはpython3 split_dataset.pyを実行する
    ```

## 環境構築
誰でも実験を再現できるように，Dockerを用いて環境構築を行う．
```code
cd ~/TaskArithmetic
bash docker.sh build    # イメージの作成，初回のみ
bash docker.sh shell    # コンテナの起動
```
### Dockerの操作
```code
docker ps       # コンテナの一覧
docker ps -a    # コンテナの一覧（停止中のものも含む）
docker start <container_id>     # コンテナの起動
docker attach <container_id>    # コンテナに入る
docker stop <container_id>      # コンテナの停止
docker rm <container_id>        # コンテナの削除
docker rmi <image_id>           # イメージの削除
```

## Original Codes
https://github.com/mlfoundations/task_vectors

https://github.com/gortizji/tangent_task_arithmetic

https://github.com/kyrie-23/linear_task_arithmetic
