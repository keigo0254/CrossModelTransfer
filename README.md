# Task Arithmetic [Ilharco+, [ICLR'23](https://openreview.net/forum?id=6t0Kwf8-jrj)]

## Dataset
1. [リンク](https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&id=5C5E061130630A68%21107&cid=5C5E061130630A68&parId=root&parQt=sharedby&o=OneUp)からNWPU-RESISC45.rarをダウンロードし，/home/user/に置く．
1. download_sample.shの内容を各自の環境に合わせて変更し，実行する．**APIキーを公開しないようにすること！！！**
1. データセットを分割するために，split_dataset.pyを実行する．

## Environment
1. モデルを保存する用のディレクトリmodelを/home/user/に作成する．
1. Dockerイメージを作成し，コンテナに入る．
    ```code
    bash docker.sh build            # イメージの作成
    bash docker.sh shell            # コンテナの作成，コンテナに入る
    exit                            # コンテナから出る
    docker start <container_name>   # コンテナを起動
    docker attach <container_name>  # コンテナに入る
    docker stop <container_name>    # コンテナを止める
    ```


## Original Codes
https://github.com/mlfoundations/task_vectors

https://github.com/gortizji/tangent_task_arithmetic

https://github.com/kyrie-23/linear_task_arithmetic