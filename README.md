# Task Arithmetic

タスク算術[Ilharco+, [ICLR'23](https://openreview.net/forum?id=6t0Kwf8-jrj)
]を実装し，アレンジを加えたコードです．


## Dataset

1. RESISC45を/home/user/にダウンロード: https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&id=5C5E061130630A68%21107&cid=5C5E061130630A68&parId=root&parQt=sharedby&o=OneUp
1. Kaggleへ登録してユーザ名とAPIキーを取得 (参考: https://qiita.com/makaishi2/items/0d9ca522733a6d9b1ca8)
1. 環境に合わせてdownload_sample.shとsplit_dataset.pyを変更する
1. APIキーをアップロードしないように，download_sample.shのファイル名をdownload.shに変更しておく
1. 以下のコードを実行し，ダウンロード・分割を行う
```code
bash download.sh
python split_dataset.py
```

参考: https://github.com/mlfoundations/task_vectors/issues/1


## Environment

あらかじめデータセット，モデルを保存する用のフォルダをどこかに作成しておいてください(想定はdataset, model)．
Dockerfile内のUSER_NAME, GROUP_NAMEを自由に設定してください．

wandbでのロギングのため，.env_exampleのようにAPIキーを入れて，.envファイルを作成してください．

**絶対にAPIキーを公開しないようにしてください．**

```code
bash docker.sh build
bash docker.sh shell
```

## Original Codes
https://github.com/mlfoundations/task_vectors

https://github.com/gortizji/tangent_task_arithmetic

https://github.com/kyrie-23/linear_task_arithmetic