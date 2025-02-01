# Task Arithmetic

タスク算術[Ilharco+, [ICLR'23](https://openreview.net/forum?id=6t0Kwf8-jrj)
]を実装し，アレンジを加えたコードです．

## Environment

あらかじめデータセット，モデルを保存する用のフォルダをどこかに作成しておいてください(想定は/home/dataset, /home/model)．
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