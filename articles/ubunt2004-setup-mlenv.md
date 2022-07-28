---
title: "Ubuntu20.04に機械学習(GPU)環境を設定する方法" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["python", "ml"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

# はじめに

こんにちは、[わっしー](https://twitter.com/kwashizzz)です。
本記事では、Ubuntu20.04にGPUありの機械学習環境を設定する方法を紹介します。一応、AzureのVMで動作確認しました。
よく躓く、CUDA、pythonの導入部分も記載しています。

早速、方法を紹介します。

# 1. sshでサーバ接続

```
ssh -i ./*.cer azureuser@xx.xxx.xxx.xxx
```

もし、`Permissions 0644 for ‘xxx.key’ are too open.`がでる場合は、

```
chmod 600 xxx.key
```
で権限を変更してください。

もし、Vscodeでサーバに入りたい場合は、以下のように設定してください。

1. Remote-ssh拡張機能をインストール  
`Enable Remote Command`の許可を設定する必要があります。
また、未確認ですが、プレリリースバージョンへの切り替えも必要かもしれません。

2. `.ssh/config`に以下の設定を追加 (AzureのVMサーバ用です)

```
Host SampleGpuServer
    HostName xx.xxx.xxx.xxx
    User azureuser
    IdentityFile ~/.ssh*cer
    RemoteCommand sudo su -
    RequestTTY true
    ForwardAgent yes
```

# 2. ubuntu環境設定
適宜、必要なライブラリをインストールしてください。

```
apt-get update
apt-get install -y build-essential vim \
    wget curl git zip gcc make openssl \
    libssl-dev libbz2-dev libreadline-dev \
    libsqlite3-dev python3-tk tk-dev python-tk \
    libfreetype6-dev libffi-dev liblzma-dev libsndfile1 ffmpeg -y
```
# 3. GPU関連の環境設定

まずは、gpuが備わっているのか確認します。

```
lspci | grep -i NVIDIA
```

次に、CUDAドライバーのインストールです。
[Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)のホームページから、インストール方法を選択してください。

OS情報の確認は、`cat /etc/os-release` でできます。

インストールが終わったら、`nvidia-smi`でCUDAが入っていることを確認してください。

# 4. pythonのインストール

今回は、pyenvでpythonの導入を行います。個人的には、pyenvでpythonのバージョンを選択し、venvで仮想環境の作成を行います。

まずは、pyenvによるpythonのインストールです。

```
git clone https://github.com/yyuu/pyenv.git /root/.pyenv
echo "export HOME=/root" >> ~/.bashrc 
echo "export PYENV_ROOT=$HOME/.pyenv" >> ~/.bashrc 
echo "export PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH" >> ~/.bashrc 
source ~/.bashrc
pyenv --version
pyenv install 3.9.13
pyenv global 3.9.13
python --version
pyenv rehash
```

# 5. Pythonの仮想環境設定・ライブラリインストール

以下のようにして、pythonの仮想環境を設定できます。

```
git clone https://github.com/xxxxxxx/xxxxxxxxxxx.git
cd ./xxxxxxxxxxxx
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt 

# pytorch
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# 確認
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

# 最後に

以上、Ubuntu20.04での機械学習環境の構築に関してでした。AzureのVMで環境構築したため、良い機会なのでまとめました。

---

画像・自然言語・音声に関する機械学習の研究開発やMLOpsを行っています。もし、機械学習に関して、ご相談があれば、[@kwashizzz](https://twitter.com/kwashizzz)のアカウントまでDMしてください！
これまでの、[機械学習記事のまとめ](https://zenn.dev/kwashizzz/articles/my-ml-articles-summary)です。

