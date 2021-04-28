---
title: "DockerによるGPU機械学習・Jupyter Notebook 環境作成方法のメモ" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["Docker", "機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
---

機械学習の実験環境をチームの人と共有したいということは、多々あると思います。  
その際に、バージョンの問題が発生したりして実験環境が再現できないということは、できるだけ避けたいものです。

このようなことを避けるため、Dockerを用いることは、必須と言えるでしょう！  
また、最近は、VSCodeでコンテナにアクセスしてファイル変更もできるため、Dockerの使用して、プログラムが書きづらいということもないです。

セキュリティは考えていないなどいたらない部分があると思うので、ベストプラクティスがあれば教えて欲しいです。。。
# 環境作成

Dockerによる機械学習環境の作成方法の一例を示します。  
以下のことができれば、必要最低限の環境になると思います。
- Jupyter notebookを使用できること
- Pythonのライブラリを管理できること
- GPUが使用できること

早速、解説していきます。

```
-.docker
    |- Dockerfile
    |- requirements.txt
    |jupyter_notebook_config.py

-docker-compose.yml
```

requirements.txtは、pythonのライブラリに関する情報をまとめたものです。  
以下のコマンドで作成できます。

```
pip freeze > requirements.txt
```

jupyter notebookに関する設定も、ファイルにまとめています。これは、適宜変更してください。

```python:jupyter_notebook_config.py
import os
from IPython.lib import passwd

c = c
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = int(os.getenv('PORT', 8888))
c.NotebookApp.open_browser = False

if 'PASSWORD' in os.environ:
  password = os.environ['PASSWORD']
  if password:
    c.NotebookApp.password = passwd(password)
  else:
    c.NotebookApp.password = ''
    c.NotebookApp.token = ''
  del os.environ['PASSWORD']
```

Dockerfileは、以下の通りです。これも作りたい環境に合わせて適宜変更してください。  
多分、インストールしているライブラリは、ぐちゃぐちゃです。

```docker:Dockerfile
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y build-essential vim \
    wget curl git zip gcc make openssl \
    libssl-dev libbz2-dev libreadline-dev \
    libsqlite3-dev python3-tk tk-dev python-tk \
    libfreetype6-dev libffi-dev liblzma-dev -y

RUN git clone https://github.com/yyuu/pyenv.git /root/.pyenv
ENV HOME  /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv --version
RUN pyenv install 3.8.6
RUN pyenv global 3.8.6
RUN python --version
RUN pyenv rehash

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# JupyterNotebookのパスワード
RUN mkdir $HOME/.jupyter
COPY jupyter_notebook_config.py $HOME/.jupyter/
ENV PASSWORD password

WORKDIR /workspace
```

最後に、以上のファイルを動かす、docker-composeファイルです。
データは、別途管理していることもあるので、volumeを別に設定しています。
commandには、jupyterの起動コマンドを設定しており、設定したportでアクセスできます。例えば、http://localhost:18082です。

```yaml:docker-compose.yml
version: '2.3'

services:
  ml-dev:
    build: .docker
    container_name: ml-env
    image: ml-env
    volumes: 
      - $PWD:/workspace
      - /data:/data
    command: 'jupyter-lab --allow-root --port 18082 --ip 0.0.0.0 --no-browser'
    ports:
      - 18081-18090:18081-18090
    runtime: nvidia
```

ここでは、18081-18090の範囲でportを開いているので、例えばtensorboardを使用したい場合、コンテナ内で、以下のコマンドで使用できます。
```
tensorboard --logdir <logpath> --port 18085 --host=0.0.0.0
```

以上がファイル構成です。docker-compose.yalがあるディレクトリで、
```
docker-compose up -d
```
とすれば、環境作成終了です！

オレオレ環境なので、より良い環境作成方法があったら、ぜひ教えてください！


