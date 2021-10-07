---
title: "AWS CDKをDocker環境で使用する (Python)" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["ml", "python"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

こんにちは！鷲崎([@kwashizzz](https://twitter.com/kwashizzz))です。
普段は、スポーツ分析や音声処理、AWSでの推論環境構築などやっています。宣伝になりますが、もし、モデル開発や推論環境構築でお困りのことがありましたら、連絡いただければ、相談にのれます！ 

機械学習の推論環境構築で、重用するのがAWS CDKです。機械学習エンジニアでも、慣れたPythonを用いて、AWSのサービスのIoCが可能になります。
今回は、このAWS CDKをDocker環境で使用する方法について説明します。

[API Reference](https://docs.aws.amazon.com/cdk/api/latest/docs/aws-construct-library.html)を見れば、使用するサービスに関する記述は書いてあります。

# ファイル構成

- .aws
    |- config
    |- credentials

- .docker
    |- Dockerfile
    |- requirements.txt

- .gitignore
- .docker-compose.yml

## 1. AWSのクレデンシャル

AWSのサービスをデプロイするため、適切な権限を与えたクレデンシャルを`.aws`ディレクトリ配下に置く必要があります。以下の2つの設定を参考にしてください。また、`aws cli`を用いて、`aws configure`からも作成できます。

```
# .aws/config
[default]
region=ap-northeast-1
output=jso
```

```
# .aws/credentials
[default]
aws_access_key_id = XXXXXXXXXXXXXXXX
aws_secret_access_key = XXXXXXXXXXXXXXXXXXXXXXXXXXX
```

ここで重要なことが、gitの管理で、これらのcredentialを無視することです。Webに公開は、絶対ダメです。
そこで、`.gitignore`ファイルに、以下を追記します。
```
.aws/
```

## 2. cdkで使用するライブラリ群

`.docker/requirements.txt`に、使用するライブラリを記載します。以下、例を示します。
推論環境を作成するために、`s3`や、`step functions`, `lambda`、`batch`は多用しています。

```
boto3==1.17.110

aws-cdk.assets==1.120.0
aws-cdk.aws-apigateway==1.120.0
aws-cdk.aws-applicationautoscaling==1.120.0
aws-cdk.aws-autoscaling==1.120.0
aws-cdk.aws-cloudformation==1.120.0
aws-cdk.aws-cloudfront==1.120.0
aws-cdk.aws-cloudwatch==1.120.0
aws-cdk.aws-codebuild==1.120.0
aws-cdk.aws-codecommit==1.120.0
aws-cdk.aws-dynamodb==1.120.0
aws-cdk.aws-ec2==1.120.0
aws-cdk.aws-ecr==1.120.0
aws-cdk.aws-ecr-assets==1.120.0
aws-cdk.aws-ecs==1.120.0
aws-cdk.aws-iam==1.120.0
aws-cdk.aws-lambda==1.120.0
aws-cdk.aws-s3==1.120.0
aws-cdk.aws-stepfunctions==1.120.0
aws-cdk.aws-stepfunctions-tasks==1.120.0
aws-cdk.core==1.120.0
aws-cdk.aws-batch==1.120.0
```

## 3. Dockerファイル

`.docker/Dockerfile`を、以下のように設定します。

```dockerfile
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y build-essential vim \
    wget curl git zip gcc make openssl \
    libssl-dev libbz2-dev libreadline-dev \
    libsqlite3-dev python3-tk tk-dev python-tk \
    libfreetype6-dev libffi-dev liblzma-dev libsndfile1 -y

# cdkをインストールするため、npmをインストールします。
RUN apt-get install curl
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash
RUN apt-get install -y nodejs
RUN node -v
RUN npm -v

# pythonをインストールします。使用したいバージョンを選択してください。
RUN git clone https://github.com/yyuu/pyenv.git /root/.pyenv
ENV HOME  /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv --version
RUN pyenv install 3.8.10
RUN pyenv global 3.8.10
RUN python --version
RUN pyenv rehash
RUN pip install --upgrade pip

# ライブラリをインストールします。
COPY requirements.txt .
RUN pip install -r requirements.txt

# aws cdk をインストールします。最新のバージョンにアップデートした方が良いです。
RUN npm install -g aws-cdk@1.120.0

WORKDIR /workspace
```

## 4. docker-compose.yml

`docker-compose`ファイルです。もしかしたら、`sleep`せずに、`tty`にした方が良いかもしれません。

```yaml
version: '2.3'

services:
  aws-cdk-client:
    build: .docker
    container_name: ml-aws-cdk-client
    image: ml-aws-cdk-clinet
    volumes: 
      - $PWD:/workspace
      - ./.aws/:/root/.aws/
    command:  /bin/sh -c "while :; do sleep 10; done"
    ports:
      - 18041-18050:18041-18050
```

# 起動してデプロイするまで

まず、AWS CDKを使用する環境をDockerで構築します。ここでの`WORKDIR`は`/workspace`としています。

```
docker-compose up -d
```

vs codeなどを用いて、コンテナ内で、以下の作業を行います。

### 初回実行時 は、aws cdkのプロジェクトを作成します。

プロジェクトを実行するディレクトリを作成します。プロジェクトに合わせて、ディレクトリ名を変更してください。
```
mkdir app
cd app
```

pythonでaws cdkの環境を作成します。
```
cdk init --language python
```

CDKを環境に初めてデプロイする際に、「ブートストラップスタック」をインストールする必要があります。
```
cdk synth
cdk bootstrap
```


### デプロイ作業

以下のコマンドでデプロイできます。

```
cdk deploy
```

もし、作成したスタックを削除したい場合は、以下のコマンドで削除できます。

```
cdk destroy
```

### 便利コマンド

cdkデプロイ時に、今回の例の場合、`app/app.py`が実行されます。この時、引数を渡したい場合は、`-c`で渡すことができます。

```
cdk deploy -c stage=dev
```

`app/app.py`では、以下のようにして、読み込むことができます。

```python
from aws_cdk import core
app = core.App()
stage = app.node.try_get_context('stage') 
```

また、`app/app.py`に複数のスタックを定義してデプロイする場合、エラーがでます。その時は、

```
cdk deploy --all
```

とすれば良いです。もちろん、NestedStackを使用した方が良いと思います。

デフォルトでは、`app/app.py`が実行されます。しかし、他のファイルの実行によりデプロイしたい場合もあると思います。その時は、以下のようにして、ファイルを指定すれば良いです。ファイルの指定というよりは、デプロイ時に実行するコマンドになります。

```
cdk deploy --all "python other_file.py --stage dev"
```


Lambdaなどで、cdkを実行したい場合、リソース作成時に、作成して良いか確認される動作に困ることがあります。その時は、強制実行オプションがあります。

```
cdk deploy --require-approval never
```

もちろん、destryoにもあります。

```
cdk destroy --force
```


# まとめ

以上、簡単にですが、DockerでAWS CDK実行環境の構築と、デプロイ方法でした。
この記事も含め、機械学習エンジニアが推論環境を楽に構築するTipsの記事を今後も、作成していきたいと思います。よろしくお願いします。