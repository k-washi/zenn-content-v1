---
title: "SORACOM AI カメラ 「SORACOM Plus Camera Basic」の初期設定〜データをAWS S3に保存するまで解説！" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["iot", "python"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

こんにちは、鷲崎です。最近、SORACOMのAI カメラ「[SORACOM Plus Camera Basic](https://soracom.jp/soracom_plus/camera_basic/)」を楽しんでいます。このカメラは、電源をつなぐだけで、通信が可能になる、エッジAIカメラです。通信部分は、SORACOM Air のセルラー通信がになってくれており、機械学習エンジニアがアルゴリズム開発に集中できるという利点があります。

他にも、以下のような素晴らしい利点があります。
1. エッジ側で処理が行えるので、画像など重いデータの送信量を減らせる
2. プログラムのデプロイがリモートで可能 (SSHでカメラ内に入る必要がない)
3. エッジから、AWS lambdaを実行することが可能

本記事では、このカメラを用いて、エッジで撮影した画像を、AWS S3に保存する方法を紹介したいと思います。

[【Ask SA!】S+ Camera Basic 入門と AWS サービスとの連携による表情解析とダッシュボード作成](https://blog.soracom.com/ja-jp/2020/06/29/asksa-spluscamera_with_aws/) という公式の記事が出ています。この記事も、AWSとの連携に関する記事として、かなり有益で、本記事も大きく影響を受けています。

# 1. 初期設定

カメラの初期設定と、カメラの遠隔操作に使用するSORACOM Mosaicに関して説明します。
SORACOM Mosaicとはエッジデバイスの管理を遠隔からすることができるWebサービスです。

1. SIMの設定
   [https://users.soracom.io/ja-jp/docs/mosaic/prep-mosaic/](https://users.soracom.io/ja-jp/docs/mosaic/prep-mosaic/) を参考に、Simを設定します。また、SORACOM Mosaic用にグループを作成し、 SORACOM Air、メタデータサービス 及び SORACOM Harvest Filesを有効化します。そして、そのグループにSIMを割り当てます。

2. SORACOM Mosaicにて、カメラ画像を取得
   [SORACOM Mosaic コンソールを利用してデバイスを操作する](https://users.soracom.io/ja-jp/docs/mosaic/use-mosaic-console/)を参考にしてください。

# 2. 開発の準備

エッジ側で動くプログラム(例えば、撮影して、画像をs3に保存するなど)を開発するため、以下の準備を行います。

1. アクセス管理 (SORACOM Access Management; SAM)にユーザを追加する
   ローカルから、デバイスにプログラムをアップデートするために必要となります。
   [SAM ユーザーを作成する](https://users.soracom.io/ja-jp/docs/sam/create-sam-user/) を参考に作成します。

2. SORACOM Napterの設定
   ローカルから、カメラにアクセスできるようにするために必要となります。
   [SORACOM Napter SORACOM Air for セルラーを使用した IoT デバイスへ簡単にセキュアにリモートアクセス](https://users.soracom.io/ja-jp/docs/napter/)を参考にしてください。

# 3. 開発環境の作成

Dockerfileでローカル開発環境を構築します。
[アルゴリズムの更新方法](https://users.soracom.io/ja-jp/docs/mosaic/develop-algorithm/)を主に参考にして、構築しました。

1. requirements.txtの作成

[SORACOM Mosaic Python module 一覧](https://users.soracom.io/ja-jp/docs/mosaic/modules/)のモジュールを記載します。

後の処理で、エラーが出たため```tflite-runtime==***```のみコメントアウトしました。

もし、開発に必要なライブラリがあったら、追記、もしくは、別途インストールをしてください。

2. Dockerfileの作成

Dockerfileを作成します。
1. コメントアウトしたtfliteを別途インストールしています。


```dockerfile: Dockerfile
FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y build-essential vim \
    wget curl git zip gcc cmake make openssl \
    libssl-dev libbz2-dev libreadline-dev \
    libsqlite3-dev python3-tk tk-dev python-tk \
    libfreetype6-dev libffi-dev liblzma-dev -y jq sudo

RUN git clone https://github.com/yyuu/pyenv.git /root/.pyenv
ENV HOME  /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv --version
RUN pyenv install 3.7.3
RUN pyenv local 3.7.3
RUN python --version
RUN pyenv rehash

# このディレクトリに作成することが重要
RUN sudo mkdir -p /opt/soracom/python/
RUN chown -R $(whoami) /opt/soracom/
RUN python -m venv /opt/soracom/python/

# pyenvの環境を構築
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# tfliteは別途ダウンロードした
RUN pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl
RUN wget https://github.com/soracom/soracom-cli/releases/download/v0.6.2/soracom_0.6.2_linux_amd64.tar.gz
RUN tar xvf ./soracom_0.6.2_linux_amd64.tar.gz
RUN cp ./soracom_0.6.2_linux_amd64/soracom /usr/local/bin/

WORKDIR /workspace
```

docker-compose.ymlファイルも作成します。

```yaml: docker-compose.yml
version: '2.3'

services:
  ai-camera-dev:
    build: .
    container_name: ai-camera-dev
    image: ai-camera-dev
    command: /bin/sh -c "while :; do sleep 10; done"
    volumes: 
      - $PWD:/workspace
    ports:
      - 18071-18080:18071-18080
```

```docker-compose up -d``` コマンドでDockerを用いた開発環境構築は終了です。

# 4. Dockerコンテナ内で開発環境を設定

VScodeのリモートエクスプローラから、開発用コンテナを選択することで、楽にコンテナ内で開発できます。
コンテナ内に入ると、soracom cliの設定と、SORACOM Mosaicデプロイツールの取得を行います。

1. 実行したいPython開発環境を設定  
   ```source /opt/soracom/python/bin/activate``` で、ライブラリをインストールしたPython開発環境を設定できます。

2. SORACOM cli の設定  
   [SORACOM CLI をインストールし SIM カードの一覧を取得する](https://users.soracom.io/ja-jp/tools/cli/getting-started/) の記事を参考にしつつ、SORACOM CLIを設定します。 ステップ1は、Dockerで環境構築時に行っているので、ステップ2より行います。

   認証情報の設定
   ```bash
   LANG=ja soracom configure
   ```

   確認のため、SIMカードの一覧を取得する
   ```
   soracom subscribers list
   ```

3. SORACOM Mosaicのデプロイツールを取得  
   ローカルで開発したプログラムをデバイスにデプロイする必要があり、そのツールが準備されています。　　

   [ここ](https://drive.google.com/file/d/12kSdp0ksraM4bUCR5avHYu_Xoes8VnFx/edit)から、デプロイツールをダウンロードしてください。そして、以下のコマンドが動作するか確認します。

   ```bash
   jq
   mosaic_deploy.sh
   ```

   もし、動作しない場合は、権限を与えれば良いかもしれません。

# 5. サンプルアルゴリズムに関して

[ここ](https://drive.google.com/file/d/16iLRfPCJmdvRh5Dbc9ETFslAPq9U5f12/edit)からダウンロードして、解凍してください。

まずは、簡単に、カメラから画像を取得する方法を説明します。

1. 画像取得用のURLを発行
   SORACOM Mosaicにて、Remote Accessというフォームより、[SORACOM Napter](https://users.soracom.io/ja-jp/docs/napter/)によるURLを発行します。設定として、USE TLSにチェックを入れ、portを8080、開発環境のIPアドレスを設定してください。IPアドレスに関しては、空のままだと、現在表示しているWebサイトを表示しているIPが設定されます。

   取得したURLに対して、以下のURLを叩けば、画像が表示されます。  
   ```bash
   https://<ip>.napter.soracom.io:<port>/v1/startCameraCapture
   https://<ip>.napter.soracom.io:<port>/v1/cameraJpegImage
   ```

   ここで取得する画像は、思ったより赤みがかっているかもしれません。しかし、これは、暗視も可能にするIRカメラを使用している影響です。精度を保証するわけではありませんが、画像が赤みがかっている場合でも、意外と機械学習モデルは認識してくれます。

2. ローカルの開発環境から画像へのアクセス  
   発行したURLを環境変数に設定します。  
   ```
   export DEVICE_INTERFACE_URI=https://{Napter で設定した URL}
   ````

   ダウンロードしたサンプルプログラムのディレクトリに移動し、サンプルプログラムを実行します。
   ```
   cd ${sample_dir}
   ./CameraApp0
   ```

   サンプルプログラムの場合、```/tmp```ディレクトリに定期的にカメラ画像が保存されます。

   サンプルプログラム内には、画像の取得ライブラリなども入っているので、基本的に```./CameraApp0.py```のプログラムを変更していけば、開発できると思います。

# 6. デプロイ
   実際のデバイスにプログラムをデプロイする方法を説明します。

   4で取得したデプロイツールを以下のコマンドで実行します。
   ```
   ./mosaic_deploy.sh -d <Device id>  /workspace/CameraApp0
   ```
   <Device id>の部分は、SORACOM Mosaicで確認できるデプロイしたいデバイスのIDです。


# 7. 開発時のTips

1. 環境変数の設定
   SORACOM MosaicのAPP(CAMERAAPP0)という項目にて、環境変数を設定できます。プログラムでは、
   ```python
   os.getenv( 'SORACOM_ENV_WAIT', 5)
   ```
   のような形で読み込めます。ローカルで確認する時は、環境変数が設定されておらず、デフォルト値が読み込まれます。

   自由に環境変数名を設定することができないため、もし、別途パラメータを注入したい場合、```SORACOM_ENV_FREE_PARAM```が使用できます。これは、辞書型で、以下のように設定できます。
   ```
   {'PARAM1':1, 'HOGEHOGE': 'test param', 'PARAM2': 'test2'}
   ```

   これ変数は、プログラムにて以下のように、読み込むことができます。
   ```
   FREE_PARAM = str(os.getenv('SORACOM_ENV_FREE_PARAM', '{}'))
   FREE_PARAM = json.loads(FREE_PARAM)

   PARAM1 = int(FREE_PARAM.get('PARAM1', 0))
   PARAM2 = str(FREE_PARAM.get('PARAM2', 'param2'))
   ```

2. ライブラリの取得
   Pythonプログラムで使用できるライブラリは、[ここ](https://users.soracom.io/ja-jp/docs/mosaic/modules/)に記載されています。しかし、要件によっては、必要なライブラリのインストールが必要なことがあると思います。そこで、PreSetupというファイルがサンプルプログラムに含まれており、このPreSetupにインストール処理を書くことで、対処できます。PreSetupは、デバイス側で```CameraApp0.py```が実行される前に、実行されるファイルです。例えば、以下のように記載できます。

   ```
   #!/bin/bash
   echo PreSetup
   pip install Keras==2.3.1
   pip freeze

   mkdir -p /opt/app
   wget -O /opt/app/ssd_mobilenet_v2_coco_quant_postprocess.tflite "https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
   ```

   ここでは、Kerasのインストールと、モデルの重みのダウンロードを行っています。これで、機械学習モデルによる予測に必要なライブラリやファイルをダウンロードできます。

# 8. s3に画像ファイルを送信する

   s3への画像送信方法は、[【Ask SA!】S+ Camera Basic 入門と AWS サービスとの連携による表情解析とダッシュボード作成](https://blog.soracom.com/ja-jp/2020/06/29/asksa-spluscamera_with_aws/) という公式の記事に記載されています。ここでは、実装中に調査したことについて説明したいと思います。

   1. s3のpresigned URLの取得  
      s3に保存するためのpresigned URLをAWS Lambdaで発行します。
      AWS Lambdaへのアクセスは、[SORACOM Funk](https://users.soracom.io/ja-jp/docs/funk/)を用いています。ここでは、AWS Lambdaの実行権限を与えたロールと実行したいARNを記載することで設定できます。

      デバイス上で実行されるS3のPresigned URL取得関数は、以下のようになります。この関数によりSORACOM Funkを介して、S3に画像ファイルを送信するためのPresigned URLを発行されます。

      ```python
      def send_req_s3_presignedurl(file_name, bucket_name):
         """
         s3のpresigenedurlを取得する
         file_name: s3に置く際のパス
         bucket_name: バケットの名前
         """
         msg = json.dumps({
                  "type": "get_upload_url",
                  "file_name": file_name,
                  "bucket_name": bucket_name
            }
         endpoint = 'http://funk.soracom.io'
         req = urllib.request.Request(
            endpoint,
            data = msg.encode(),
            headers = {"content-type":"application/json"}
         )
         with urllib.request.urlopen(req) as res:
            response_data = json.loads(res.read().decode('utf8'))
         return response_dat
      ```
   2. s3に画像を送信

      1により発行されたPresigned URLと, デバイスから取得したRGB画像をもとに、s3に画像を保存するプログラムは、以下の通りです。
      
      ```python
      def upload_s3(url_data, image):
      """
      画像をs3にアップロードする
      url_data: presiendurlの取得結果
      image: rgb img
      """
      # convert to jpg
      if image is None:
         raise ValueError("Upload s3 can not convert img")

      # 画像をエンコード
      encode_param = [cv2.IMWRITE_JPEG_QUALITY, UPLOAD_QUALITY]
      _, im_data = cv2.imencode('.jpg', image, encode_param)
      im_data_byte = im_data.tobytes()
      files = {'file': im_data_byte}

      # 送信する情報をまとめる
      if "url" in url_data and "fields" in url_data:
         url = url_data["url"]
         fields = url_data["fields"]
      else:
         raise ValueError("url and fields not included in Presigned url")
      
      # s3に送信
      try:
         res= requests.post(url, data=fields, files=files)
         logger.debug(res.text)
      except Exception as e:
         raise ValueError(f"Upload s3 post fail {e}")
      return True
      ```


   基本的には、上記に示した記事とプログラムで、画像をs3に送信できると思います。


# まとめ

以上、SORACOMのAI カメラ「[SORACOM Plus Camera Basic](https://soracom.jp/soracom_plus/camera_basic/)」で取得した画像をs3に保存する方法を紹介しました。最近は、モデルの軽量化手法も多く開発されてきており、エッジで処理することが当たり前になる時代がきているかもしれません。その中で、SORACOMのこのカメラは、面倒な通信部分などは、SORACOM側で処理してくれており、AIや画像処理の実装に集中できるので、機械学習エンジニアとしても重宝しています！


