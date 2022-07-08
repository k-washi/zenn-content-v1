---
title: "Hugging face t5-base-japaneseをTensor-RTで高速化してみた" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習", "python"] # タグ。["markdown", "rust", "aws"]のように指定する
published: True # 公開設定（falseにすると下書き）
---

# はじめに

こんにちは、わっしーです。
画像・自然言語・音声に関する機械学習の研究開発やMLOpsを行っています。特に最近は、音声関連の研究開発に力を入れています。もし、機械学習に関して、ご相談があれば、[@kwashizzz](https://twitter.com/kwashizzz)のアカウントまでDMしてください！
これまでの、[機械学習記事のまとめ](https://zenn.dev/kwashizzz/articles/my-ml-articles-summary)です。

本記事では、Hugging faceで公開されている、日本語で事前学習されたT5 ([sonoisa/t5-base-japanese](https://huggingface.co/sonoisa/t5-base-japanese))をFineTuningしたモデルの推論を[Tensor-RT](https://github.com/NVIDIA/TensorRT)で高速化する方法について説明します。


環境構築に関しては、[公式の記事](https://developer.nvidia.com/blog/optimizing-t5-and-gpt-2-for-real-time-inference-with-tensorrt/)を参考にしています。詳細に関しては、下部に記載しています。
Tensor-RTのバージョンに関しては、最新のreleaseである、ver 8.2があります。しかし、変換後の推論が上手くできなかったため、22/6/1時点で最新のmainブランチを使用しています。

基本的には、[公式t5変換notebook](https://github.com/NVIDIA/TensorRT/blob/main/demo/HuggingFace/notebooks/t5.ipynb)に則って、Tensor-RTに変換していきます。

# 結果

どのくらい推論速度が速くなるかということですが、Encoderの処理 + decoderの処理を99回ループさせた際の処理速度は、以下の通りです。

|||
|-|-|
|T5 hugging face generate | 1.225 sec |
| Tensor RT | 0.261 sec |

Tensor RTの方が、約 5倍速くなっています。


# 詰まった部分の解説

[公式t5変換notebook](https://github.com/NVIDIA/TensorRT/blob/main/demo/HuggingFace/notebooks/t5.ipynb) の実装に対して、修正した部分を解説します。

1. T5_VARIANT = 't5-small' は、't5-base' など対象のモデルに合わせる必要がある。

2. TensorRT/demo/HuggingFace/T5/T5ModelConfig.pyのT5ModelTRConfigをモデルのconfigファイルに合わせる。
  今回は、モデルのVOCAB SIZEが32128から、32000へ変更した。

3. fp16の設定

fp16を使用しない場合、false

```python
metadata=NetworkMetadata(variant=variant, precision=Precision(fp16=False), other=T5Metadata(kv_cache=False))
```

4. Tensor-RTへの変換時の設定ファイルをFineTuningしたモデルのconfig.jsonに合わせる。

以下、[sonoisa](https://zenn.dev/sonoisa)様より、コメントいただきました。
> sonoisa/t5-base-japaneseモデル（原論文の設定のT5モデル）であれば、tfm_config.d_ff = 3072とtfm_config.feed_forward_proj = "relu"ですね。
> T5 v1.1モデル（例えば、megagonlabs/t5-base-japanese-web）であれば、tfm_config.d_ff = 2048とtfm_config.feed_forward_proj = "gated-gelu"になります。

```python
from T5.trt import T5TRTEncoder, T5TRTDecoder

tfm_config = T5Config(
    use_cache=True,
    num_layers=T5ModelTRTConfig.NUMBER_OF_LAYERS["t5-base"]
)
tfm_config.d_ff = 2048
tfm_config.vocab_size = 32000 # tokenizer.vocab_sizeは32100だが、configには、32000と書かれている。
tfm_config.num_heads = 12
tfm_config.d_model = 768
tfm_config.feed_forward_proj = "gated-gelu"
print(tfm_config)
```

5. 推論処理を別途実装

以下、推論処理です。
途中に出てくるクラスは、notebookに記載されています。

```python
T5_MODEL_DIR = "..."
t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_DIR)
tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_DIR)
config = T5Config(T5_MODEL_DIR)

text = "あいうえお"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs.input_ids.to(device)
attention_mask = inputs.attention_mask.to(device)

input_ids = input_ids.unsqueeze(0) # batch x seqlen
attention_mask = attention_mask.unsqueeze(0) # batch x -1

max_length = 512

decoder_input_ids = torch.as_tensor(
    [[tokenizer.pad_token_id]], dtype=torch.long
).to(device)

# エンコーダの計算
encoder_last_hidden_state = t5_trt_encoder(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))

s = time.time()
for step in range(max_length):
    # デコーダの計算
    outputs = t5_trt_decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_last_hidden_state,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True
            )
    logit_for_next_step = outputs.logits[:, -1, :]
    _decoder_input_ids = torch.argmax(logit_for_next_step , dim=-1)
    if _decoder_input_ids[0].item() == tokenizer.eos_token_id:
        break
    decoder_input_ids = torch.cat([decoder_input_ids, _decoder_input_ids[:, None]], dim=-1) # past keyが使えない
    
    print(tokenizer.decode(decoder_input_ids[0]))
    print(step, time.time() - s)
print(tokenizer.decode(_decoder_input_ids, skip_special_tokens=True)) # 最終的なテキスト
```



# 環境構築

Tensor-RTの環境構築は面倒なので、公式のDocker(ubuntu 20.04のversoin)を修正したものを使用しました。
後ほど記載する、requirements.txtに加えて、pytorchのインストールをしていますが、適宜修正してください。

```dockerfile
ARG CUDA_VERSION=11.4.2
ARG OS_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}
LABEL maintainer="NVIDIA CORPORATION"

ENV TRT_VERSION 8.2.5.1
SHELL ["/bin/bash", "-c"]

RUN mkdir -p /workspace

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

# Update CUDA signing key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
    git \
    pkg-config \
    sudo \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make \
    build-essential

# Install python3
RUN apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      python3-wheel &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip;

# Install TensorRT
RUN if [ "${CUDA_VERSION}" = "10.2" ] ; then \
    v="${TRT_VERSION%.*}-1+cuda${CUDA_VERSION}" &&\
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub &&\
    apt-get update &&\
    sudo apt-get install libnvinfer8=${v} libnvonnxparsers8=${v} libnvparsers8=${v} libnvinfer-plugin8=${v} \
        libnvinfer-dev=${v} libnvonnxparsers-dev=${v} libnvparsers-dev=${v} libnvinfer-plugin-dev=${v} \
        python3-libnvinfer=${v}; \
else \
    v="${TRT_VERSION%.*}-1+cuda${CUDA_VERSION%.*}" &&\
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub &&\
    apt-get update &&\
    sudo apt-get install libnvinfer8=${v} libnvonnxparsers8=${v} libnvparsers8=${v} libnvinfer-plugin8=${v} \
        libnvinfer-dev=${v} libnvonnxparsers-dev=${v} libnvparsers-dev=${v} libnvinfer-plugin-dev=${v} \
        python3-libnvinfer=${v}; \
fi

# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN pip3 install torch==1.11.0+cu113 torchaudio==0.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip3 install jupyter jupyterlab
# Workaround to remove numpy installed with tensorflow
RUN pip3 install --upgrade numpy

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc && rm ngccli_cat_linux.zip ngc.md5 && echo "no-apikey\nascii\n" | ngc config set

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu
ENV TRT_OSSPATH /workspace/TensorRT
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"
WORKDIR /workspace
```

このDockerコンテナを立ち上げるdocker-composeファイルが以下です。
Dockerfileのパスなどは適宜修正してください。

```yaml
version: '3'

services:
  t5-ja-trt:
    build:
      context: .
      dockerfile: ./docker/Dockerfile
    container_name: t5-ja-trt
    image: t5-ja-trt-image
    shm_size: '24gb'
    tty: true
    volumes: 
      - $PWD:/workspace
    command: '/bin/bash'
    ports:
      - 18121-18130:18121-18130
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            capabilities: [gpu]

```

requirements.txtは、Tensor-RTに記載されているものに加えて、t5-base-japaneseを使用するため、transformesなどもダウンロードする必要があります。また、Tensor-RTに記載されていませんが、`polygraphy`のインストールも必要となります。

```
transformers==4.11.3
sentencepiece==0.1.96
onnx==1.10.2
onnxruntime==1.8.1
-f https://download.pytorch.org/whl/cu113/torch_stable.html

pycuda<2021.1
pytest
--extra-index-url https://pypi.ngc.nvidia.com
onnx-graphsurgeon

polygraphy
```

上記の`docker-compose.yml`を用いて、`docker-compose up -d`でコンテナを立ち上げた後、コンテナ内で、TensorRTをダウンロード、コンパイルします。

```shell
git clone https://github.com/nvidia/TensorRT TensorRT
cd TensorRT
# git checkout release/8.2 # 今回はmainを使用するためcheckoutしない
git submodule update --init --recursive

# TensorRTのビルド
cd $TRT_OSSPATH
mkdir -p build && cd build
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out
make -j$(nproc)
```

# まとめ

以上、t5-base-japaneseの高速化の方法でした。実際、t5の処理速度で悩む場面は多いと思うので、ぜひ試して見てください。
