---
title: "【Gooble Colab】Stable Diffusionを動かしてみた" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["python", "ml"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
---

# はじめに

こんにちは、[わっしー](https://twitter.com/kwashizzz)です。最近、画像生成のアプリやデモが流行っており、生成モデルに入力するプロンプトテキストが呪文と言われるようにまでなっています。私もStable Diffusionを試し、以下のようなピンク色のアニメキャラクターを生成できました。

![](https://storage.googleapis.com/zenn-user-upload/cadfcc09e1b9-20220902.jpg)

本記事では、Google Colab上で[Stable Diffusion](https://github.com/CompVis/stable-diffusion)を動かす方法を紹介します。

※Google ColabのPro+(GPU:A100)を使用し、動作確認をしています。

# 環境設定

まずは、実行に必要なライブラリやStable Diffusionの重みをダウンロードします。

```sh
# stable diffusionのダウンロード
!git clone https://github.com/CompVis/stable-diffusion.git
```

次に、ライブラリのインストールをします。

```sh
%%capture

# ライブラリのダウンロード
%cd /content/stable-diffusion
!pip install albumentations==0.4.3\
    opencv-python==4.1.2.30\
    pudb==2019.2\
    imageio==2.9.0\
    imageio-ffmpeg==0.4.2\
    pytorch-lightning==1.4.2\
    omegaconf==2.1.1\
    test-tube>=0.7.5\
    streamlit>=0.73.1\
    einops==0.3.0\
    torch-fidelity==0.3.0\
    torchmetrics==0.6.0\
    kornia==0.6\
    pillow==9.0.1\
    -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers\
    -e git+https://github.com/openai/CLIP.git@main#egg=clip\
    -e .

# torchtext関連
# PyTorch 1.10(torchtext 0.11)まではlegacyに移動されいましたが利用することは可能でした。しかし、PyTorch 1.11(torchtext 0.12)で完全に削除されてしまいました。
!pip install torchtext==0.11.0
!pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

!pip install transformers==4.19.2 diffusers invisible-watermark
%cd /content/stable-diffusion
!pip install -e .
```

次に、`huggingface`から、重みのダウンロードします。
重みのダウンロードの前に、まず、[huggingface](https://huggingface.co/)のサイトに行き、ログインします。その後、[stable-diffusion-v-1-4-original](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)に移動し、*Access repository*を押します。

その後、Google Colab上でも、ログインします。

```
!huggingface-cli login
```

次に、重みのダウンロードです。

```
!git config --global credential.helper store
!git lfs install
!git clone https://huggingface.co/CompVis/stable-diffusion-v-1-4-original
```

そして、ダウンロードしたファイルを移動させます。

```
!rm -r /content/stable-diffusion/stable-diffusion-v1
!cp -r /content/stable-diffusion/stable-diffusion-v-1-4-original /content/stable-diffusion/stable-diffusion-v1
!mv /content/stable-diffusion/stable-diffusion-v1/sd-v1-4.ckpt /content/stable-diffusion/stable-diffusion-v1/model.ckpt
!cp -r /content/stable-diffusion/stable-diffusion-v1 /content/stable-diffusion/models/ldm/
```

以上で、環境の設定が終わりです。

# 実行

実行は簡単です。以下このコマンドを実行するだけです。*prompt*の部分が重要で、いい感じにイラストを生成する際には、かなり長い文が必要な時もあります。個人的に、コツは有名なイラストレータとかの名前を入れるとかですね。

```
!mkdir -p ./imgoutputs
%cd /content/stable-diffusion
!python scripts/txt2img.py --prompt "Beautiful girl with cat ears wearing headphones,illust" --plms --outdir ./imgoutputs
```


# 最後に

簡単にですが、環境構築から、実行までの解説を行いました。実際生成した画像の多くは、アニメキャラクターとは言えない、顔面が崩壊したものだったので、少しがっかり感があります。もちろん、この精度で生成できることは、すごいことですが。
他にも、画像生成のライブラリが色々と開発され、この崩壊を解決しているものもあるので、技術の進歩は恐ろしいです。

---

画像・自然言語・音声に関する機械学習の研究開発やMLOpsを行っています。もし、機械学習に関して、ご相談があれば、[@kwashizzz](https://twitter.com/kwashizzz)のアカウントまでDMしてください！
これまでの、[機械学習記事のまとめ](https://zenn.dev/kwashizzz/articles/my-ml-articles-summary)です。