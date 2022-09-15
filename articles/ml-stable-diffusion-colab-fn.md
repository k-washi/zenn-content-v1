---
title: "【Gooble Colab】Stable DiffusionをFine Tuningして、あくたんを生成してみた!" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["python", "ml"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
---

# はじめに

こんにちは、[わっしー](https://twitter.com/kwashizzz)です。最近、画像生成のアプリやデモが流行って、めっちゃ試したくなりますよね。私も、実際に動かし、あくたんこと[湊あくあ](https://hololive.hololivepro.com/talents/minato-aqua/)さんの画像を用いてFineTuningしてみました。あくたんの、チャンネル登録よろしくお願いします。

以下、生成した画像ですが、どうでしょうか！めっちゃすごくて感動です。特にお気に入りは、真ん中左の、ちょっと生意気なあくたんの画像です。

![](https://storage.googleapis.com/zenn-user-upload/5152a7ff702a-20220915.jpg)

画質がいいやつも置いときますね。

![](https://storage.googleapis.com/zenn-user-upload/36f827b3b34e-20220915.png)

実際には、ランダムシードを変更しつつ、何百枚も生成して、よかったものを選択しています。

本記事では、自前で収集した画像群を用いてFineTuningする方法を紹介します。

※Google ColabのPro+(GPU:A100)を使用し、動作確認をしています。

# 環境構築

まずは、環境構築です。
最初に、[【Gooble Colab】Stable Diffusionを動かしてみた](https://zenn.dev/kwashizzz/articles/ml-stable-diffusion-colab)の記事に則って環境構築をしてください。

次に、Fine Tuning用のライブラリの設定です。

`textual_inversion`というライブラリをダウンロードし、必要なライブラリをインストールします。

```
%cd /content/
!git clone https://github.com/rinongal/textual_inversion.git

%cd /content/textual_inversion
!pip install -e .
```

他にも必要そうなライブラリをインストールします。動かなかった時の対応も含んでいます。適宜、いい感じに修正してください。

```
!pip install pyyaml tqdm easydict scikit-image scikit-learn tensorflow matplotlib joblib albumentations pandas hydra-core pytorch-lightning tabulate kornia tabulate packaging scikit-learn wldhx.yadisk-direct --quiet

!pip uninstall opencv-python-headless -y --quiet
!pip install opencv-python-headless==4.1.2.30 --quiet

!pip install --upgrade --force-reinstall git+https://github.com/arogozhnikov/einops.git
```

# Fine Tuning用のデータセットの準備

次に、データセットの準備をします。
準備した画像は、10枚程度で、512x512pxに、事前に変更しておきました。他の画質では、試していないので、動くかわかりません。

解凍先のディレクトリもいい感じにしてください。

```
!cp /content/drive/MyDrive/data/sd_dataset.zip /content
!unzip -o  /content/sd_dataset.zip 
```

# 学習

いろいろ準備し、やっと学習です。以下のコマンドを実行するだけです。

`actual_resume `は、[【Gooble Colab】Stable Diffusionを動かしてみた](https://zenn.dev/kwashizzz/articles/ml-stable-diffusion-colab) でダウンロードした、モデルの重みです。`data_root`は準備したデータセットディレクトリを指定してください。

```
!python main.py --base configs/stable-diffusion/v1-finetune.yaml \
               -t \
               --actual_resume /content/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt \
               -n v1 \
               --gpus 0, \
               --data_root /content/textual_inversion/sd_dataset
```

もし、以下のようにエラーがでた場合、フォントを変更してください。


```
File "/content/textual_inversion/ldm/util.py", line 25, in log_txt_as_img
    font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
```

Google Colabでは、以下のフォントが使えると思います。

```
!ls /usr/share/fonts/truetype/humor-sans/Humor-Sans.ttf
```

そのため、`/content/textual_inversion/ldm/util.py`の`font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)`を上記のファイルに変更します。

あとは、`configs/stable-diffusion/v1-finetune.yaml`の修正です。以下のkeyを、いい感じに変更してください。
```
initializer_words: ["kawaii", "anime", "pixiv"]
```

# 推論

最後に学習して取得した重みを用いて、推論です。以下のようなコマンドで実行できます。`--embedding_path`は、学習して取得した重みを選択してください。`prompt`は、`*`を入れて、`photo of *`などにすれば良いです。以下の例の文でも良いです。

```
!mkdir -p ./imgoutputs
!python scripts/stable_txt2img.py --ddim_eta 0.0 \
                          --n_samples 8 \
                          --n_iter 2 \
                          --scale 10.0 \
                          --ddim_steps 50 \
                          --embedding_path /content/textual_inversion/logs/aqua2022-09-01T15-35-52_v1/checkpoints/embeddings_gs-2999.pt \
                          --ckpt /content/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt \
                          --config /content/textual_inversion/configs/stable-diffusion/v1-inference.yaml \
                          --prompt "a anime official illustration of *, minato akua, pink haircolor, trending on pixiv fanbox, painted by makoto shinkai takashi takeuchi Twinkle," \
                          --outdir ./imgoutputs
```


# 最後に



簡単にですが、環境構築から、Fine Tuningまでの解説を行いました。少しあくたん??って思う画像が生成されていますが、あくたんの生成ができ感激です。構図などを、自分の思い通りに指定しずらかったり、顔が崩壊した画像が生成されたりとまだまだな気がしますが、来年には、改善したより凄く、使いやすいモデルが出てきている気がします。発展スピードの速さに驚きを隠せません。
