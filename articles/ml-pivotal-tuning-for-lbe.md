---
title: "顔の向き・表情・年齢を変化させてみた！Pivotal Tuning for Latent-based Editing of Real Imagesの解説" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: False # 公開設定（falseにすると下書き）
---

こんにちは、機械学習チームの鷲崎です。最近、弊社では、GANに関する技術調査を行っていまして、[女性の声を男性の声に変換してみた！CycleGAN VCを用いた音声変換の説明](https://tech.fusic.co.jp/posts/2021-06-29-ml-cycleganvc/)　や　[GANs N' Roses: Stable, Controllable, Diverse Image to Image Translation の解説！](https://tech.fusic.co.jp/posts/2021-06-20-ml-gans-n-roses/)などの解説記事をだしています。

本記事では、顔の向きや表情、年齢などの属性を変更する顔編集技術において最新の論文である、[Pivotal Tuning for Latent-based Editing of Real Images](https://arxiv.org/abs/2106.05744)に関して解説します。Pivotal Tuningによる顔編集技術を実際に試し、以下の動画のような結果が得られました。驚くべき編集能力を持っている持っていることが分かります。



![](https://storage.googleapis.com/zenn-user-upload/e14d2af3b911565256cc3e01.gif)

![](https://storage.googleapis.com/zenn-user-upload/80217ceab2c2341990a517a0.gif)

![](https://storage.googleapis.com/zenn-user-upload/f048e8f06877f376912ddcfd.gif)