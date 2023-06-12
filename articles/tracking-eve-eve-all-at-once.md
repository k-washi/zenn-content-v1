---
title: "OmniMotion - Tracking Everything Everywhere All at Once の解説" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習", "論文解説", "SAM"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
publication_name: "fusic"
---

こんにちは！[Fusic](https://fusic.co.jp/) 機械学習チームの鷲崎です。機械学習モデルの開発から運用までなんでもしています。もし、機械学習で困っていることがあれば、気軽に[DM](https://twitter.com/kwashizzz)ください。

本記事では、[Tracking Everything Everywhere All at Once](https://arxiv.org/abs/2306.05422) という論文で提案された、オプティカルフローのように、動画における各ピクセル間の軌跡を推定する新しい手法であるOmniMotoinについて解説します。
まず、動画を見たほうがイメージがつくと思います。

https://www.youtube.com/watch?v=KHoAG3gA024


# 概要

