---
title: "Sparse DETRを読んで、DETRを俯瞰する" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

こんにちは！鷲崎です。本記事では、[SPARSE DETR: EFFICIENT END-TO-END OBJECT DE-
TECTION WITH LEARNABLE SPARSITY](https://arxiv.org/abs/2111.14330)に書かれている、DETRに関連した手法について、解説します。




# DETRとは？

DETR以前の物体検出では、
1. 物体領域の候補がたくさん予測されるため、重複したBounding Box(BBox)を除去・集約する必要があった。そこで、[Nox-Maximum Suppression(NMS)](https://ohke.hateblo.jp/entry/2020/06/20/230000)などが用いられていました。
2. アンカーボックスなどを用いて、予め物体領域の候補を設定している。
のように、予め人がパラメータを設定する、人の介在部分があるため、完全なEnd-to-Endな物体検出とは言えませんでした。
例えば、小さい物体を検出したいなら、それに適したパラメータがありますし、これを人手で調整するのは大変です。

そこで、上記の欠点を排除し、End-to-Endな学習を可能にした、[DEtection TRansformer(DETR)](https://arxiv.org/abs/2005.12872)という手法が提案されました。

## DETR手法の概要

DETRの論文から引用した、以下の図がわかりやすいです。

途中までは、Transformerで言語を扱うときとほぼ同じです。**BackBone**部分では、CNNにより特徴ベクトルを抽出し、そこに位置情報として、Positional encodingを加えています。これを**Transformer**の入力として、物体の位置やクラスの情報へとエンコードしています。
**FFN**は、Transformerの出力から、位置とクラスを予測しています。Transformer部分のAttentionにより、検出物体同士の関係性も捉えられるため、NMSなど使用せずとも、重複なしの予測ができている気がします。

![DETRアーキテクチャ](https://storage.googleapis.com/zenn-user-upload/df47db2e7646-20220105.png)

とても簡単なアーキテクチャですが、内部では、様々な工夫がされています。
例えば、Hungarian Lossの導入です。

## Hungarian Loss

DETRは、1度に複数の物体の推論結果を出力できます。そのため、効率的に学習するためには、推論結果と正解ラベルの対応づけを自動で行う必要があります。そこで、用いられたのが、ハンガリアン アルゴリズムです。詳細は、[割当問題のハンガリアン法をpythonで実装してみた](https://qiita.com/m__k/items/8e2cb9067ec5d720c30d)がわかりやすかったです。

他にも、Generalize IoUなど、いろいろ工夫されています。詳しくは、[論文](https://arxiv.org/abs/2005.12872)を読んでみてください。

## 欠点

DETRでは、Transformerを用いておりメモリ使用量や計算量が増えたことで、、物体検出でよく用いられるFeature Pyramid Network (FPN)を導入できず、小さい物体の検出性能は、よくありませんでした。

# Deformable DETR









