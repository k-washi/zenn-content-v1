---
title: "Gaussian Splatting 改善のアイデアをまとめてみた" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

こんにちは、最近、3D, 音声関連の機械学習にはまっている、[わっしー](https://twitter.com/kwashizzz)です。

昔ながらのMulti View Stereoに比べるとNeRFの凄さに数年前に驚いたのですが、計算コストの大きさに手を出すのを躊躇していました。そう思っていたら、Gaussian Splatting (GS)が登場して、計算コストを抑えつつ、NeRFに匹敵する精度を出すことができるということで、最近はGSに注目しています。

特に、GSの精度の良さから、mesh化を行いBlenderなどで使用できるようにしたいと考えています。
しかし、GSで学習されたガウシアンの点群をメッシュ化した際に、メッシュがぼやけていたり、エッジが不明確であったり、最悪の場合、穴が空いてしまうことがあります。

そこで、オリジナルのGSにどのような手法を加えることで、メッシュ化の精度を向上させることができるか考え、アイデア（論文）をまとめてみました。

主に、方針としては、以下の2つがあると思います。

1. 基盤モデルを活用した事前情報の取得

2. レンダリングされる画像、法線(Normal), 深度(Depth)の損失の改善

2. ガウシアンのcloneやsplitなどdensificationの改善

本記事では、上記の課題に関して、それぞれ解説したいと思います。

※ 本記事で使用している図は、各論文から引用いたしました。

# 基盤モデルを活用した事前情報の取得

### 深度推定モデルを利用したDepth Regularization

[3D GSの本家のリポジトリ](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#depth-regularization)を見ると、Depth Regularizationという手法が追加されていました。

記載されているDepth Regularizationは、GSの学習時に、Depthの事前情報を利用して、シーンの再構成の精度を向上させる手法です。しかし、深度推定による深度マップは、一般的に、正規化されており、そのまま使用することが困難です。そこで、[DepthRegularized GS](https://arxiv.org/abs/2311.13398)という手法では、単眼の深度推定モデルによる深度マップを、GSの前処理のcolmapで得られる疎な点群で調整し、GSの学習時に利用し、幾何的な精度の向上を実現しています。

深度推定に、[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) など、強力な深度推定の基盤モデルが利用されます。以下は、Depth Anything V2の結果の一例です。どの画像でもロバストに深度推定ができており、微細な部分の深度も補助できるため、GSの深度マップのレンダリングの性能向上が期待できます。

![depth_anything_v2](https://storage.googleapis.com/zenn-user-upload/ae1a4d92f727-20241211.png)

### 法線推定モデルの活用

GSの学習時に、法線の事前情報を利用することで、メッシュ化の精度を向上させることも期待できます。
法線推定には、[StableNormal](https://arxiv.org/abs/2406.16864)という基盤モデルがあり、下図のように、かなり精度の高い法線が推定でき、GSにおいて、エッジ部分や情報の少ない平坦な部分での性能向上が期待できます。

![](https://storage.googleapis.com/zenn-user-upload/9ecc4ec9cd1b-20241211.png)

実際、論文では、メッシュ化で性能の良い2D GSと比較しており、2D GS + StableNormalの方が、エッジが鮮明で、メッシュの精度が向上していることが確認されています。

![](https://storage.googleapis.com/zenn-user-upload/a07780e644ec-20241211.png)