---
title: "わっしー 機械学習記事まとめ" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "idea" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: True # 公開設定（falseにすると下書き）
---

こんにちは、わっしーこと鷲崎です。
機械学習関連の記事を分散して書きすぎたので、まとめます。

- [twitter: @kwashizzz](https://twitter.com/kwashizzz)  
- [wantedly](https://www.wantedly.com/id/kai_washizaki)
- [zenn](https://zenn.dev/kwashizzz)

<br>

# 2021年

## [【論文解説】Self-Attention Between Datapoints - ノンパラメトリック深層モデル Non-Parametric Transformers の解説](https://tech.fusic.co.jp/posts/2021-06-13-ml-npts-between-datapoints/)

21年6月６日、「○○ is Not All You Need」系の論文の系譜である「Tabular Data: Deep Learning is Not All You Need](https://arxiv.org/abs/2106.03253)という研究が発表された。この研究では、表形式のデータにおいて、XGBoostの精度が深層ニューラルネットワークを上回ることが分かったと主張されていた。

しかし、この2日前(21年6月4日)に、表形式のデータで、競争力ある新たな深層学習アーキテクチャである、[Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning](https://arxiv.org/abs/2106.02584)という研究が発表されていた。このカオス具合が面白くて記事を書いた。論文にて提案されているNon-Parametric Transformers (NPTs)は、Boosting系の手法と比較しても同等以上の結果を示していた。

---
<br>

## [【論文読み】SegFormer: Simple and Efficient Design for SemanticSegmentation with Transformers の解説](https://tech.fusic.co.jp/posts/2021-06-05-ml-segformer/)

セマンティックセグメンテーションで軽量で性能も良い[SegFormer](https://arxiv.org/abs/2105.15203v1)というAttentionベースの最新手法に関する記事。論文には、「Transformerと軽量な多層パーセプトロン(MLP)デコーダを統合した、簡素で効率的でかつ強力なセマンティックセグメンテーションフレームワーク」と書かれていた。画像におけるAttentionに関して勉強になる論文だった。

---
<br>

## [動画にない視点の画像を作成してみた! NeRFを時間方向に拡張したNSFF : Nural Scene Flow Fieldの解説](https://tech.fusic.co.jp/posts/2021-06-03-nsff-nural-scene-flow-field/)

最近できた新しい分野であるNeural Radiance Fields(NeRF)において、動的な物体にも対応した[Neural Scene Flow Fields (NSFF)](https://www.cs.cornell.edu/~zl548/NSFF/)に関する記事。下図が試した結果。実際には、カメラを動かしながら撮った動画だが、NSFFを学習することで、ある時刻のある視点からの画像を表現できている。この出力のために学習に4日ほどかかった。

<img src="https://lh3.googleusercontent.com/ofZSgKJw5rCOrDcyfUdY6KoUuOCmwduFLiN8hVrkKQmbkQdaiZ0X59xyBStjYGsE3tk6V9uM-Ryvj4j3TOPp0aSWGFmd5Mo9poZvu3W-GwIwRrN92QD7ehE3gHFcPMDUuqViVmmrbA=w600" width="400">

---
<br>

## [オンライン複数物体追跡 SiamMOT: Siamese Multi Object Trackingの解説](https://tech.fusic.co.jp/posts/2021-06-01-ml-siam-mot/)

[（株）Fusic](https://fusic.co.jp/)では、スポーツxAIに力を入れていて、その一環で調査した論文 [SiamMOT](https://www.amazon.science/publications/siammot-siamese-multi-object-tracking)に関する記事。2021年5月末に発表されたstate-of-the-artなリアルタイムMOTの手法となる。実際にサッカーの試合動画に試し、精度が良さそうだった。


<img src="https://lh3.googleusercontent.com/BIr_cx6zoBeTg76kZBPzvMYPbBDQeY3FEl7uVHP5QyqZIH43bx6_QWXLMpxKqblKUfkAOZDJoWFve_3oTlqa2zcaqb9u47-bgsb2V3bUwyZjyDk3RghPcZ6m5JP3w9jrtL4IAfW1Xg=s0" width="400">


---
<br>

## [Involution: Inverting the Inherence of Convolution for Visual RecognitionをEfficientNetで試してみた](https://tech.fusic.co.jp/posts/2021-03-25-ml-involutoin-efficient-net/)

Involutionという、新しい演算手法を提案した[論文](https://arxiv.org/abs/2103.06255)に関する記事。空間に依存し、チャンネルに依存しないという利点があるとのこと。論文は、Resnetだったが、EfficientNetにInvolutionを導入して実験した結果も記事にした。畳み込み演算に取って代わるかというと今後に期待！

---
<br>

## [Attenton is All You Need in Speech Separation. 音源分離にもAttentionの時代が到来！](https://tech.fusic.co.jp/posts/2021-03-23-sss-attention/)

音源分離タスクにもAttentionが使われたという論文[Attenton is All You Need in Speech Separation](https://arxiv.org/abs/2010.13154)を読んだ記事。学生時代、音源分離の研究をしていたが、1chでここまで性能がでるとは思っていなかった。提案手法であるSepFormerがHugging Faceにて公開されているので、簡単に試せて、実際性能が良かった。当たり前だが、似た声の人の音源分離は難しかった。

---
<br>

## [【論文読み】 Nomalizer-Free ResNets (NFNet) with AGC - EfficientNetの画像認識精度を超えた最新のモデル](https://tech.fusic.co.jp/posts/2021-02-25-nfnets/)

[CHARACTERIZING SIGNAL PROPAGATION TO CLOSETHE PERFORMANCE GAP IN UNNORMALIZEDRESNETS](https://arxiv.org/abs/2101.08692)で発表された、バッチ正規化なしの最先端のモデルに、学習を安定されるための適応勾配クリッピング(adaptive gradient clipping; AGC)を加えることで、ImageNetデータセットでSOTAを達成した[論文](https://arxiv.org/abs/2102.06171)の解説。バッチ正規化なしという珍しさで読んだ。試したが、論文通りの性能がでなかった。


---
<br>

## [【論文読み】Exploring Simple Siamese Representation Learning](https://tech.fusic.co.jp/posts/2020-12-25-ml-simsiam-representation-learning/)

表現学習を行う手法の一つであるSiamese学習の新しいアーキテクチャを提案された[論文](https://arxiv.org/abs/2011.10566)の解説記事。ネガティブサンプリングが必要なく、stop gradient(勾配停止)と、予測レイヤを追加するのみで実装が簡単だった。性能が良くなる理由が、かなり納得がいくもので、表現学習Loveの私には、最高の論文だった。

---
<br>


# 2020年

## [【論文読み】A Survey on Deep Learning for Localization and Mapping - 自律ロボット × Deep Learning の研究動向](https://tech.fusic.co.jp/posts/2020-12-11-ml-robot-ai-loc-map-survey/)

[A Survey on Deep Learning for Localization and Mapping: Towards the Age of Spatial Machine Intelligence ](https://arxiv.org/abs/2006.12567)という自律ロボットのSLAMに関するサーベイをまとめた記事。SLAM欲に飢えていたため書いた。

---
<br>

## [【論文読み】SuperGlue - ロボティクスに欠かせない、GNNによる特徴マッチング手法](https://tech.fusic.co.jp/posts/2020-09-08-ai-super-glue-gnn/)

案件で、画像同士の特徴マッチングに関して調査していて面白かった[SuperGlue](https://arxiv.org/abs/1911.11763)のまとめ記事。特徴抽出は、SuperPointなどで行い、その後のマッチングにAttentionを導入して精度向上している。下の図が実際に試した結果。

<img src="https://storage.googleapis.com/zenn-user-upload/ddd45b0eecd6a10bb90f6ee0.png" width="600">

---
<br>

## [【論文読み】Learning to Cartoonize Using White-box Cartoon Representations - 写真をイラスト化するAI](https://tech.fusic.co.jp/posts/2020-08-17-ai-cv-gan-cartoonize-with-white-box-rep/)

CVPR2020で個人的に面白かった 写真のイラスト化に関する論文 [Learning to Cartoonize Using White-box Cartoon Representations](https://systemerrorwang.github.io/White-box-Cartoonization/paper/06791.pdf)に関する記事。損失関数の設計が、イラスト化とは?ということを明確に定義して設計されていた。下の図が試した結果。

<img src="https://storage.googleapis.com/zenn-user-upload/493e0bd97b9efb20ac1c50f4.jpg" width="400">