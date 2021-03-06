---
title: "わっしー 機械学習記事まとめ" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "idea" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: True # 公開設定（falseにすると下書き）
---

こんにちは、わっしーです。
機械学習関連の記事を分散して書きすぎたので、まとめます。

- [twitter: @kwashizzz](https://twitter.com/kwashizzz)  
- [wantedly](https://www.wantedly.com/id/kai_washizaki)
- [zenn](https://zenn.dev/kwashizzz)

# 2022

## [SimCTG: テキスト生成におけるContrastive学習推論の解説・実装](https://zenn.dev/kwashizzz/articles/ml-simctg-contrastive-framework)

文章生成のモデルの生成テキストの不自然さや、望ましくない単語の繰り返しを抑える手法として、A Contrastive Framework for Neural Text Generationで提案されたSimCTG(a SIMple Contrastive framework for neural Text Generation)を解説した記事です。また、論文の実装コードを参考にし、Encoder-Decoder形式の文章生成モデルであるT5にSimCTGを適用してみたので、実装方法も解説しています。

## [VOS: Learning What You Don't Know by Virtual Outlier Synthesis の解説・実装](https://zenn.dev/kwashizzz/articles/ml-vos-ood-det)

OOD(Out-of-distribution)検出でSoTAのVOS(Virtual Outlier Synthesis)を提案した論文の解説記事です。今までの手法は、実際のクラス外のデータが必要で、データ作成にコストがかかったり、そもそも取得不可能場合もありました。しかし、VOSでは、仮想的なクラス外データを効率よく作成し、学習するためクラス外のデータは必要ないため、かなり実用的です。実際に実験を行い、OOD検出の性能の良さに驚きました！

# 2021年

## [Solafuneの市街地画像の超解像化コンペのまぁまぁ高精度な解法](https://zenn.dev/kwashizzz/articles/solafune-sr-swinir)

Solafuneの市街地画像の超解像化コンペにて、SwinIRを用いた手法・実装についてまとめた記事です。
Solafune Award受賞しました！

## [顔編集で表情や年齢を変えてみた！Pivotal Tuning for Latent-based Editing of Real Imagesの解説](https://tech.fusic.co.jp/posts/2021-08-02-ml-pivotal-tuning/)  

顔の向きや表情、年齢などの属性を変更する最新の顔編集技術である、Pivotal Tuning for Latent-based Editing of Real Imagesについて解説しています。Pivotal Tuningは、学習済みの生成器を入力データに対してわずかにチューニングする技術で、メイクが濃い顔やメガネをかけた人など、学習データにあまり含まれていないような画像に対しても納得感のある顔編集を可能にします。

---

## [人の顔を入れ替えてみた！最新の顔すり替え手法 SimSwapの解説！](https://tech.fusic.co.jp/posts/2021-07-13-ml-sim-swap/)

従来の顔すり替え技術では、対象の顔を目的の顔にすり替えるためにチューニングする必要があったり、すり替え対象の顔における表情や視線などの属性をすり替え後に保持できていないなどの課題がありました。そこで、[SimSwap: An Efficient Framework For High Fidelity Face Swapping](https://arxiv.org/abs/2106.06340)という論文では、Simple Swap (SimSwap)と呼ばれる効率的で、任意の顔の属性を保持したまま、目的の顔にすり替えるフレームワークが提案していました。

---

## [GANs N' Roses: Stable, Controllable, Diverse Image to Image Translation の解説！](https://tech.fusic.co.jp/posts/2021-06-20-ml-gans-n-roses/)

GANで多様なスタイル変換を行う[論文](https://arxiv.org/abs/2106.06561)です。1枚の画像をデータ拡張した画像を一つのバッチとすることで、コンテンツ情報は変化し、スタイル情報は一定という学習ができ、多様なスタイル変換を可能にしている。

![](https://drive.google.com/uc?export=view&id=14Lzn15yiUQyo5UCJ0wGur3JzCfkDrkVb =400x)

---

## [【論文解説】Self-Attention Between Datapoints - ノンパラメトリック深層モデル Non-Parametric Transformers の解説](https://tech.fusic.co.jp/posts/2021-06-13-ml-npts-between-datapoints/)

21年6月６日、「○○ is Not All You Need」系の論文の系譜である「Tabular Data: Deep Learning is Not All You Need](https://arxiv.org/abs/2106.03253)という研究が発表された。この研究では、表形式のデータにおいて、XGBoostの精度が深層ニューラルネットワークを上回ることが分かったと主張されていた。

しかし、この2日前(21年6月4日)に、表形式のデータで、競争力ある新たな深層学習アーキテクチャである、[Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning](https://arxiv.org/abs/2106.02584)という研究が発表されていた。このカオス具合が面白くて記事を書いた。論文にて提案されているNon-Parametric Transformers (NPTs)は、Boosting系の手法と比較しても同等以上の結果を示していた。

---


## [【論文読み】SegFormer: Simple and Efficient Design for SemanticSegmentation with Transformers の解説](https://tech.fusic.co.jp/posts/2021-06-05-ml-segformer/)

セマンティックセグメンテーションで軽量で性能も良い[SegFormer](https://arxiv.org/abs/2105.15203v1)というAttentionベースの最新手法に関する記事。論文には、「Transformerと軽量な多層パーセプトロン(MLP)デコーダを統合した、簡素で効率的でかつ強力なセマンティックセグメンテーションフレームワーク」と書かれていた。画像におけるAttentionに関して勉強になる論文だった。

---


## [動画にない視点の画像を作成してみた! NeRFを時間方向に拡張したNSFF : Nural Scene Flow Fieldの解説](https://tech.fusic.co.jp/posts/2021-06-03-nsff-nural-scene-flow-field/)

最近できた新しい分野であるNeural Radiance Fields(NeRF)において、動的な物体にも対応した[Neural Scene Flow Fields (NSFF)](https://www.cs.cornell.edu/~zl548/NSFF/)に関する記事。下図が試した結果。実際には、カメラを動かしながら撮った動画だが、NSFFを学習することで、ある時刻のある視点からの画像を表現できている。この出力のために学習に4日ほどかかった。

![](https://lh3.googleusercontent.com/ofZSgKJw5rCOrDcyfUdY6KoUuOCmwduFLiN8hVrkKQmbkQdaiZ0X59xyBStjYGsE3tk6V9uM-Ryvj4j3TOPp0aSWGFmd5Mo9poZvu3W-GwIwRrN92QD7ehE3gHFcPMDUuqViVmmrbA=w600 =400x)


---


## [オンライン複数物体追跡 SiamMOT: Siamese Multi Object Trackingの解説](https://tech.fusic.co.jp/posts/2021-06-01-ml-siam-mot/)

[（株）Fusic](https://fusic.co.jp/)では、スポーツxAIに力を入れていて、その一環で調査した論文 [SiamMOT](https://www.amazon.science/publications/siammot-siamese-multi-object-tracking)に関する記事。2021年5月末に発表されたstate-of-the-artなリアルタイムMOTの手法となる。実際にサッカーの試合動画に試し、精度が良さそうだった。


![](https://lh3.googleusercontent.com/BIr_cx6zoBeTg76kZBPzvMYPbBDQeY3FEl7uVHP5QyqZIH43bx6_QWXLMpxKqblKUfkAOZDJoWFve_3oTlqa2zcaqb9u47-bgsb2V3bUwyZjyDk3RghPcZ6m5JP3w9jrtL4IAfW1Xg=s0 =400x)

---


## [Involution: Inverting the Inherence of Convolution for Visual RecognitionをEfficientNetで試してみた](https://tech.fusic.co.jp/posts/2021-03-25-ml-involutoin-efficient-net/)

Involutionという、新しい演算手法を提案した[論文](https://arxiv.org/abs/2103.06255)に関する記事。空間に依存し、チャンネルに依存しないという利点があるとのこと。論文は、Resnetだったが、EfficientNetにInvolutionを導入して実験した結果も記事にした。畳み込み演算に取って代わるかというと今後に期待！

---


## [Attenton is All You Need in Speech Separation. 音源分離にもAttentionの時代が到来！](https://tech.fusic.co.jp/posts/2021-03-23-sss-attention/)

音源分離タスクにもAttentionが使われたという論文[Attenton is All You Need in Speech Separation](https://arxiv.org/abs/2010.13154)を読んだ記事。学生時代、音源分離の研究をしていたが、1chでここまで性能がでるとは思っていなかった。提案手法であるSepFormerがHugging Faceにて公開されているので、簡単に試せて、実際性能が良かった。当たり前だが、似た声の人の音源分離は難しかった。

---


## [【論文読み】 Nomalizer-Free ResNets (NFNet) with AGC - EfficientNetの画像認識精度を超えた最新のモデル](https://tech.fusic.co.jp/posts/2021-02-25-nfnets/)

[CHARACTERIZING SIGNAL PROPAGATION TO CLOSETHE PERFORMANCE GAP IN UNNORMALIZEDRESNETS](https://arxiv.org/abs/2101.08692)で発表された、バッチ正規化なしの最先端のモデルに、学習を安定されるための適応勾配クリッピング(adaptive gradient clipping; AGC)を加えることで、ImageNetデータセットでSOTAを達成した[論文](https://arxiv.org/abs/2102.06171)の解説。バッチ正規化なしという珍しさで読んだ。試したが、論文通りの性能がでなかった。


---


## [【論文読み】Exploring Simple Siamese Representation Learning](https://tech.fusic.co.jp/posts/2020-12-25-ml-simsiam-representation-learning/)

表現学習を行う手法の一つであるSiamese学習の新しいアーキテクチャを提案された[論文](https://arxiv.org/abs/2011.10566)の解説記事。ネガティブサンプリングが必要なく、stop gradient(勾配停止)と、予測レイヤを追加するのみで実装が簡単だった。性能が良くなる理由が、かなり納得がいくもので、表現学習Loveの私には、最高の論文だった。

---



# 2020年

## [【論文読み】A Survey on Deep Learning for Localization and Mapping - 自律ロボット × Deep Learning の研究動向](https://tech.fusic.co.jp/posts/2020-12-11-ml-robot-ai-loc-map-survey/)

[A Survey on Deep Learning for Localization and Mapping: Towards the Age of Spatial Machine Intelligence ](https://arxiv.org/abs/2006.12567)という自律ロボットのSLAMに関するサーベイをまとめた記事。SLAM欲に飢えていたため書いた。

---


## [【論文読み】SuperGlue - ロボティクスに欠かせない、GNNによる特徴マッチング手法](https://tech.fusic.co.jp/posts/2020-09-08-ai-super-glue-gnn/)

案件で、画像同士の特徴マッチングに関して調査していて面白かった[SuperGlue](https://arxiv.org/abs/1911.11763)のまとめ記事。特徴抽出は、SuperPointなどで行い、その後のマッチングにAttentionを導入して精度向上している。下の図が実際に試した結果。

![](https://storage.googleapis.com/zenn-user-upload/ddd45b0eecd6a10bb90f6ee0.png =600x)

---


## [【論文読み】Learning to Cartoonize Using White-box Cartoon Representations - 写真をイラスト化するAI](https://tech.fusic.co.jp/posts/2020-08-17-ai-cv-gan-cartoonize-with-white-box-rep/)

CVPR2020で個人的に面白かった 写真のイラスト化に関する論文 [Learning to Cartoonize Using White-box Cartoon Representations](https://systemerrorwang.github.io/White-box-Cartoonization/paper/06791.pdf)に関する記事。損失関数の設計が、イラスト化とは?ということを明確に定義して設計されていた。下の図が試した結果。

![](https://storage.googleapis.com/zenn-user-upload/493e0bd97b9efb20ac1c50f4.jpg =400x)