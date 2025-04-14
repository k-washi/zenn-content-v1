---
title: "音声合成モデル VITSの性能と速度改善をしてみた" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
publication_name: "fusic"
---

こんにちは、最近、3D, 音声関連の機械学習にはまっている、[わっしー](https://twitter.com/kwashizzz)です。

音声合成の中でもVITSは、よく使用されるモデルの一つです。最近だと、BERT-VITS2が有名かもしれません。そのVITSモデルの性能と速度改善を行いました!

最終的に、性能をキープした状態で、GPU上で3倍、CPU上で5倍の速度改善を行うことができました。

# 改善のまとめ

## 性能の改善

特に以下の工夫を行うことで、VITSの性能を改善しました。

1. 正解音と生成音によるWavLM特徴量の差による損失、adversarial lossを追加
2. monotonic alignmentを破棄し、[EfficnetTTS2](https://openreview.net/forum?id=__czv_gqDQt)に似たアーキテクチャ(Alignment Predictorなど)を導入

## 速度の改善

VITSモデルのアーキテクチャの変更と、`torch.compile`を使用することで、速度の改善を行いました。

1. ボコーダーを[MS-iSTFT](https://github.com/MasayaKawamura/MB-iSTFT-VITS)に変更
2. 処理の一部を[ConvNext](https://arxiv.org/abs/2201.03545) likeな処理に変更
3. Alignment Predictorを導入
4. `torch.compile`による最適化

# 実験条件

本記事では、JSUT BASIC5000_0001~4500で、訓練を行い、JSUT BASIC5000_4501~5000で、評価で評価を行いました。
速度測定は、JSUT BASIC5000_4501~4600で行い、後半の50サンプルでReal Time Factor (RTF)を計算しました。
サンプルレートは、22050Hzです。

実験環境は以下の通りです。
GPU: NVIDIA GeForce RTX 3090
CPU: Intel(R) Core(TM) i9-10900 CPU @ 2.80GHz

評価は、AutoMOS, SpeechBertScoreで行い、速度はRTF(1秒生成するのにかかる時間)で評価しました。

AutoMOSは、[SpeechMos](https://github.com/tarepan/SpeechMOS)ライブラリのtarepan/SpeechMOS:v1.2.0を使用し、自動でMOSを計算しました。(5点満点で、高いほど良い音声)
[SpeechBertScore](https://arxiv.org/abs/2401.16812)は、正解音と生成音の[rinna/japanese-hubert-base](https://huggingface.co/rinna/japanese-hubert-base)による特徴量の類似度を計算しました。(0~1で、高いほど良い音声)

# 結果

以下、実験結果です。

| モデル | パラメータ数 | AutoMos ↑ | SpeechBertScore ↑ | RTF(GPU) | RTF(CPU) |  |
| --- | --- | --- | --- | --- | --- | --- |
| VITS | 30.19M | 3.93033 | 0.91275 | 0.02380384 | 0.20747716 |  |
| VITS2 | 42.84M | 3.91726 | 0.91268 | 0.02558740 | 0.21570882 |  |
| FlyTTS | 31.01M | 3.72532 | 0.91167 | 0.01727137 |x |  CPUで動作せず...なぜ... |
| MS-iSTFT-VITS | 28.17M | 3.881 | 0.91229 | 0.01032783 | 0.05252593 | ms-istft-vits + wavlmloss adv Loss + ConvNext like |
| MS-iSTFT-ET2 | 26.65M | **3.95708** | **0.91951** | **0.00774977** | **0.04336844** | efficenttts2 + wavlmloss adv Loss + ConvNext like |

EfficnetTTS2(ET2)のボコーダをMS-iSTFT(ConvNext)に変更することで、wavlmlossとadversarial lossを追加した結果、AutoMos, SpeechBertScoreともに改善されました。
特に、EffcientTTS2ベースにすると、SpeechBertScoreが明らかに改善されました。

VITS2の工夫も導入使用しようとしましたが、durationに関するadversarial loss関連がうまく実装できていない気がしました...(性能も1話者での評価では、そんなに変わらなかったので、今回は見送りました。)  
FlyTTSは、最終的なAutoMOSがあまり良くなかったです。

次に、torch.compileを使用した場合の速度改善の結果です。

| モデル | パラメータ数 | RTF(GPU) | RTF(CPU) |  |
| --- | --- | --- | --- | --- |
| MS-iSTFT-ET2 | 26.65M | 0.00774977 | 0.04336844 |  |
|+torch.compile  | 26.65M | 0.00715229 | 0.04052205 |  |

最終的に、評価指標上で性能を改善し、GPU上で3倍、CPU上で5倍の速度改善を行うことができました!


# 最後に..

CPU上で、GPUくらい早く推論できると嬉しかったですが、なかなか難しいですね。少し残念な気持ちがありますが、性能をキープしたまま、速度を改善でき、工夫のしがいがありました！

次は、速度のプロファイリングして、どこが遅いのか、細かい部分を見て少しずつでも改善していきたいと思います。


最後に宣伝になりますが、[Fusic](https://fusic.co.jp/)では、音声、画像、言語、3Dに関するAIの研究開発を行っています。機械学習のPoCからアプリ化まで、幅広い部分をサポートできると思いますので、もし、困っている方がいましたら、お気軽にご相談ください！[お問い合わせ](https://fusic.co.jp/contact)も、[わっしー](https://twitter.com/kwashizzz)へのDMも、大歓迎です！