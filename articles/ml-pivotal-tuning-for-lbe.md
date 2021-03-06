---
title: "Pivotal Tuning for Latent-based Editing of Real Imagesの解説" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

こんにちは、機械学習チームの鷲崎です。最近、弊社では、GANに関する技術調査を行っていまして、[女性の声を男性の声に変換してみた！CycleGAN VCを用いた音声変換の説明](https://tech.fusic.co.jp/posts/2021-06-29-ml-cycleganvc/)　や　[GANs N' Roses: Stable, Controllable, Diverse Image to Image Translation の解説！](https://tech.fusic.co.jp/posts/2021-06-20-ml-gans-n-roses/)などの解説記事をだしています。

本記事では、顔の向きや表情、年齢などの属性を変更する最新の顔編集技術である、[Pivotal Tuning for Latent-based Editing of Real Images](https://arxiv.org/abs/2106.05744)を解説します。Pivotal Tuningは、学習済みの生成器を入力データに対してわずかにチューニングすることで、メイクが濃い顔やメガネをかけた人など、学習データにあまり含まれていないような画像に対しても納得感のある顔編集が可能になります。Pivotal Tuningによる顔編集技術を実際に試し、以下の動画のような結果が得られました。各属性において驚くべき編集能力を持っていることが分かります。


|顔の向き|表情|年齢|
|-|-|-
|![](https://storage.googleapis.com/zenn-user-upload/e14d2af3b911565256cc3e01.gif)|![](https://storage.googleapis.com/zenn-user-upload/80217ceab2c2341990a517a0.gif)|![](https://storage.googleapis.com/zenn-user-upload/f048e8f06877f376912ddcfd.gif)|

最近、このような顔編集技術には、事前学習済みのStyleGANが使用されています。潜在空間と生成画像の対応を示した下図(論文中Figure.2)を見てください。実際に顔を編集する場合、下図の潜在空間の任意の点(潜在コード)を入力とし、顔画像を生成します。顔編集のプロセスとしては、参照する実際の画像に対応する潜在コードを探索するInversionをおこない、発見した潜在コードを移動させることで画像の属性を編集しています。

しかし、ここで問題になってくるのが、StyleGANにおける潜在空間の歪みと編集能力のトレードオフです。実際の画像と生成画像のギャップを歪みとし、一方で、顔の属性編集の容易さを編集能力と言っています。左側が一般的なStyleGANで、右側が論文にて提案されているPivotal Tuningを行った例です。

実際の画像から対応する潜在コードを探索したとします。画像Aを生成する潜在コードを選択した場合、実際の画像とAにはギャップがあるため歪みが大きいです。一方で、潜在空間が高密度なため、少しの潜在コードの編集で顔の属性を編集できます。潜在コードの変化が少ない場合、余計なゴミが混ざりにくいというイメージです。

一方で、画像Bを生成する潜在コードの場合、実際の画像と生成画像が似ており歪みが小さいのですが、潜在空間が低密度で潜在コードを大きく編集する必要があるため、ゴミが混じり、実際に顔の向きを変更した生成画像の右上にゴミ(artifacts)が発生しています。

![](https://storage.googleapis.com/zenn-user-upload/157e30e131d36ab64e504b1d.png)

では、なぜ実際の画像と似た画像に対応した潜在コード付近の潜在空間が低密度かと言うと、当たり前のことですが、実際の画像が学習データに含まれていないためだと考えられます。実際、下図(論文中Figure.4)の左端のようにメイクが濃い場合や顔以外の体の一部が写っている場合などの難しい例では、学習データが少ないため、実際の画像に対応した潜在コードから画像を生成した場合でも、学習に使用したデータに基づいて生成してしまい、実際の画像とはギャップが発生してしまいます。そこで、この問題を解決するために提案された手法が**Pivotal Tuning**です。

上図の右側は、Pivotal Tuningを行った例です。StyleGANに対してPivotal Tuningを行うことで、画像Cのように、実際の画像に似た画像を生成する潜在コードを編集能力が良い潜在空間まで遷移させています。この遷移時に、編集能力の質の低下が気になりますが、Pivotal Tuningでは、編集能力を保持したまま生成器をチューニングし、state of the artな顔編集の品質を実現しています。

![](https://storage.googleapis.com/zenn-user-upload/e6c92807eda90f914dc97128.png)


# Pivotal Tuning

一般的な顔編集では、固定した重みを用いたStyleGANで、素晴らしい品質の生成を達成していました。しかし、学習に使用していないデータに関する潜在空間の分布は、編集能力に欠けています。そこで、Pivotal TuningによりStyleGANをチューニングすることで、対象の画像の潜在コードを編集能力の高い潜在コードに遷移させ、高性能な編集性能を達成します。

以下、詳細な手法に関して説明します。

## 1. Inversion

まず、編集能力の高い潜在コードを探索するInversionを行う必要があります。探索にはLearned Perceptual Image Patch Similarity (LPIPS)による知覚損失を使用します。これは、Perceptual path lengthと言う潜在変数の変化の滑らかさを表す指標でよく使用されます。LPIPS損失($\mathcal{L}_{LPIPS}$)は、VGGなどのモデルにより推定した画像の特徴量のL2距離により計算されます。LPIPS損失を含む以下の式を用いて潜在コード$w$とノイズ$n$を最適化することで、対象とする潜在コード$w_p$を計算できます。

$$
w_p, n = \mathrm{argmin}_{w,n} \mathcal{L}_{LPIPS}(x, G(w, n; \theta)) + \lambda_n \mathcal{L}_n(n)
$$

$G$は$\theta$を重みとした生成器による生成画像で、$x$は対象となる実際の画像です。$\lambda_n$, $\mathcal{L}_n$はノイズに関する正則化の割合と損失です。ノイズ正則化は、ノイズ情報の影響を防ぎinversionの性能を改善するために使用されています。

## 2. Pitovatl tuning

pivotal tuningは、Inversionにより求めた潜在コード$w_p$を用いて、生成器$G$の重み$\theta^{\star}$を学習します。損失は、LPIPS損失と実画像と生成画像のL2誤差$\mathcal{L}_{L2}$を用いて、以下の式になります。

$$
\mathcal{L}_{pt} = \mathcal{L}_{LPIPS}(x, G(w_p; \theta^{\star})) + \lambda_{L2} \mathcal{L}_{L2}(x, G(w_p; \theta^{\star}))
$$

しかし、この損失では、大幅に潜在コードを修正し、逆に視覚的な品質が損なわれることがあります。そこで、局所的な修正に留めるための損失を追加する必要があります。論文では、まず、ガウス分布に則ったランダムなベクトル$z$とStyleGANの射影ネットワーク$f$を用いて計算される潜在コード$w_z = f(z)$を用いて、pivotal tuningの潜在コード$w_p$をわずかにシフトさせた潜在コード$w_r$を計算しています。

$$
w_r = w_p +  \alpha \frac{w_z - w_p}{||w_z - w_p||_2}
$$

そして、この潜在コードを用いてオリジナルの生成器とチューニング中の生成器の画像を比較しています。

$$
\mathcal{L}_{R} = \mathcal{L}_{LPIPS}(x_r, x_r^{\star}) + \lambda_{L2} \mathcal{L}_{L2}(x_r, x_r^{\star})
$$

ここでは、オリジナル生成器による画像を$x_r = G(w_r; \theta)$、チューニング中の生成器による画像を$x_r^{\star} = G(w_r; \theta^{\star})$としています。
この損失により、オリジナル生成器とチューニング中の生成器において、わずかに遷移させた潜在コードを入力として生成した顔編集画像の誤差を小さくするように学習できます。つまり、顔編集を行った際の潜在コードがオリジナルの生成器から大きくズレないように学習しています。これにより、視覚的な品質の劣化を防いでいます。

上記をまとめて、Pivotal tuningにより学習される重み$\theta^{\star}$は、

$$
\theta^{\star} = \mathrm{argmin}_{\theta^{\star}} \mathcal{L}_{pt} + \lambda_R \mathcal{L}_{R} 
$$

により最適化されます。


# まとめ

本記事では、学習済みのStyleGANを対象画像に対してわずかにチューニングすることで、ロバストな編集が可能にするPivotal tuningについて解説しました。学習後に、対象の画像に対してチューニングを行うという処理は、挙動が分かり辛いため実際の機械学習システムに取り入れるのは難しいと思っていました。しかし、チューニングの度合いを制限する処理に関しても提案されており、実際、かなり精度が向上していたこともあり、未知のデータに対する対処法の一つとしてシステムに取り入れても良いかもしれないと思いました。

