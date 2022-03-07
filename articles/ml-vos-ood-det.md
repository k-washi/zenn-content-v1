---
title: "VOS: Learning What You Don't Know by Virtual Outlier Synthesis の解説・実装" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

# はじめに

こんにちは、わっしーです。
画像・自然言語・音声に関する機械学習の研究開発やMLOpsを行っています。もし、機械学習に関して、ご相談があれば、[@kwashizzz](https://twitter.com/kwashizzz)のアカウントまでDMしてください！
これまでの、[機械学習記事のまとめ](https://zenn.dev/kwashizzz/articles/my-ml-articles-summary)です。

本記事では、Out-of-Distribution (OOD)検出の新しい手法である[VOS: Learning What You Don't Know by Virtual Outlier Synthesis](https://openreview.net/forum?id=TW7d65uYu5M)に関して紹介します。OOD検出とは、例えば、犬と猫の分類モデルに対して人の画像を入れた場合、学習したクラス以外と認識するタスクのことです。

今までの手法は、実際のクラス外のデータが必要で、データ作成にコストがかかったり、そもそも取得不可能場合もありました。しかし、VOS(Virtual Outlier Synthesis)では、クラス外のデータは必要ありません。そのため、かなり実用的な手法だと思い、記事にしました。

また、本記事では、犬猫のクラス分類を例に、実際の実装も紹介します。

# VOSについて

手法の概要は、論文中Fig.2の以下の図です。

![](https://storage.googleapis.com/zenn-user-upload/ece1e7d35ae6-20220302.png)

物体検出の例ですが、牛が学習データに含まれておらず、一般的な物体検出モデルでは、通行人と誤って検出されてしまいましたが、VOSによりOODと検出されています。基本的には、一般的な物体検出と同じですが、VOSは以下の特徴があります。

1. VOSは、In-Distribution(ID)データ(学習データ)とOODデータ(外れ値データ)を用いて、外れ値かどうか2値分類を行うモデルを学習しています。ここで、外れ値のデータとして、モデルの最終FC層の一つ前の層(Classification Head部分)から、仮想的な外れ値(Virtual Outliers)を合成しています。GANで外れ値画像を合成することも考えられますが、より低次元の特徴空間で合成することで、効率良いOOD検出の学習が可能になっています。

2. 論文では、分類などのタスクに加えて、OOD検出タスクも同時に学習する損失関数が提案されています。

3. 画像分類、物体検出どちらでも使用可能です。今までのアプローチと異なり、VOSでは、物体検出におけるOODにも取り組んでいます。物体検出でOODを行うことで、多数の物体から構成された画像でも部分的なOOD検出を可能にしています。

4. VOSは画像検出・物体検出のモデルの分類部分を一部拡張するだけで使用でき、特徴抽出部分を修正する必要がないため、事前学習済みのモデルが使用できます。

物体検出バージョンは、画像分類の単純な拡張なため、本記事では、画像分類メインで説明します。

# 外れ値の合成手法

一般的な学習方法では、In-Distribution(ID)データのみで最適化されるため、下図左(論文中Fig.1 (b))のように、IDがガウス分布から形成されるが、IDから離れた部分を過信するような結果となることが多々あります。理想的には、下図右のように、IDデータに対してスコアが高く、それ以外のOOD領域では、スコアが低くなるようにして、OOD検出の決定境界を学習したいです。そこで、仮想外れ値の合成が必要となります。

![](https://storage.googleapis.com/zenn-user-upload/f93a6fcd4392-20220307.png)


仮想外れ値の合成では、物体インスタンスの特徴表現が、クラス条件付き多変量ガウス分布を形成していると仮定しています。ガウス分布の平均と分散は、特徴表現(ニューラルネットワークの最後から2番目の層)のサンプルから計算できます。

$$
\begin{equation}
\hat{\mu_k} = \frac{1}{N_k} \sum_{i: y_i = k} h(x_i, b_i) \tag{1}
\end{equation}
$$

$$
\begin{equation}
\hat{\Sigma} = \frac{1}{N} \sum_k \sum_{i: y_i = k} (h(x_i, b_i) - \hat{\mu_k})(h(x_i, b_i) - \hat{\mu_k})^T \tag{2}
\end{equation}
$$

後ほど、解説しますが、実装では、クラス条件付き待ち行列$｜Q_k｜$に、特徴表現を溜めて行き、エポックの途中からOOD検出の学習のため、溜めた特徴表現を用いてこれらのパラメータをオンライン推定しています。

この多変量ガウス分布の尤度が低い部分から、サンプリングした値が仮想外れ値と考えられます。そこで、以下の式のように、閾値($\epsilon$)より低い尤度の領域から、サンプリングするようにしています。


$$
\begin{equation}
\mathcal{V}_k = \{  v_k | \frac{1}{(2 \pi)^{m/2} |\hat{\Sigma}|^{1/2}}  \exp( - \frac{1}{2} (v_k - \hat{\mu}_k)^T \hat{\Sigma}^{-1} (v_k - \hat{\mu}_k)) < \epsilon  \} \tag{3}
\end{equation}
$$

このサンプリングされるものは、ニューラルネットワークの最後から2番目の層に一致する特徴表現です。そのため、最終fc層の重み$W_{cls}^T$を用いて、以下の式で、分類器の出力が得られます。

$$
\begin{equation}
f(v; \theta) = W_{cls}^T v \tag{4}
\end{equation}
$$