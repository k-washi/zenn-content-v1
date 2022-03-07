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

次に、上記の方法で仮想的な外れ値を合成できるとし、どのような損失で学習するのか解説します。

# VOSの損失

論文では、IDに対しては低いOODスコアを、合成された外れ値に対しては、高いOODスコアを生成するように、モデルを正則化しながら学習する損失を提案しています。

物体検出バージョンは、画像分類の単純な拡張なため、簡単のため、画像分類メインで説明します。

## 分類のための不確かさ正則化

理想的には、データ密度$\log p(x)$を捉える何らかの関数により、IDとOODデータ間の分離境界を最適化する損失を設定する必要があります。
しかし、$\log p(x)$を直接推定することは、空間$X$全体からサンプリングする必要があるため、計算上困難です。そこで、Energy-based Out-of-distribution Detection](https://arxiv.org/abs/2010.03759)で言及されているログ分配関数を用いることが考えられます。

ログ分配関数は、自由エネルギーとも呼ばれ、OOD検出のための有効な不確かさ測定であることが示されています。ラベル$y$に対応するロジット出力を$f_y (x; \theta)$とした時、$p(y | x) = \frac{p(x, y)}{p(x)} = \frac{e^{f_y (x; \theta)}}{\sum_{k=1}^{K} e^{f_k (x; \theta)}}$ となり、$\log p(x)$に比例するらしいです。

VOSでは、合成された外れ値に対して正のエネルギーを持ち、IDデータに対して負のエネルギー値を持つ、エネルギー関数に基づく不確かさ損失を定義しています。

$\mathcal{L}_{uncertainty} = \mathbb{E}_{v \sim V} \mathbb{1} \{ E(v; \theta) > 0 \} + \mathbb{E}_{x \sim D} \mathbb{1} \{ E(x; \theta) \leq 0 \}$

これは、密度を推定するよりもシンプルな損失ですが、0/1損失の設計は困難なため、以下の式のように、滑らかに近似したバイナリシグモイド損失に置き換えた損失をVOSでは使用しています。

$$
\begin{equation}
\mathcal{L}_{uncertainty} = \mathbb{E}_{v \sim V} [ -\log \frac{1}{1 + \exp^{- \theta_u E(v; \theta)}} ] + \mathbb{E}_{x \sim D} [ -\log \frac{\exp^{- \theta_u E(x; \theta)}}{1 + \exp^{- \theta_u E(x; \theta)}} ]   \tag{5}
\end{equation}
$$

今までのアプローチと比較し、上記の不確かさ損失は、ハイパーパラメータフリーで使用が容易で、かつ、OOD検出時には、シグモイド関数により、確率的スコアを生成することができます。

もし、物体検出に使用する場合は、バウンディングボックスも入力として、以下の式で表されます。

$$
\begin{equation}
E(x, b; \theta) = - \log \sum_{k=1}^K w_k \cdot e^{f_k(x, b;\theta)}  \tag{6}
\end{equation}
$$

そして、VOSでは、物体検出、物体分類の損失と組み合わせ、以下の損失で最適化が行われます。

$$
\begin{equation}
min_{\theta} \mathbb{E}_{(x, b, y) \sim D} [ \mathcal{L}_{cls} +  \mathcal{L}_{loc}] + \beta \cdot  \mathcal{L}_{uncertainty}  \tag{7}
\end{equation}
$$
