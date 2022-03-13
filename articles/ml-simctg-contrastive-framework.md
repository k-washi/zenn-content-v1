---
title: "SimCTG: テキスト生成におけるContrastive学習推論の解説・実装" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習", "python"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

# はじめに

こんにちは、わっしーです。
画像・自然言語・音声に関する機械学習の研究開発やMLOpsを行っています。もし、機械学習に関して、ご相談があれば、[@kwashizzz](https://twitter.com/kwashizzz)のアカウントまでDMしてください！
これまでの、[機械学習記事のまとめ](https://zenn.dev/kwashizzz/articles/my-ml-articles-summary)です。

本記事では、文章生成のモデルの欠点である生成テキストの不自然さや、望ましくない単語の繰り返しを抑える手法として、[A Contrastive Framework for Neural Text Generation](https://arxiv.org/abs/2202.06417)で提案されたSimCTG(a **SIM**ple **C**ontrastive framework for neural **T**ext **G**eneration)を紹介します。

また、[論文の実装コード](https://github.com/yxuansu/SimCTG/tree/bb54480e5c43d62d5b660d5cdabaee7c7d7af442)を参考にし、Encoder-Decoder形式の文章生成モデルであるT5にSimCTGを適用してみたので、実装方法を紹介します。

※ [論文の実装コード](https://github.com/yxuansu/SimCTG/tree/bb54480e5c43d62d5b660d5cdabaee7c7d7af442)には、GPT-2に適用した実装があるので、GPT-2を使いたい人は、[こちら]((https://github.com/yxuansu/SimCTG/tree/bb54480e5c43d62d5b660d5cdabaee7c7d7af442)を参考にしてください。

# なぜSimCTGが必要なのか?

> 昨今のDecoderを持つモデルでは、入力文＋過去に生成した単語列から次の単語列を予測するAuto-Regressiveな生成が行われています。この際に、計算量の関係から、Gready SearchやBeam Searchなどが用いられています。

参考: [テキスト生成における decoding テクニック: Greedy search, Beam search, Top-K, Top-p
](https://zenn.dev/hellorusk/articles/1c0bef15057b1d)

しかし、これらの探索手法を用いた場合、同じ単語を繰り返すなどの問題が発生します。そこで、`huggingface`の文章の`generete`関数では、`no_repeat_ngram_size`や`repetition_penalty`という引数があり、このパラメータによって繰り返しを抑えることができますが、パラメータによっては、同じ単語列が出なくなり不自然な生成になってしまうことがあります。

下図(論文中Fig.1)の(a)は、GPT-2によって生成された各トークンにおける特徴ベクトル(Transformer最終層)のコサイン類似度行列です。見ての通り、文中のトークン間の類似度が0.95以上になっており、互いの特徴ベクトルが近いことがわかります。当然ですが、これは、異なるステップで繰り返しトークンを生成してしまう原因となります。論文中では、このように特徴ベクトルが偏っていることを異方性と言っています。

![Fig.1](https://storage.googleapis.com/zenn-user-upload/c51f3beef16d-20220313.png)

理想的には、生成されたトークンの特徴ベクトル間を識別可能にするため、上図(b)のようにトークン間の類似度が低くなるようにすべきです。つまり、各トークンの特徴ベクトルは偏らないようにする必要があります。（等方性にすべき。）

そこで、最尤推定（MLE）で言語モデルを学習し、最も可能性の高いシーケンスをデコードしていくという従来手法に対し、識別可能で等方的な特徴ベクトルの学習を促すSimCTG損失を提案し、加えて、SimCTGによる学習を補完する新しい復号化手法であるContrastive Search (対照探索)を提案した。

# SimCTG

SimCTGは、最尤推定の損失（クロスエントロピー損失)と対照損失(Constrastive Loss)からなります。

単語列を$x$とした場合、最尤推定の損失は、以下のようになります。一つ前までの単語列($x_{ < i}$)から生成される単語が$x_i$である確率分布 $p_{\theta}(x)$を最大化するような損失$\mathcal{L}_{MLE}$としてます。

$$
\begin{equation}
\mathcal{L}_{MLE} = - \frac{1}{|x|} \sum_{i=1}^{|x|} \log p_{\theta} (x_i | x_{ < i})
\tag{1}
\end{equation}
$$

次に対照損失$\mathcal{L}_{CL}$は、類似度関数$s$を用いて、以下の式になります。これは、同じ単語の特徴ベクトルの類似度を高くし、異なる単語の特徴ベクトルの類似度を小さくするような損失です。

$$
\begin{equation}

\mathcal{L}_{CL} = \frac{1}{|x| \times (|x| - 1)} \sum_{i=1}^{|x|} \sum_{j=1, j \neq i}^{|x|} 
\max \{ 0, \rho - s(h_{x_i}, h_{x_i}) + s(h_{x_i}, h_{x_j}) \}

\tag{2}
\end{equation}
$$

マージン $\rho \in [-1, 1]$はハイパーパラメータで、$h_{x_i}$がトークン($x_i$)の特徴ベクトルです。類似度関数は、

$$
\begin{equation}

s(h_{x_i}, h_{x_j}) = \frac{h_{x_i}^Th_{x_j}}{||h_{x_i}|| \cdot || h_{x_j} ||}

\tag{3}
\end{equation}
$$

で計算できます。

上記の損失を用いてSimCTG損失は、

$$
\begin{equation}

\mathcal{L}_{CTG} = \mathcal{L}_{MLE} + \mathcal{L}_{CL}

\tag{4}
\end{equation}
$$

となります。

このSimCTG損失を用いて学習することで、同じトークンの特徴ベクトルは近くになり、異なるトークン間の特徴ベクトルは遠くなります。

# Contrastive Search

対照損失を用いたSimCTGで学習したモデルを用いて次の単語を予測する場合、一つ前までの単語列とは異なる単語である可能性が高いです。その知識を取り入れ、従来手法と同様に確率の高い単語を探索することに加え、一つ前までの単語列と類似度が低い単語を探索するContrastive Searchが提案されています。

予測確率が高い単語の$k$個の集合$V^k$とした場合のContastive Searchによる単語選択は、

$$
\begin{equation}

x_t = \mathrm{argmax}_{v \in V^k} \{ (1 - \alpha) \times p_{\theta}(v | x_{ < t}) 
- \alpha \times (\mathrm{max} \{ s(h_v, h_{x_j}) : 1 \leq j \leq t - 1 \}) \}

\tag{5}
\end{equation}
$$