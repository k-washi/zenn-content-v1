---
title: "Flow Matching - 拡散モデルにかわる生成技術" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習", "論文解説", "Diffusion", "FlowMatching"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
publication_name: "fusic"
---

こんにちは！[Fusic](https://fusic.co.jp/) 機械学習チームの鷲崎です。最近、音声や言語処理に興味がありますが、機械学習モデルの開発から運用までなんでもしています。もし、機械学習で困っていることがあれば、気軽に[DM](https://twitter.com/kwashizzz)ください。

本記事では、Flow Matching (FM)と、その発展版であるOptimal Transport Conditional Flow Matching (OT-CFM)を解説します。最近の生成AIでは、拡散モデルがよく使用されていますが、Flow Matchingは、拡散モデルに取って代わる可能性がある生成技術と考えています。

おもに、[Improving and Generalizing Flow-Based Generative Models with Minibatch
Optimal Transport](https://arxiv.org/abs/2302.00482)という論文を参考に解説していきたいと思います。また、本記事の図は、この論文から参照いたしました。

Flow Matachingは、音声合成界隈では、Metaが発表した音声合成手法である[Voicebox](https://voicebox.metademolab.com/)や、より高速で高性能な音声合成手法である[Matcha-TTS](https://arxiv.org/abs/2309.03199)に使用されてきています。
Matcha-TTSに関しては、弊社のToshikiが、記事を書いているので、ぜひ、読んでみてください。

[【音声合成】Matcha-TTS🍵で日本語音声を生成してみる](https://zenn.dev/fusic/articles/bd7da12adf5901)

個人的に、Flow Matchingは、拡散モデルが発展した手法と考えて論文を読み始めたのですが、それだとかなり勘違いしてしまっていました。
数年前に拡散モデルとは別に、Normalize Flowという手法が流行った気がします。その、Normalize Flowの発展版と考えるとスムーズに納得できました。
そこで、本記事では、参考になる記事などを示しつつ、拡散モデルとNormalize Flowの関係性からFlow Matchingまでの過程を解説していきたいと思います。

理解が及ばなかった部分が多々あるため、ご指摘をたくさんいただけると嬉しいです。よろしくお願いいたします。

# 基礎

Flow Matchingを理解するために、拡散モデルの拡散過程を連続時間化した、確率微分方程式(SDE)を説明します。
そして、それを常微分方程式(ODE)にに変換した確率フローODEを説明します。

その後、確率フローODEが論文で解説されているNormalize Flow (NF)を連続化したContinuous Normalize Flow (CNF)と、どのように関連するか説明します。

SDEとODEの詳細に関しては、[拡散モデル データ生成技術の数理](https://www.amazon.co.jp/dp/400006343X)という本を読んでください。

## 拡散モデルの確率微分方程式(SDE)

拡散モデルでは、拡散・逆拡散過程のステップ数を増やせば増やすほど、離散化誤差が小さくなり生成性能が向上します。
そこで、このノイズを付加する過程のステップ数を無限大に増やして連続化した場合の拡散モデルを考えます。
そのとき、このモデルは確率微分方程式(SDE)とみなすことができます。

拡散過程に相当する拡散SDEは、以下の式で与えられます。

$dx = f(x, t)dt + g(t)dw$

$dx$はデータ$x$の微小時間あたりの変化量です。この変化量は、決定的に変化する量$f(x, t)dt$と、ランダムに変化する量$g(t)dw$からなります。$f(x, t)$は、データ$x$と時刻で決まります。また、$w$は、標準ウィーナー過程と呼ばれ、$dw$は、微小時間間隔$\tau$において、平均$0$、分散$\tau$となる正規分布です。

一方で、逆拡散過程は、以下のSDEで与えられます。

$dx = \lbrack f(x, t) - g^2(t) \nabla_x \rm{log} p_t (x) \rbrack dt + g(t) d\bar{w} $

時刻$t$のスコア$\nabla_x \rm{log} p_t (x)$が、わかりさえすれば、事前分布$p_T(x)$から、データ分布$p_0(x)$への変換経路を求めることができます。

### 例えば、Denoising Diffusion Probabilistic Model (DDPM)の場合、以下のようなSDEになります。

元のDDPMの式は、以下です。

$x_{t+1} = \sqrt{1 - \beta_{t}}x_{i-1} + \sqrt{\beta_i} z_{i-1}  $

この式を、SDEにしたDDPMの式は以下です。

$dx = -\frac{1}{2}\beta(t)xdt + \sqrt{\beta(t)}dw$

ここでの、$\beta$は、ノイズスケジュールのパラメータです。

## 確率フロー常微分方程式 (ODE)

任意のSDEは、同じ周辺分布$\lbrace p_t(x) \rbrace_{t\in \lbrack 0,T \rbrack}$を持つ常微分方程式(ODE)に変換することができます。SDEの拡散過程・逆拡散過程は、以下の式に変換されます。

$dx = \lbrack f(x, t) - \frac{1}{2}g^2(t) \nabla_x \rm{log} p_t (x) \rbrack dt$

確率的な要素が除外され、拡散過程では$dx$, 逆拡散過程では$-dx$のように、同じ式で取り扱うことができます。

## Normalizing Flow

Normalizing Flow (NF)は、複雑な分布を、より単純な分布へ変数変換（Normalize)する関数のことです。

例えば、シンプルな確率密度関数$p_Z(z)$と、複雑な確率密度関数$p_Y(y)$の間の変数変換を行うとします。そのとき、$y$から$z$への変数変換を$z=f(y)$としたとき、$p_Y(y)$は、以下の式で表せます。

$p_Y(y) = p_Z(z) | \frac{\partial f}{\partial y} |$

NFでは、この変数変換を行う関数$f$は、可逆であるという性質を満たします。

$g = f^{-1}$

また、NFでは、この性質をもつ関数$f_i$による合成関数で、変数変換の式を定義しています。

$f = f_1 \circ f_2 \circ \dots \circ f_N$

$g = f^{-1} = f^{-1}_N \circ \dots \circ f^{-1}_1$

ざっくりしたイメージですが、$f_i$で徐々に分布を変換していくのが、ノイズを除去していく拡散モデルに似ている気がしました！！

## Continuous Normalizing Flow

[Neural Ordinary Differential Equations (Neural ODE)](https://arxiv.org/abs/1806.07366) という論文で、連続時間化したNormalizing Flow を Continuous Normalizing Flow (CNF)と呼んでいました。

Neural ODEでは、Residual Networkのように、残差を計算の結果を元の値に付加することで値を更新する方法に着目し、以下のように常微分方程式の式に似た概念を導入しています。

$x_{t+1} = x_t + \frac{df(x)}{dt} = x_t + f(x, t)$

この表現を用いる利点は、変化量の微分を表す単一のネットワークを使用すればよく、メモリの使用量を削減できます。加えて、逆変換の計算が可能で、RNNなど離散的な時間を扱うモデルとはことなり、連続時間を扱うことができます。

※　これの特殊系が確率フローODEになります。

**これを、CNFに発展させたものが以下になります。**

このNeural ODEの考えで、Normalizing Flowを取り扱うと、以下のように、確率変数$x_0$から$x_1$の変化を、連続時間の変化量の積分で操作することができます。

$x_1 = x_0 + \int^{t_1}_{t_0} f(x(t), t) dt$

この微分は、$dx/dt = f(x(t), t)$で、どの時間$t$においても、$f(x(t), t)$という単一のネットワークで表現されています。

また、この変換の逆変換も、以下のように書くことができます。

$x_0 = x_1 + \int^{t_0}_{t_1} f(x(t), t) dt$

上記のことから、拡散モデルの拡散過程と逆拡散過程に類似した表現をすると、拡散過程にあたるODEは、

$dx = f(x, t) dt$

となり、逆拡散過程にあたるODEは、

$dx = -f(x, t) dt$

のように表現できます。

## なぜ、拡散モデルがよいとされているのか？

拡散モデルのDDPMの損失は、以下になります。

$\rm{E}_{t, q(z), p_t(x|z)} || s_{\theta}(t, x) - \nabla_x \rm{log} p_t(x|z)||^2_2$

これから分かる通り、拡散モデルは、各時刻ごとの生成過程を独立で学習可能です。これは、複数のステップを繰り返すことで巨大な計算過程を学習でき、かつ、計算グラフの一部分を抜き出して学習できるという利点があります。このようなアプローチをSimulation Freeと呼びます。

一方で、CNFの損失は、以下になります。

$ \rm{E}_{q(x_1)} \lbrack \rm{log} p_0(x_0) - \int \rm{Tr}(\frac{\partial f}{\partial x_t}) dt \rbrack$

見ての通りですが、全時刻に渡って積分した値を使用しており、計算グラフのすべてを使って学習が必要となる欠点があります。このようなアプローチは、Simulation based trainingと呼ばれます。これが原因で、学習効率が悪く、拡散モデルほど使用されていません。

# Flow Matching

CNFを安定して学習可能なFlow Matchingについて解説します。
上記のように、時間全体の学習が必要である点が、CNFの欠点と言えます。
そこで、CNFを時刻ごとに、学習可能にするために、Flow Matchingが、開発されました。

まず、微小時変ベクトル場$u$のODEは、以下になります。

$dx = u_t (x) dt $

$u_t(x)$は、$u(t, x)$と同様です。初期条件$\phi_0 (x) = x$のODEの解は、積分写像$\phi_t$となります。$\phi_t(x)$が時刻$0$から$t$までベクトル場$u$に沿って輸送される点$x$を表現しています。また、密度$p_t$は、時刻$0$から時刻$t$まで$u$に沿って輸送される点$x \sim p_0$の密度です。
そして、時変密度$p_t$は、$\partial p / \partial t = - \nabla \cdot (p_t u_t)$と、初期条件$p_0$の条件下で、$p$は$u$による確率経路で、$u$は$p$の確率フローODEとなります。

このとき、Flow Matchingは、ニューラルネットワーク$v_{\theta}(t, x)$が、時間依存のベクトル場$u_t(x)$に回帰するような損失を持ち、以下のようになります。

$L_{FM}(\theta) = \rm{E}_{t\sim \mathcal{U}(0,1), x\sim p_t(x)} ||v_{\theta}(t, x) - u_t(x)||^2$

[Flow Matching for Generative Modeling](https://openreview.net/forum?id=PqvMRDCJT9t) では、正規分布によるガウス確率経路$p_t(x) = \mathcal{N}(x | u_t, \sigma_t^2)$を考えています。それは、積分写像が$\phi_t(x) = \mu_t + \sigma_t x$を満たすとき、ベクトル場は、以下のようになるとしています。

$u_t(x) = \frac{\sigma_t^{\prime}}{\sigma_t} (x - \mu_t) + \mu_t^{\prime}$

Flow Matchingにより、このベクトル場を回帰するように$v_{\theta}(t, x)$を学習することで、CNFの学習を安定して行えるようになりました。


# Conditional Flow Matching

Flow matchingは、ガウス確率経路を仮定していました。そこで、ガウス分布の仮定を緩和し、2つの分布間の条件付き確率経路(ODE Bridge)の学習を可能にした、Conditional Flow Mataching (CFM)が提案されました。

潜在条件変数$z$において、潜在変数による分布$q(z)$と確率経路$p_t(x|z)$による周辺確率経路$p_t(x)$は、以下の式となります。

$p_t(x) = \int p_t(x|z)q(z)dz$

もし、初期条件分布$p_0 (x|z)$からベクトル場$u_t(x|z)$によって確率経路$p_t(x|z)$が生成されるとすると、周辺ベクトル場$u_t(x)$は、

$u_t(x) := \rm{E}_{q(z)} \frac{u_t(x|z)p_t(x|z)}{p_t(x)}$

となる。そして、周辺ベクトル場$u_t(x)$によって、初期条件$p_0(x)$から周辺確率経路$p_t(x)$が生成されます。
そのため、条件付き確率経路$p_t(x|z)$とベクトル場$u_t(x|x)$から$u_t$を計算したいが、分母の$p_t(x)$の計算は積分を含み困難です。

そこで、CFMの損失を

$\rm{L}_{CFM}(\theta) = \rm{E}_{t, q(z), p_t(x|z)} ||v_{\theta}(t, x) - u_t(x|z)||^2$

としたとき、特定の条件下で、

$\nabla \rm{L}_{CFM}(\theta) = \rm{L}_{FM}(\theta)$

であることが論文で示されました。
つまり、条件付きベクトル場$u_t(x|z)$を計算できるなら、周辺ベクトル$u_t(x)$に回帰するニューラルネット$v_{\theta}$を学習できることを示しました。

これをCNFとし、以下のアルゴリズム(論文中 Algorithm 1)で計算されます。

![](https://storage.googleapis.com/zenn-user-upload/eebf91be8324-20231207.png)


これより下の項目では、$q(z)$, $p_t(-, z)$, $u_t(-|z)$によって定義される様々なCFMについて説明します。


# Flow Matching from a Gauusian

[Flow Matching for Generative Modeling](https://openreview.net/forum?id=PqvMRDCJT9t) で説明されたFlow MatchingをCFMの特殊なケースとしての解釈を説明します。
この論文では、$z = x_1$とし、標準正規分布$p_0(x|z) = \mathcal{N}(x; 0, 1)$から、$p_1(x|z) = \mathcal{N}(x; x_1, \sigma^2)$への確率経路と設定しています。

つまり、確率経路は、

$p_t(x|z) = \mathcal{N}(x | tx_1, (t_{\sigma} - t + 1)^2)$

となり、ベクトル場は、

$u_t(x|z) = \frac{x_1 - (1-\sigma)x}{1 - (1 - \sigma)t}$

となります。

下図は、論文中のFigure 1で、Flow Matchingのイメージです。ガウス分布がデータサンプルへ分散を小さくしながら遷移している確率経路が見えますね。

![](https://storage.googleapis.com/zenn-user-upload/6e79e34c8494-20231207.png)


# Independet CFM

# Optimal Transport CFM

# Schrödinge Bridge CFM

![](https://storage.googleapis.com/zenn-user-upload/7ed690f1dc10-20231207.png)

![](https://storage.googleapis.com/zenn-user-upload/c8b7ec11bd22-20231207.png)
![](https://storage.googleapis.com/zenn-user-upload/671c2e40a4e3-20231207.png)
![](https://storage.googleapis.com/zenn-user-upload/7c7d8cd1eec4-20231207.png)

![](https://storage.googleapis.com/zenn-user-upload/d1b0965ec0d2-20231207.png)
![](https://storage.googleapis.com/zenn-user-upload/86fc1ad56589-20231207.png)
![](https://storage.googleapis.com/zenn-user-upload/d02dd2a714f5-20231207.png)



# 参考文献
- [Normalizing Flow入門 第1回 変分推論](https://tatsy.github.io/blog/posts/2020/2020-12-30-normalizing_flow%E5%85%A5%E9%96%80_%E7%AC%AC1%E5%9B%9E/)
- [Normalizing Flow入門 第2回 Planar Flow](https://tatsy.github.io/blog/posts/2021/2021-01-03-normalizing_flow%E5%85%A5%E9%96%80_%E7%AC%AC2%E5%9B%9E/)
- [Normalizing Flow入門 第7回 Neural ODEとFFJORD](https://tatsy.github.io/blog/posts/2021/2021-01-11-normalizing_flow%E5%85%A5%E9%96%80_%E7%AC%AC7%E5%9B%9E/)
- [Morpho Tech log - A Brief Survey of Schrödinger Bridge (Part I)](https://techblog.morphoinc.com/entry/2023/09/12/100000)