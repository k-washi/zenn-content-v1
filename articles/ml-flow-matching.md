---
title: "Optimal Transport Conditional Flow Matching - 拡散モデルに取って代わる次世代の生成技術？" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習", "論文解説", "拡散モデル", "FlowMatching"] # タグ。["markdown", "rust", "aws"]のように指定する
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
皆さん、Normalize Flowという手法を聞いたことがありますでしょうか？現在、拡散モデルがバズワード化しすぎて、あまり耳にしないかもしれませんが、分布を変換する関数を学習するとてもおもしろいモデルです。そのNormalize Flowの発展版がFlow Matchingと考えるとスムーズに納得できた気がします。
そこで、本記事では、拡散モデルとNormalize Flowの関係性からFlow Matchingまでの過程とその発展を解説していきたいと思います。

理解が及ばなかった部分が多々あるため、ご指摘をたくさんいただけると嬉しいです。よろしくお願いいたします。

# 基礎

Flow Matchingを理解するために、まず、拡散モデルの拡散過程を連続時間化した、確率微分方程式(SDE)を説明します。
そして、それを常微分方程式(ODE)に変換した確率フローODEを説明します。

その後、確率フローODEが、Normalize Flow (NF)を連続化したContinuous Normalize Flow (CNF)と、どのように関連するか説明します。

SDEとODEの詳細に関しては、[拡散モデル データ生成技術の数理](https://www.amazon.co.jp/dp/400006343X)という本がとても参考になります！ぜひ読んでみてください。

## 拡散モデルの確率微分方程式(SDE)

拡散モデルでは、拡散・逆拡散過程のステップ数を増やせば増やすほど、離散化誤差が小さくなり生成性能が向上します。
そこで、ステップ数を無限大に増やして連続化した場合の拡散モデルを考え、このモデルは確率微分方程式(SDE)とみなすことができます。

拡散過程に相当する拡散SDEは、以下の式で与えられます。

$dx = f(x, t)dt + g(t)dw$

$dx$はデータ$x$の微小時間あたりの変化量です。この変化量は、決定的に変化する量$f(x, t)dt$と、ランダムに変化する量$g(t)dw$からなります。$f(x, t)$は、データ$x$と時刻$t$で決まります。また、$w$は、標準ウィーナー過程と呼ばれ、$dw$は、微小時間間隔$\tau$において、平均$0$、分散$\tau$となる正規分布です。

一方で、逆拡散過程は、以下のSDEで与えられます。

$dx = f(x, t) -  (t)  (x) dt + g(t) d$

$dx = [ f(x, t) - g^2(t) \nabla_x \rm{log} p_t (x) ] dt + g(t) d \bar{w}$

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

$p_t(x|z) = \mathcal{N}(x | tx_1, (t \sigma - t + 1)^2)$

となり、ベクトル場は、

$u_t(x|z) = \frac{x_1 - (1-\sigma)x}{1 - (1 - \sigma)t}$

となります。

下図は、論文中のFigure 1で、Flow Matchingのイメージです。ガウス分布がデータサンプルへ分散を小さくしながら遷移している確率経路が見えますね。

![](https://storage.googleapis.com/zenn-user-upload/6e79e34c8494-20231207.png)


# Independet CFM

CFMの基本形として、潜在変数$z$を、初期点$x_0$と、目標点$x_1$で同定し、$q(z) = q(x_0)q(x_1)$としたI-CFMを説明します。
$x_0$と$x_1$の間の確率経路をガウス分布の移動と考えると、以下の確率経路になります。

$p_t(x|z) = \mathcal{N}(x | tx_1 + (1 - t) x_0, \sigma^2)$

そして、この確率経路の平均と分散をベクトル場の定式(flow matchingの項に記載)にあてはめると、ベクトル場は、

$u_t(x|z) =  x1 - x0$

となります。かなりシンプルな形式になっていますが、これにより$p_t(x|z)$は効率的にサンプリングでき、$u_t$は効率的に計算できるため、$\rm{L}_{CFM}$の勾配計算も効率的とのことです。

下図は、論文中のFigure 1で、I-CFMのイメージです。固定の分散の分布が移動している事がわかります。これを見ると、拡散モデルと異なり、分布間の移動を行っているような気がしますね。そもそも、拡散モデルのようにノイズからノイズ除去で生成するより、特定の分布から生成したほうが効率良い気がします。どなんでしょうか...

![](https://storage.googleapis.com/zenn-user-upload/c8b7ec11bd22-20231207.png)

I-CFMのアルゴリズムは、以下になります。これだけ見ると、かなりシンプルですね。

![](https://storage.googleapis.com/zenn-user-upload/d02dd2a714f5-20231207.png)

# Optimal Transport CFM

まず、2-Wasserstein距離による最適輸送(OT)に関して説明します。そして、OTを用いてCFMに関して説明します。

## 2-Wasserstein距離

最適輸送問題は、ある測度から別の測度へのマッピングを、コストが最小化するように求めるものである。論文では、2-Wasserstein距離を用いており、分布$q_0$と$q_1$の間のコスト$c(x, y) = ||x - y||$ を用いて、以下の式で表されます。


$\mathcal{W}(q_0, q_1)^2_2 = \rm{inf} \int_{\mathcal{X}} c(x, y)^2 d \pi (x, y)$

そして、2-Wassrstein距離の動的形式は、ある測度を他の測度に変換するベクトル場$u_t$上の最適化問題としても定義できます。

$\mathcal{W}(q_0, q_1)^2_2 = \rm{inf} \int_{\rm{R}^d} \int^{1}_{0} p_t(x) ||u_t(x)||^2 dt dx$

L2正則化を持つCNFが動的最適輸送に近似可能なことは証明されていますが、この式の計算には、多くの積分とバックプロパゲーションが必要なため数値的にも、効率的にも問題がありました。そこで、CFMとして、直接ベクトル場を回帰することで、これらの問題を回避することが提案されました。

## OT-CFM

上記の2-Wasserstein距離をCFMに当てはめるため、2-Wasserstein最適輸送写像$\pi$を$q(z)$とすることが提案されました。

$q(z) := \pi (x_0, x_1)$

これにより、$q_0$, $q_1$から、独立にサンプリングしていた$x_0$, $x_1$を、最適輸送写像によって、同時にサンプリングします。
I-CFMに対して、この修正を行ったものが、OT-CFMになります。

論文では、

>OT-CFMは、$q(x_0)$と$q(x_1)$の間の静的OT写像と中間時間ステップで条件付きフローの回帰のみを用いて、シミュレーションフリー（時間ごとに学習可能）な動的OT問題を解いた最初の手法である。

とのことです。

下図は、論文中のFigure 1で、OT-CFMのイメージです。I-CFMと異なり、時刻$0$から$1$の分布の移動がOTのおかげでスムーズに見えます。

![](https://storage.googleapis.com/zenn-user-upload/671c2e40a4e3-20231207.png)

実際、論文中の実験では、下図のようにより効率的な分布の移動ができています。下図は、moonから9つのガウス分布の生成を行っていますが、左側のI-CFMより、右側のOT-CFMが明らかにシンプルな遷移を行っています。

![](https://storage.googleapis.com/zenn-user-upload/7ed690f1dc10-20231207.png)

下図は、OT-CFMのアルゴリズムを示しています。

アルゴリズム中にOTのミニバッチ近似が出てきます。大きなデータセットにおいて、OTにおける輸送計画$\pi$の計算と保存は、困難な場合があり、OTのミニバッチ近似を使用するのが実用的です。実際、正確なOTの解に対して誤差が生じますが、多くのアプリケーションに適応可能になります。

実際アルゴリズムを見ると、変わったところは、OTの部分くらいかと思います。

![](https://storage.googleapis.com/zenn-user-upload/7c7d8cd1eec4-20231207.png)



# Schrödinge Bridge CFM

論文中には、シュレディンガーBridge(SB)によるCFMもでてきます。勝手なイメージですが、I-CFMとOT-CFMの中間に当たるイメージを持っていますが、理論が複雑で説明が難しいので省きます。ぱっとみ、SB-CFMより、OT-CFMのほうが精度が良さそうでした。ただ、[Schrodinger Bridges Beat Diffusion Models on Text-to-Speech Synthesis](https://huggingface.co/papers/2312.03491)という論文もでていたので、次回の記事を書く際にしっかり理解したいと思います。（誰か書いて！！）

ちなみにですが、Schrödinge Bridgeに関しては、以下の記事がとても参考になりました。すごく良い記事です！

- [Morpho Tech log - A Brief Survey of Schrödinger Bridge (Part I)](https://techblog.morphoinc.com/entry/2023/09/12/100000)

# 比較

最後に比較結果です。
下図は、分布の適合度（一致度？）を示した2-Wasserstein $\mathcal{W}^2_2$と、最適輸送性能(Normalized Path Energy)で最適輸送性能を比較したものです。値が小さいほど性能がよいです。OT-CFMは、$\mathcal{W}^2_2$もNPEも、良さそうです。


![](https://storage.googleapis.com/zenn-user-upload/d1b0965ec0d2-20231207.png)

また、下図の左より、学習時の検証セットに対する誤差の収束も早いことがわかります。

![](https://storage.googleapis.com/zenn-user-upload/86fc1ad56589-20231207.png)

# プログラム上でどう記載するのか？

理論ばかり書いても、まぁわからないので、いくつか実装例を見てみましょう。

まずは、CFMを使用している音声合成モデルである[Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py)です。ここでは、$x_1$が目標となるデータで、エンコーダの出力であるメルスペクトログラム$mu$をflow matchingでいい感じに変換し$x_1$に近づけるため、$mu$という引数があります。

実装を見ると、ランダムな$z$をサンプリングしていますが、I-CFNという認識でよいのでしょうか。（ただ、[他のCFMの実装](https://github.com/atong01/conditional-flow-matching/blob/21cd0c888186f6e2b76deb393800361b8a850e9b/examples/cifar10/train_cifar10.py#L147C13-L147C13)の初期値も同様になっていました。）また、気になる点としては、確率経路の平均を計算する部分と、ベクトル場を計算する部分に、`sigma_min`が入っています。ただ、この値は、かなり小さい($1e^{-4}$)とかなので、無視しても良いかもしれませんが、実装上必要なのでしょうかね...　([他の実装](https://github.com/atong01/conditional-flow-matching/blob/main/runner/configs/model/otcfm.yaml)では、$0.1$など割りと大きめの値が設定されている場合がありました。)

また、確率経路から、データ$x$をサンプルしていたはずですが、ここでは計算せず、直接ニューラルネットワークに入力しています。こちらも実装ならではなのかなという気がします。

```python:https://github.com/shivammehta25/Matcha-TTS/blob/c8d0d60f87147fe340f4627b84588e812e5fbb00/matcha/models/components/flow_matching.py#L89C5-L120C23
def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        x0 = torch.randn_like(x1)

        # 確率経路の平均計算
        y = (1 - (1 - self.sigma_min) * t) * x0 + t * x1
        
        # ベクトル場の計算
        u = x1 - (1 - self.sigma_min) * x0

        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), spks), u, reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )
        return loss, y
```

これを、OT-CFMに改造してみましょう。実際動作させていないですが、おそらく以下のようになると思います。最適輸送には、[POT: Python Optimal Transport](https://pythonot.github.io/)というライブラリを使用します。

```python
from functools import partial
import ot as pot

class OTCFM()
    def __init__(self, ot_method):
        if ot_method == "exact":
            self.ot_fn = pot.emd
        elif ot_method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif ot_method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif ot_method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)

    def get_map(self, x0, x1):
        """Compute the OT plan (wrt squared Euclidean cost) between a source and a target
        minibatch.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch

        Returns
        -------
        p : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        """
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        return p
    def sample_map(self, pi, batch_size):
        r"""Draw source and target samples from pi  $(x,z) \sim \pi$

        Parameters
        ----------
        pi : numpy array, shape (bs, bs)
            represents the source minibatch
        batch_size : int
            represents the OT plan between minibatches

        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays, shape (bs, bs)
            represents the indices of source and target data samples from $\pi$
        """
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size)
        return np.divmod(choices, pi.shape[1])

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        x0 = torch.randn_like(x1)
        
        # OTを用いて、x0, x1をサンプル
        pi = self.get_map(x0, x1)
        i_arr, j_arr = self.sample_map(pi, x0.shape[0])
        x0, x1 = x0[i_arr], x1[j_arr]

        y = (1 - (1 - self.sigma_min) * t) * x0 + t * x1
        u = x1 - (1 - self.sigma_min) * x0

        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), spks), u, reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )
        return loss, y
```

# 最後に

今回は、Flow Matchingをより効率的にした、OT-CFMについて解説しました。個人的に、数式が多く難しいな～と思いながらも、今後使える技術だと思い、記事にしてみました。この記事で、Flow Matchingの理解の手助けになればと思います。

今後は、シュレディンガーBridgeに関する記事か、実際にOT-CFMを使用した音声合成などの記事を作成できればと思っています。

---

最後に宣伝になりますが、機械学習でビジネスの成長を加速するために、[Fusic](https://fusic.co.jp/)の機械学習チームがお手伝いしています。機械学習のPoCから運用まで、すべての場面でサポートした実績があります。もし、困っている方がいましたら、ぜひ[Fusic](https://fusic.co.jp/)にご相談ください。[お問い合わせ](https://fusic.co.jp/contact/)から気軽にご連絡いただけますが、[TwitterのDM](https://twitter.com/kwashizzz)からでも大歓迎です！


# 参考文献
- [Normalizing Flow入門 第1回 変分推論](https://tatsy.github.io/blog/posts/2020/2020-12-30-normalizing_flow%E5%85%A5%E9%96%80_%E7%AC%AC1%E5%9B%9E/)
- [Normalizing Flow入門 第2回 Planar Flow](https://tatsy.github.io/blog/posts/2021/2021-01-03-normalizing_flow%E5%85%A5%E9%96%80_%E7%AC%AC2%E5%9B%9E/)
- [Normalizing Flow入門 第7回 Neural ODEとFFJORD](https://tatsy.github.io/blog/posts/2021/2021-01-11-normalizing_flow%E5%85%A5%E9%96%80_%E7%AC%AC7%E5%9B%9E/)
