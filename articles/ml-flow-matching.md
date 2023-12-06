---
title: "Flow Matching - 拡散モデルにかわる生成技術" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習", "論文解説", "Diffusion", "FlowMatching"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
publication_name: "fusic"
---

こんにちは！[Fusic](https://fusic.co.jp/) 機械学習チームの鷲崎です。最近、音声や言語処理に興味がありますが、機械学習モデルの開発から運用までなんでもしています。もし、機械学習で困っていることがあれば、気軽に[DM](https://twitter.com/kwashizzz)ください。

本記事では、Flow Matching (FM)と、その発展版であるOptimal Transport Confitional Flow Matching (OT-CFM)を解説します。最近の生成AIでは、拡散モデルがよく使用されていますが、Flow Matchingは、拡散モデルに取って代わる可能性がある生成技術と考えています。

実際、音声合成界隈では、Metaが発表した音声合成手法である[Voicebox](https://voicebox.metademolab.com/)や、より高速で高性能な音声合成手法である[Matcha-TTS](https://arxiv.org/abs/2309.03199)に使用されてきています。
Matcha-TTSに関しては、弊社のToshikiが、記事を書いているので、ぜひ、読んでみてください。

[【音声合成】Matcha-TTS🍵で日本語音声を生成してみる](https://zenn.dev/fusic/articles/bd7da12adf5901)

個人的に、Flow Matchingは、拡散モデルが発展した手法と考えて論文を読み始めたのですが、それだとかなり勘違いしてしまっていました。
数年前に拡散モデルとは別に、Normalize Flowという手法が流行った気がします。その、Normalize Flowの発展版と考えるとスムーズに納得できました。
そこで、本記事では、参考になる記事などを示しつつ、拡散モデルとNormalize Flowの関係性からFlow Matchingまでの過程を解説していきたいと思います。

理解が及ばなかった部分がたたあるため、ご指摘をたくさんいただけると嬉しいです。よろしくお願いいたします。

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

# Flow Matching

CNFを安定して学習可能なFlow Matchingについて解説する。

# Conditional Flow Matching

Flow matchingは、ガウス確率経路を仮定しているため、2つの分布間の条件付き確率経路を定式化し、ガウス確率経路である仮定を緩和するよう修正したConditional Flow Mataching (CFM)について解説する。

# Independet CFM

# Optimal Transport CFM

# Schrödinge Bridge CFM



# 参考文献
- [Normalizing Flow入門 第1回 変分推論](https://tatsy.github.io/blog/posts/2020/2020-12-30-normalizing_flow%E5%85%A5%E9%96%80_%E7%AC%AC1%E5%9B%9E/)
- [Normalizing Flow入門 第2回 Planar Flow](https://tatsy.github.io/blog/posts/2021/2021-01-03-normalizing_flow%E5%85%A5%E9%96%80_%E7%AC%AC2%E5%9B%9E/)
- [Normalizing Flow入門 第7回 Neural ODEとFFJORD](https://tatsy.github.io/blog/posts/2021/2021-01-11-normalizing_flow%E5%85%A5%E9%96%80_%E7%AC%AC7%E5%9B%9E/)
- [Morpho Tech log - A Brief Survey of Schrödinger Bridge (Part I)](https://techblog.morphoinc.com/entry/2023/09/12/100000)