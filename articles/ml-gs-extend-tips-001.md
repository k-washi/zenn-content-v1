---
title: "Gaussian Splatting 改善のアイデアをまとめてみた" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

こんにちは、最近、3D, 音声関連の機械学習にはまっている、[わっしー](https://twitter.com/kwashizzz)です。
[Fusic](https://fusic.co.jp/)の[アドベントカレンダー 16日目](https://qiita.com/advent-calendar/2024/fusic)の記事です！

昔ながらのMulti View Stereoに比べるとNeRFの凄さに数年前に驚いたのですが、計算コストの大きさから手を出すのを躊躇していました。そう思っていたら、Gaussian Splatting (GS)が登場し、計算コストを抑えつつ、NeRFに匹敵する精度を出すことができるということで、最近はGSに注目しています。

特に、GSの精度の良さから、mesh化を行いBlenderなどで使用できるようにしたいと考えています。
しかし、GSで学習されたガウシアンの点群をメッシュ化した際に、メッシュがぼやけていたり、エッジが不明確であったり、最悪の場合、穴が空いてしまうことがあります。

そこで、オリジナルのGSにどのような手法を加えることで、メッシュ化の精度を向上させることができるか考え、アイデア（論文）をまとめてみました。

主に、方針としては、以下の2つがあると思います。

1. 基盤モデルを活用した事前情報の取得

2. レンダリングされる画像、法線(Normal), 深度(Depth)の損失の改善

2. ガウシアンのcloneやsplitなどdensificationの改善

本記事では、上記の課題に関して、それぞれ解説したいと思います。

※ 本記事で使用している図は、各論文から引用いたしました。

# 基盤モデルを活用した事前情報の取得

### 深度推定モデルを利用したDepth Regularization

[3D GSの本家のリポジトリ](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#depth-regularization)を見ると、Depth Regularizationという手法が追加されていました。

Depth Regularizationは、GSの学習時に、Depthの事前情報を利用して、シーン再構成の精度を向上させる手法です。しかし、深度推定による深度マップは、一般的に、正規化されており、そのまま使用することが困難です。そこで、[DepthRegularized GS](https://arxiv.org/abs/2311.13398)という手法では、単眼の深度推定モデルによる深度マップを、GSの前処理で実行するcolmapで得られる疎な特徴点群で、スケール等を調整し、GSの学習時に利用しています。

深度推定に、[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) など、強力な深度推定の基盤モデルが利用されます。以下は、Depth Anything V2の結果の一例です。どの画像でもロバストに深度推定ができており、微細な部分の深度も補助できるため、GSの深度マップのレンダリングの性能向上が期待できます。

![depth_anything_v2](https://storage.googleapis.com/zenn-user-upload/ae1a4d92f727-20241211.png)

### 法線推定モデルの活用

GSのガウシアンから、メッシュを作成するためには、メッシュの表面の法線が必要です。そのため、[2D GS](https://arxiv.org/abs/2403.17888)のように法線も同時に学習することで、メッシュ化の精度を向上させることができます。

しかし、2D GSの法線推定は、画像の情報しか利用していないため、エッジ部分や平坦な部分での性能が低いことがあります。そこで、[StableNormal](https://arxiv.org/abs/2406.16864)という基盤モデルによる事前情報を利用することで、GSの法線推定の精度を向上させることができます。

下図は、StableNormalの結果の一例です。エッジ部分や平坦な部分でも、正確な法線を推定できていることが確認できます。

![Stable Normal Result](https://storage.googleapis.com/zenn-user-upload/9ecc4ec9cd1b-20241211.png)

また、論文では、2D GSと法線推定の性能を比較しており、下図のように、2D GS + StableNormalの方がエッジが鮮明でメッシュ化の精度が向上していることが確認されています。

![2D GS w SN Comp](https://storage.googleapis.com/zenn-user-upload/a07780e644ec-20241211.png)

# 損失の改善

### レンダリングされる画像の損失の改善

GSの学習に使用される画像では、隣り合った画像同士で、色見や明るさが異なることが多々あり、そのまま学習した場合、性能の劣化につながります。
[VastGaussian](https://arxiv.org/abs/2402.17427)という手法の論文に書かれている例では、下図のように、見え方が異なる画像を使用して学習した結果、GSによる3次元空間上に*Floaters*という、ぼやっとした部分が発生してしまっています。

![VastGaussian_Fig2](https://storage.googleapis.com/zenn-user-upload/bb4eeb91b1b3-20241212.png)

そこで、[VastGaussian](https://arxiv.org/abs/2402.17427)では、下図のようにAppeareance Modelingという、画像ごとの見た目のブレを考慮した損失のアーキテクチャーを提案しています。この手法は、レンダリングした画像を低解像度にダウンサンプリングし、訓練可能な埋め込み$l_i$をピクセル単位で連結し$D_i$を取得する。その後、$D_i$をCNNに通し、元の解像度になるようにアップサンプリングし、外観のずれを表現した画像$M_i$を取得します。そして、最後に、$M_i$とレンダリングした画像をかけ合わせて、画像ごとの外観のずれを考慮した画像を取得します。損失は、幾何的な構造を重視するSSIM損失をレンダリング画像と正解画像に適用し、見た目のずれを重視するL1損失を$M_i$による外観のずれを考慮した画像と正解画像に適用します。論文のAblaition Studyによると、この手法を適用することで、SSIMが0.858から0.882に向上し、Floatersが減少していることが確認されています。

![VastGaussian_Appearence model](https://storage.googleapis.com/zenn-user-upload/48d5c125430d-20241212.png)

### 法線/深度の損失の改善

3D GSでは、photometric lossというレンダリング画像と正解画像の誤差でのみ学習を行っていました。しかし、2D GSや[RaDe-GS](https://arxiv.org/abs/2406.01467)では、Normal lossや、DDepth distortion losを追加しており、メッシュ化した際の精度を向上させています。


Normal Lossは、以下の式で表されます。

$$
L_n = \sum_{i=1} \omega_i (1 - \mathbf{n}_i^T \mathbf{N})
$$

ここで、$\mathbf{n}_i$はカメラを向いたガウシアンの法線、$\mathbf{N}$は深度マップの勾配をによって推定された法線、$\omega_i$は、透明度から計算されるブレンド重みです。

また、Depth Lossは、近傍のDepthを用いて計算され、以下の式で表されます。

$$
L_d = \sum_{i=1} \omega_i \omega_j |d_i - d_j|
$$

ここで、$d_i$はピクセル$i$の深度、$\omega_i$は透明度から計算されるブレンド重みです。

2D GSにおいて、下図のように評価が行われています。
Normal lossとDepth distortion lossを追加することで、メッシュ化の精度を向上していることがわかります。
特に法線の学習により、メッシュ化時に、穴が空いたりする現象が減っています。
これは、法線が不正確の場合、推定法線が逆になり、穴が空いてしまうためと考えられます。

![2d gs result](https://storage.googleapis.com/zenn-user-upload/a0bf6d3293f6-20241212.png)

### 法線学習時のエッジの取り扱い

[AtomGS](https://arxiv.org/abs/2405.12369)では、法線の学習時に、エッジの取り扱いを考慮することで、周波数の高い微細な領域において、複雑な構造の法線を推定できるように、工夫しています。

下図の(a)は、深度マップの勾配から法線を推定した例です。加えて、(b)は、法線マップの勾配の大きさを示す曲率の例です。この曲率マップにおいて、値が高いほど法線がばらついており、値が低いほど法線が滑らかであり、この曲率マップを最適化することで、幾何的な平滑性を高めることができると考えられます。しかし、この結果、エッジや微細な構造のような高周波成分を不注意に平滑化し、過平滑化された結果が得られることがあります。

そこで、RGBの勾配から得られる図(c)のようなエッジマップを活用しています。曲率マップに対して、エッジマップの勾配が低い滑らかな部分は高い重みにし、より平滑化を行い、逆に、勾配が高い鋭いエッジ部分は低い重みを受け細部を保持する、重み$\omega(x)=(x - 1)^q$をNormal lossに適用しています。

以下、エッジ重みを適応したNormal lossの式です。

$$
L_n = \frac{1}{N} \sum_{i}^{H} \sum_{j}^{W} |\nabla N| \otimes \omega(|\nabla I|)
$$

この損失を適応した結果、下図(d)のように、エッジ部分がより考慮された法線が推定されていることがわかります。

![](https://storage.googleapis.com/zenn-user-upload/40ef42b52bd5-20241212.png)

# densificationの改善

GSの学習時に、ガウシアンが大きい場合は分割したり、足りない場合は、クローンしたりするdensificationの戦略が重要になります。例えば、ガウシアンが大きい場合、ぼやけた表現になってしまうover-reconstructionの状態になってしまうので、ガウシアンを分割し、複数のガウシアンでより細かい表現を行う必要があります。

[AbsGS](https://arxiv.org/abs/2404.10484)は、このover-reconstruction時の分割する基準に関して、改善点を提案しています。

下図は、(a)のような画像を、(b)のような1つの大きなガウシアンで表現した場合の例です。
この場合、もちろんのことですが、(b)のようにレンダリングした画像は、ぼやけた表現になってしまいます。
(c)のように、この結果と正解の誤差をL1損失で計算し、x軸方向に勾配をとった例が(d)です。

極端な例ですが、一般的なGSでは、この勾配をガウシアンの範囲で合計した値が任意の閾値を超えた場合、そのガウシアンを分割するようにdensificationの戦略をとっています。

![](https://storage.googleapis.com/zenn-user-upload/1f7fa3ae7a0b-20241212.png)