---
title: "オンライン複数物体追跡 SiamMOT: Siamese Multi Object Trackingの解説" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---
こんにちは、鷲崎です。弊社では、スポーツxAIという分野に取り組んでおり、画像分類や物体検出など多くの画像処理タスクを活用しています。今回は、選手の追跡タスクのため複数物体追跡(MOT; Multi Object Tracking)の最新手法に関して調査しました。

本記事で解説する[SiamMOT(Siamese Multi Object Tracking)](https://www.amazon.science/publications/siammot-siamese-multi-object-tracking)は、2021年5月末に発表された最新でstate-of-the-artなリアルタイムMOTの手法です。実際にサッカーの試合動画に試し、下図のようにかなりの精度で追跡できていることがわかります。

![サッカーMOT1](https://lh3.googleusercontent.com/BIr_cx6zoBeTg76kZBPzvMYPbBDQeY3FEl7uVHP5QyqZIH43bx6_QWXLMpxKqblKUfkAOZDJoWFve_3oTlqa2zcaqb9u47-bgsb2V3bUwyZjyDk3RghPcZ6m5JP3w9jrtL4IAfW1Xg=s0 "")
![サッカーMOT2](https://lh3.googleusercontent.com/r5Xw5fUNqEnQBIhSd3BsmDZcFzDKpU8iHNThPS64buAPkOSbw1mQLm_OOGhHnmiPgoJIGbD4VrApleV5e7pMwYoEq0UB4aCoYbJhWL2egHA1ELs3TkLQo3zGnThDPug2EaPW5HEMmQ=s0 "")

# 複数物体追跡 MOT とは?

MOTとは、上図で示したように、物体のインスタンス(上図の人を囲んだ四角いボックスなど)を検出し、それらを時間的に関連づけて、インタンスの追跡を行うタスクのことです。MOTタスクに関する詳細は、[【物体追跡】Multi Object trackingの解説【MOT】](https://qiita.com/hampen2929/items/e30442283060afc26435)の解説記事が参考になります。

Siam-MOTは、[SORT(Simple and On-line Realtime Tracking)](https://arxiv.org/abs/1602.00763)という手法をベースにして、動きのモデル化を工夫しています。

SORTでは、カルマンフィルターを用いて、単純な幾何学的な特徴(位置, ボックスの形状など)でインスタンスの動きをモデル化しています。最近の追跡手法では、深いネットワークを学習することで、視覚的な特徴と幾何学的な特徴の両方に基づいてインスタンスの動きをモデル化しており、単純なSORTよりかなり精度が良いです。

視覚的な特徴も使用したMOT手法として、[DeepSORT](https://arxiv.org/abs/1703.07402)という手法が、有名ですが、Faster RCNNなどで物体検出し、その結果と前フレームの検出結果を追跡用のネットワークに入力し追跡していたため、速度が遅いという欠点がありました。そこで、[Center-Track](https://arxiv.org/abs/2004.01177)では、前フレームの画像・物体情報(物体の中心であろう点)と、現在の画像のみをネットワークの入力とし、追跡を可能にしています。

[Center-Track](https://arxiv.org/abs/2004.01177)は、Tracking data augumentationなど追跡データセットの拡張方法なども記載されており、面白い論文でした！

また、SORT, Siam-MOTは、オンラインMOTです。オンラインとは、将来の画像情報が得られず、現在の画像情報から過去のインスタンスとの関連付けを行うことです。


# SiamMOT

SiamMOTは、領域提案ネットワーク(Region Proposal Network; RPN)と領域ベースの検出ネットワークで構成されるFasterRCNNという物体検出器をベースにしています。RPNが、物体が写っているボックスを提案し中身が物体か、背景なのか判断するネットワークで、検出ネットワークがボックスの中に何が写っているか判断するネットワークです。

SiamMOTは、このFaster-RCNNに領域ベースのSiamise追跡器を導入することで、インスタンスレベルの動きのモデル化を達成しています。

![SiamMot Overview](https://storage.googleapis.com/zenn-user-upload/481ca6ea80cfd965a888f742.png)

このアーキテクチャは、上図(論文中Fig.1)のような構成になっています。

タスクとては、画像$I^{t}$で追跡中の物体を次の時刻の画像$I^{t+\delta}$でも追跡することです。なので、ネットワークへの入力は、画像$I^{t},I^{t+\delta}$と時刻$t$で追跡中のインスタンスの領域$R^t = \left\{R^t_1, \cdots, R^t_i, \cdots \right\}$となります。

SiamMOTでは、検出ネットワーク(上図のRPN features, Detect部分)により、検出されたインスタンスの集合$R^{t+\delta}$を出力します。一方で、追跡器(Siamese Trackerの部分)は、インスタンス情報$R^t$を時刻$t+\delta$に伝搬させ、インスタンス情報$\tilde{R}^{t+\delta}$を生成します。

Siamese Trackerに関して、論文中では、暗黙的な動作モデル(IMM)と明示的な動作モデル(EMM)の2つが提案されています。共通点は、できるだけインスタンス$i$の領域に対して、信頼度を大きくし、一方でその他の物体に関しては、信頼度を小さくするため、[SiamFC](https://arxiv.org/pdf/1606.09549.pdf)などのSiamise学習に基づいた追跡器の公式に則ったものになっていることです。一方で、異なる点として、暗黙的な動作モデルは多層パーセプトロンを使用しており、明示的な動作モデルは、下図(論文中Fig.2)のようなアーキテクチャになっています。


明示的な動作モデルの重要なことは、時刻$t$における物体領域$R^t_i$に対応する特徴マップ$f^t_{R_i}$を用いて、時刻$t+\delta$における特徴マップ$f^{t+\delta}_{S_i}$から、同じ物体であろう確率を計算しています。この時、特徴マップ$f^{t+\delta}_{S_i}$は、時刻$t$の物体領域$R^t_i$を$r (>1)$倍したものから計算できます。また、$v^{t+\delta}_i$は、インスタンス$i$に対する視認性の信頼度になります。

基本的に、明示的な動作モデルの方が性能が良いとのことでした。

Siamise学習に関しては、弊社Tech blogの[【論文読み】Exploring Simple Siamese Representation Learning](https://tech.fusic.co.jp/posts/2020-12-25-ml-simsiam-representation-learning/)を参考にしてください。

![SiamTracker](https://storage.googleapis.com/zenn-user-upload/a93aa665a60e2e9025a312f9.png)

その後、検出ネットワークと追跡器の出力は、SORT同様に、空間マッチングにより関連付けされ、時刻$t$と$t+\delta$の間のインスタン間を紐づけることができます。

# 実験の結果

論文における実験結果(Table.1)は、下のテーブルのようになっています。指標の横の矢印は、指標が高い方が良いか、低い方が良いか示しています。1秒あたりの処理数を示すFPSは、提案手法が小さく速度が遅いという結果が出ています。しかし、リアルタイムで使用できる範囲かと思います。

よく複数物体追跡で使用される評価指標であるMOTA(誤検出(FP)と見逃し(FN)とIDの変更の観点からの精度指標)は、改善しています。誤検出(FP)が増えているのは、気になります。ID swは、IDの割り当て失敗回数です。 

![実験の結果](https://storage.googleapis.com/zenn-user-upload/6132c8b412c7bc4539c771aa.png)

# まとめ

本記事では、最新の物体追跡手法である、Siam-MOTを解説しました。リアルタイムで動き、他の手法と比較してMOTA指標の精度が良いです。物体追跡は、画像と画像をつなぐ重要な情報で、スポーツなど動的な物体が多い対象ほど、複雑になるが重要でもあるタスクになります。スポーツxAIのタスクは、他にもあるので、シーン検出などを今後はやっていきたいと思います。