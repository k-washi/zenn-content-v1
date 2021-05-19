---
title: "機械学習 今月読んだ論文・記事まとめ 2021年5月" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

# [Learning to Resize Images for Computer Vision Tasks](https://arxiv.org/abs/2103.09950v1)

解説記事：[画像タスクの性能を向上させる新しいリサイザー！](https://ai-scholar.tech/articles/image-recognition/image_resize)

一般的に、学習、評価の際に、リサイズ処理を用いて画像を低解像度(224x224)にダウンサンプリングすることが多い。そこで、適応型画像リサイザーというリサイズモデルを提案し、性能を向上させた。このリサイズモデルは、特定のアーキテクチャに最適なバッチサイズと画像解像度を効率的に探索している。

# [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413v1)

解説記事: [Transformerが緻密な予測タスクで高精度を達成！](https://ai-scholar.tech/articles/others/dense_transformer)

CNNは、特徴マップのダウンサンプリングをしており、異なるスケールの特徴を抽出している。メモリを削減できるが、しかし、情報の損失につながる。そこで、ViT(Vision Transformer)をバックボーンとして使用するDPT（Dense Prediction Transformer)を提案した。特に、大量の学習データがある場合、高密度の予測タスク(例えば、Semantic Segmentationなど)の大幅な改善ができている。

# [Understanding Robustness of Transformers for Image Classification](https://arxiv.org/abs/2103.14586)

解説記事: [トランスフォーマーはコンピュータビジョンのためにあるのか？](https://ai-scholar.tech/articles/transformer/robust_transformers)

画像摂動を用いて、ViTの評価性能を研究している。ViTは、同サイズのResNetsと比較して、小さいデータセットではロバスト性が低かった。しかし、データセットのサイズが大きくなると、よりロバストになっていた。

レイヤー間の相関関係をみたとき、高い相関性を持っており、一部のブロックは不要と考えられている。しかし、Multi-head-attentionやフィードフォワード層などを取り除く実験にて、精度低下が見られ、相関性が重要であることを示した。

長距離Attentionへの依存性に関して、Attentionの距離を制限すると性能が劣化した。また、ネットワークの最後の層でパッチ間のAttentionを削除しても制度への影響は少なかったが、最初の層を削除すると悪影響を及ぼした。

# [Pervasive Label Errors in Test SetsDestabilize Machine Learning Benchmarks](https://arxiv.org/abs/2103.14749)

ComputerVision，自然言語，Audioの各データセットのうち，よく使用される10個のテストセットにおけるラベルエラーを特定し，これらのラベルエラーがベンチマーク結果に与える影響を調査している。10個のデータセットの平均エラー率は3.4%と推定され、例えばImageNet検証セットでは2916個のラベルエラーが6%を占めていた。実世界のデータセットのように誤り率が高いデータセットでは、容量の小さいモデルの方が実用的に有用であることが分かってきた。例えば、ImageNetにおいて誤ってラベル付けされたテスト例の割合が6％増加しただけで、ResNet-18はResNet-50を上回った。このことからも、訓練データが低品質であろうと、テストデータを高品質にすべきであることを示唆した。

# [Representing Numbers in NLP: a Survey and a Vision](https://arxiv.org/abs/2103.13136)

数字に関する最近のNLP研究の分類、まとめ。結論として、サブワードを用いた単語表現が、数字に対して最適ではないこと指摘していた。また、数字に関する問題を総合的に解決するために必要な特異性、網羅性、帰納バイアスにて、未解決の研究課題がある。

# [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)

解説記事: [2021年最強になるか！？最新の画像認識モデルEfficientNetV2を解説](https://qiita.com/omiita/items/1d96eae2b15e49235110)
EfficientNetにFused-MBConvとProgressive Learningを組み込むことで、SoTAモデルより6.8倍小さく、11倍も速い学習で、高い分類制度を示した。学習の遅さに対する、戦略として、最初に小さい画像サイズを用い徐々に画像サイズを上げていくProgressive Learningを適用した。Fused-MBConvは、速度が遅いDepthwise畳み込みを3x3の畳み込みに置き換えた。

# [量子コンピューティングは機械学習にどのような利益をもたらすか](https://ainow.ai/2021/04/28/253678/)

量子機械学習のアーキテクチャとして、データを量子的にするか、アルゴリズムを量子的にするか、両方とも量子的にするかで検討されている。
以下、機械学習を量子でどのように改善させるかの検討点で、詳細は記事！
1. 量子コンピュータで線形代数のサブルーチンを高速化できれば、機械学習を高速化できる。
2. 量子的並列性はモデルをより速く訓練するのに役立つ。
3. 量子コンピュータは、古典的なコンピュータにはできない方法で、高度に相関した分布をモデル化できる。


