---
title: "CCC2021 まとめ" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "idea" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: False # 公開設定（falseにすると下書き）
---

# Video Recognition

3DCNNによる行動認識

Recent Advances in Video Action Recognition with 3D Convolutions 

基本的な問題設定

入力：動画 => 出力:行動ラベル

動画認識における主なConv
2D, 3D(時間を含めたとい意味で3DConv), 
(2+1)D Conv (3D Convの亜種で高コスパ, 3x3x3より3x3+3が良いみたいな感じ)

初期の代表的なモデル

### Two-stream CNN
2D CNN
RGB + Optical Flow

3D CNNの認識性能が低め　(データセットが公開)
大規模動画データセットであるKineticsの公開以降、性能向上

### C3D
3Dモデルの初期の代表モデル
時間方向のカーネルサイズ3が良い

### I3D

GoogLeNet( inception v1)をベースにした3D CNN

学習済み2D Convのパラメータを時間方向に重複コピー(inflation)する事前学習も提案

### P3D

(2 + 1)D Convと同様の Pseudo 3D Conv

R(2 + 1)Dも提案

S3D: Separable Conv

### SlowFast Network

低フレームと高フレームレートのtwo-streamネットワーク

### X3D

高コスパな九蔵を探索し、入力動画の解像度やフレーム数、フレームレートをあげるのが性能向上に大きく寄与

### 引用数が多い論文

- Learning Spatiotemporal Features with 3D Convolutional Netowrks
  定番手法の提案
- Wuo Vadis, Acition Recognition? A New model and the Kinetics Dataset 
  データセット付きで提案
- Non-local Neural Networks
  動画認識以外の文脈でも引用される　（汎用性のある手法)

### 今後流行りそう

動画認識 x Transformer
 - TimeSformer, STAM, ViViT, Video Swin Transformerなどなど

系列データである自然言語で有効なTransformerが同じく時系列データである動画像でも有効と期待
メモリをかなり食うので、きつい

## Dataset

Kinetics (デフォルト) 
Something-Something (人と物体のインテラクション動画)
Sports-1M
UCF-101

# Efficient Traning (効率的学習)

Fashion Cluster Database

### Efficient Training

1. 学習の効率化 (NAS Stochastic depth)
2. モデルの軽量化 (Pruning, Model compression)
3. 軽量なモデルの提案 (EfNet, MobileNet, BinaryConv...)


### モデルの歴史

SENet: Squeeze-extend
mobile net: depthwise separable conv. (ch,widhを別々に)
mobile net v2: inverted residual block

Mobail NAS (MNAS): 階層的なarch searchが可能になり、探索の軽量化

### Efficientなモデルの歴史

1. EfficientNet
2. EfficientNAS (グラフ理論によりNASの探索空間を効率化)
3. EfficientDET (SOTAなDet)
4. EfficientNet v2 (depthwise convの廃止とスケーリングを工夫)
5. RepVGG (reparameterizationによるVGGで効率な推論を可能に)

ViTの登場~
1. DeiT
2. Efficient DETR


### MobileNets

畳み込み層をdepthwise(ch方向)とpointwise(空間方向)に分割

width multiplier (入出力のチャネル数に従おさんすることで各層のwidthを調整)
Resolution multiplier

### MobileNets v2

linerar bottlenexksによって非線形性の原因である特徴をフィルタリングし invert...

### EfficientNet

depth, width, resolutionを同時にスケールアップ

Mobile Inverse bottlenetにSEを導入

### EfficientDet

解像度が異なる特徴に対して、Weighted bi-directional feature pyramid net (BiFPN)を導入

SOTAと比較し、52.2APと最高で4~9倍パラメータ効率が良く１０倍以上高速

### EfficientNet v2

Effnetの課題
画像サイズが大きい場合、学習の速度が低下
Depthwise Convをアサイそうに用いると学習の速度が低下
depth, width, resolutionを均等にスケールアップする方法は不適

Fused-MBConvを提案

### Efficient DETR

DETRシリーズにおける学習速度の低下の原因
6つのデコーダをもち object queriesの反復的な更新
object container(queires + ref point)をランダムに初期化

特徴マップを利用して、object containerを初期化することでデコーダそうを1つに削減
queriesをエンコーダの特徴を用いて初期化することで、1つに削減した性能を改善

### Efficient Neural Architecture Search via Parameter Sharing

従来のNAS 450GPU, 3,4日 => 1GPU, 16時間未満

NASにDAGを適用することで、探索空間ないのchild modelsの間でparameを共有可能
グラフ理論を使用


### Meta Servay

Efficient Traningの研究はどのように進めるべきか

# Cross modality meta-survey of dataset

データセットの構築に注目
作成コストがかかるので、なるべく効率的に効果の高いデータを作成したい

疑問: どう作るのが良いか

### データセットの事例に学ぶ

データ元

Paper with code (有効なデータセットを探すのに有用なサイト)
Hugging face (自然言語処理のモデルとデータセットが集う)

### 画像classification

Tencent ML-Images (11166cls, 17,698,491枚)
ImageNet (21841, ...)
OpenImages (19957cls, 59,919,574 Google)

### 物体検出

MSCOCO (80cls, 328000枚, 640, 480)
Open Images (6000+, 9000000枚)

### Instance Segmentation

OpenImages V6 (350cls, 2785498枚)

### Semantic segmentation

ADE20K
Cityscapes
SUN dataset
SIFT-flow dataset


### 3Dモデル

ABC (CAD Models) (1,000,000)
ScanObjectNN
Objectron

### 3Dシーン

KITTI (交通シーン)
Matterport3D (部屋レベル)
ScanNet (家レベル)

### 医療画像

LUNA2016
BIMCV COVID-19+ コロナ患者の胸部X線&CT画像　(2クラス, ...)
PADChest(胸部X線画像, 174箇所19病名104部位, 160,000+)

### コーパス 

Wiki-40B
WMT (wmt14は11言語)

## 考察

規模
- データ数、ラベル数、対応タスク数
- 対象範囲 (一般or特定)

ソース
- 権利問題 (CC-BYのデータのURL, 手前で撮影等の肖像権)
  
アノテーション
- 精度バランスの設計 (クラウドソーシング, or 自力)
- アノテーション方法の工夫

データセットを作る時

カテゴリ内分散を大きくする必要がある
教師ラベルの注意点 (隣り合った、テクスチャの近いデータに対するアノテーションはやめたほうが良い)
しかし誤ったものが入っていたものとしても平滑化項として働くので、数%誤っていても量を増やしたほうが良いのでは...



# Transformer メタサーベイ

Vision and Languageのサーベイ

## Transformerが良い理由
NLP界隈の悩み　時系列をRNNで扱うのは時間がかかる　（逐次的に更新される隠れ層が入力なので遅い並列化して一度に処理できない）

Self-attentionで時系列データを一度に学習

RNNより処理が早い->大規模学習向き
Self-attentionによる表現力が強い (indactive biadがないほうが良い)

自分のタスクでも使いたいが...

## Transformer block

MHAは役に立つ?
SAの内積はベクトルの各要素にわたる待機的な類似度
小さなベクトルに切り分け計算 (トークン間の多様な類似性を発見)

同じ性質を学習しがち

LayerNorm(入力系列をトークンごとに正規化)はBatchNormじゃだめ？
Q 画像は系列長が固定だけど、BatchNormじゃだめなの？ => batchnormだと統計量が変わる

2層のNLPは役に立つ？ tfの2/3のそうを占める mix


POSTNormはオワコン?
PreNormの方が良いという報告
しかし、慎重に最適化すればPostNormの方が良い => residual blockにf(x)を追加

Positional encordingに工夫するところはあるか? Sin,Cosが良いのか=>決着がついていない

絶対位置か相対いち? 分類では絶対位置、スパン予測では相対位置が強い
長い文だと相対位置が良い

位置Embをattentionに足す　(Modified Attention Matrix; MAM)

## 計算量がきつい

文脈情報をメモリに記憶 (Transformer-XL)
固定マスキング (strided attentoin, fixed attention)
動的マスク(Reformer, keyとqueryの高いところが欲しいというアイデア)
圧縮 (Linformer, SAをSVD 情報の欠損が少ない)

## 画像で扱うのに良い方法は?

パッチで区切る => 高速であるが、大量の画像があれば高精度　(ViT, ViLT)
CNNの特徴量をトークンに (DETR, UNITER系,VinVL)

特徴量を離散化 (VAEなど) => DALE-Eなど

## CVタスクにおける事前学習済みモデル
まだまだない

Transformerじゃなくても?

Single Headed Atention + RNNでも同等の精度

MLP: 転置してMLPに入力 (MLP-Mixerなど)

# 将来的に...

Transformerはindactive biasがないので、大規模データセットが必要
CNNなどと組み合わせると良いのでは...

# 画像生成・生成モデルのメタサーベイ

NeRF (ナーフ)- 色と密度(Nerural Radiance Field)を推定しVolume Rendering

## NeRFの変換

計算効率化・高速化
KiloNeRF (小さなMLPを多数準備し、NeRFを表現することでNeRFレンダリングの高速化)

動的シーンへの適応
Neural Scene Flow Field (NSFF); tech blogに書いたやつw
未知観測領域をどうするかが今後の課題

カメラパラメータの制限
iNeRF (analysis-by-synthesisの枠組みから6DoFポーズ推定)
NeRF-- NeRFと同時にポーズ最適化

NeRFの汎化性
新規シーンのNeRFを少量データから学習したい
Learnit 
DietNeRF: CLIP Encoderから得られた特徴を使い異なる視点間のSemantic consistency lossを取ることで、少量の視点から学習可能

陰関数表現と微分可能レンダリングの土台があってこそ

# Object-oriented Representation Learning (物体指向の表現学習)

物体指向の?
物体ごとの個別の表現を獲得 (物体の分離、物体ごとの操作)

下流タスクに有効なオブジェクトごとの個別の表現を獲得すること
OORLは教師なしで物体検出の手法があるため、保持タスクにおけるアドバンテージ獲得の可能性

AIR (オブジェクト度とに潜在変数を用意)からの派生がたくさんある

基本的に前景と背景の分離？
Scene Mixture

# Cycle Consistency

Breaking the cycle -- Colleague is all you need

Attention-GAN for Object Transifiguration in wild images
Attention mapを利用

Selfi2Animeの画像













