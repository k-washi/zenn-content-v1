---
title: "【CVPR2022】画像異常検知 PatchCoreの実装解説" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["python", "ml"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

# はじめに

こんにちは、[わっしー](https://twitter.com/kwashizzz)です。本記事では、CVPR2022で発表された画像異常検知手法である[PatchCore](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf)の実装について解説します。

まずは、実際に試した結果です。下図の上は正常画像、下は異常画像です。異常部分が赤くなっており、製品が欠損していることがわかります。

![](https://storage.googleapis.com/zenn-user-upload/08dabbd0fbe0-20220827.png)

PatchCoreの詳細については、[外観検査向け異常検知手法に関する論文紹介](https://techblog.leapmind.io/blog/20220701-introduction_to_patchcore/)の記事がわかりやすいです。

PatchCoreの利点は、**ImageNetなどのデータセットで学習された事前学習モデルの特徴マップを用いるため深層学習モデルの訓練の必要ない**ことです。
手法としては、

1. 正常な画像群の特徴マップにおける局所的な部分をパッチ特徴量としメモリバンクに保存する
2. 高速化のためランダム射影で次元削除した特徴量に対してGreedy法を用い、メモリバンク内のパッチ特徴量の数を削減
3. テスト画像の各位置の特徴量に対して、近傍法でメモリバンク内の最も近い特徴量を選択し、その距離を異常度のスコアとする

です。難しそうに思えますが、以下の実装を見ながらだと、理解できると思います。

# 実装

次に実装の詳細を解説します。[k-washi/anomaly_detection_exp_v1](https://github.com/k-washi/anomaly_detection_exp_v1/blob/main/src/model/patchcore.py)に、実装を載せています(簡単のため解説では、修正している部分があります)。基本的にpytorch lightningに則った実装をしています。

PatchCoreは、Resnetなどのモデルの中間層を使用するため、`hook`を用いて取得できるようにします。

```python
class PatchCoreModelModule(LightningModule):
    def __init__(self, cfg: Config) -> None:
        super(PatchCoreModelModule, self).__init__()

        ...

        self.model = torch.hub.load(
            'pytorch/vision:v0.9.0',
            'wide_resnet50_2',
            pretrained=True
        )
        
        # wide resnet50のlayer2とlayer3の出力をForward Hookを使用して取得
        self.features = []
        def hook_t(module, input, output):
            self.features.append(output)
        
        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)

    def forward(self, x_t):
        # 特徴量を出力
        # length:2 
        # y[0]:torch.Size([b, 512, 32, 32]) 
        # y[1]torch.Size([b, 1024, 16, 16])
        
        self.features = []
        _ = self.model(x_t)
        return self.features
```

まずは、全正常画像に対する特徴量の取得です。`training_step`でこの処理を行います。

```python

class PatchCoreModelModule(LightningModule):
    
    ...

    def training_step(self, batch, batch_idx): # save locally aware patch features
            x, _, _, _ = batch
            features = self(x)
            embeddings = []
            for feature in features:
                # Average Poolingを行うことで、特徴マップを局所的にぼやかすのと同等の処理をしている。
                # 同じ大きさの特徴マップを返す ex. torch.Size([b, 512, 32, 32]) => torch.Size([b, 512, 32, 32])
                m = torch.nn.AvgPool2d(3, 1, 1)
                embeddings.append(m(feature.cpu()))
            
            # 2つの特徴マップを結合
            embedding = embedding_concat(embeddings[0], embeddings[1])
            
            # 位置ごとの特徴量を格納(1つの特徴量はチャンネル方向の特徴)
            self.embedding_list.extend(reshape_embedding(np.array(embedding)))

# 上記のメソッドで使用している、関数です
def embedding_concat(x, y):
    """
    yをxに合わせてアップサンプリング
    Args:
        x (_type_): torch.Size([b, 512, 32, 32])
        y (_type_): torch.Size([b, 1024, 16, 16])

    Returns:
        torch.Size([b, 1536, 32, 32])
    """
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2) # 2
    y = F.interpolate(y, scale_factor=s, mode="bilinear")
    z = torch.cat([x, y], dim=1)
    return z
    

def reshape_embedding(embedding):
    """batchx32(x位置座標)x32(y位置座標) 個のチャンネル方向特徴量のリストを作成

    Args:
        embedding (_type_): torch.Size([b, 1536, 32, 32])

    Returns:
        List[torch.Size([1536]), ....]
    """
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list


```

全ての正常画像に対する特徴量を取得した後は、特徴量の次元削除とGreedy法による特徴選択です。

```python
from sklearn.random_projection import SparseRandomProjection

class PatchCoreModelModule(LightningModule):
    
    ...
    def training_epoch_end(self, outputs): 
        total_embeddings = np.array(self.embedding_list) # List[torch.Tensor((1536,)), ...] 各位置の特徴のリスト
        # Random projection
        # 高速化のため、使用する次元をランダム射影で選び次元削減する
        # Johnson-Lindenstrauss lemmaに則って低次元に射影するランダムな行列を計算している
        # Johnson-Lindenstrauss lemma: 高次元のユークリッド空間内の要素をそれぞれの要素間の距離をある程度保ったまま、別の(低次元の)ユークリッド空間へ線型写像で移せるという補題
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings) # 射影する行列を学習

        # Greedy法により、特徴量の数をN個選択する
        print(total_embeddings.shape)
        selector = kCenterGreedy(total_embeddings,0,0)
        selected_idx = selector.select_batch(model=self.randomprojector, N=int(total_embeddings.shape[0]*0.001))
        self.embedding_coreset = total_embeddings[selected_idx]
        
        print('initial embedding size : ', total_embeddings.shape) #  (245760, 1536)
        print('final embedding size : ', self.embedding_coreset.shape) # (245, 1536) ここまで特徴量の数を減らせる

        # 特徴量の保存
        with open(self.embedding_dir / 'embedding.pickle', 'wb') as f:
            pickle.dump(self.embedding_coreset, f)


```

Greedy法を実装した、`kCenterGreedy`クラスは以下に実装を記載しています。

```python
import numpy as np
from sklearn.metrics import pairwise_distances

from src.model.utils.sampling_methods.sampling_der import SamplingMethod

class kCenterGreedy(SamplingMethod):

    def __init__(self, X, y, seed, metric='euclidean'):
        self.X = X
        self.y = y
        self.flat_X = self.flatten_X()
        self.name = 'kcenter'
        self.features = self.flat_X
        self.metric = metric
        self.min_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []

    def select_batch(self, model, N):
        """
        Greedy法による特徴量選択
        """

        # 特徴量の次元削除, (feat_num, 1536) => (feat_num, 306)
        self.features = model.transform(self.X)
        
        # N個の特徴量を選択
        new_batch = []
        for _ in range(N):
            if self.already_selected is None:
                #初期化: feat_num(Xの特徴数)の中からランダムに一つindexを選択
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                # 現在選択されている特徴量との距離が最大の点(特徴)を選択
                ind = np.argmax(self.min_distances)

            x = self.features[[ind]] # 特徴量取得
            dist = pairwise_distances(self.features, x, metric=self.metric) #kcenter指標でindで選択された特徴料xとの距離を計算
            
            # 各特徴量に対する最も小さな距離を残す
            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1,1) # (feat_num, 1)
            else:
                # 以前計算された距離と現時点で計算した距離の小さい方法残す
                self.min_distances = np.minimum(self.min_distances, dist) #  (feat_num, 1)
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'% max(self.min_distances))

        return new_batch
```

最後に、新しい画像に対する異常検知です。正常画像による選択された特徴量に対する異常値を計算します。

```python

from sklearn.neighbors import NearestNeighbors

class PatchCoreModelModule(LightningModule):
    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        # ここではbatch_size=1を想定
        x, gt, label, x_type = batch

        # 正常画像の特徴量を読み込みます
        self.embedding_coreset = pickle.load(open(self.embedding_dir / 'embedding.pickle', 'rb'))
        
        # テスト画像の特徴量を計算します (train_stepと同じです)
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding_test = np.array(reshape_embedding(np.array(embedding_.cpu())))


        # k近傍法
        # 最も近い特徴量をn_neighbores個探索します
        nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree', metric='minkowski', p=2).fit(self.embedding_coreset)
        score_patches, _ = nbrs.kneighbors(embedding_test) # 正解特徴量との距離 (1024, 9) : (特徴マップ32x32, 近傍特徴量9個)
        anomaly_map = score_patches[:,0].reshape((32, 32)) # 最も近傍な特徴量との距離を1列から特徴マップの形式にreshape
        
        anomaly_map_resized = cv2.resize(anomaly_map, (254, 254)) # 元の画像サイズにresize
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4) # 結果がシャープすぎるので少しぼかす
```


# 最後に

以上、簡単に処理の流れを解説しました。かなり精度良く異常部分を検出してくれていますが、一方で、製品ごとに、異常値のスコアが異なっていたりと閾値の設定が必要になります。

まだまだ、これのみに外観検査を任せられる精度ではない気がします。閾値の設定を低くめに設定し、人が目視する量を減らすために使うなどコスト削減などには使えるかもしません。

---

画像・自然言語・音声に関する機械学習の研究開発やMLOpsを行っています。もし、機械学習に関して、ご相談があれば、[@kwashizzz](https://twitter.com/kwashizzz)のアカウントまでDMしてください！
これまでの、[機械学習記事のまとめ](https://zenn.dev/kwashizzz/articles/my-ml-articles-summary)です。