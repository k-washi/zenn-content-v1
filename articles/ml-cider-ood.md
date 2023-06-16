---
title: "CIDER: 分布外(OOD)に適した埋め込みとは？" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習", "論文解説", "OOD"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
publication_name: "fusic"
---

こんにちは！[Fusic](https://fusic.co.jp/) 機械学習チームの鷲崎です。機械学習モデルの開発から運用までなんでもしています。もし、機械学習で困っていることがあれば、気軽に[DM](https://twitter.com/kwashizzz)ください。

本記事では、[CIDER: Exploiting Hyperspherical Embeddings for Out-of-Distribution Detection](https://ai.papers.bar/paper/fde80d32a02311ecbb8c3d25c114d5ed) という、Out-of-Distribution (OOD)の検出というタスクで、hyperspherical space(超球形空間)への埋め込みを利用する新しい表現学習フレームワークに関して解説します。

OODというタスクは、例えば、画像分類において、訓練に使用していないクラスが推論に入力された際に、分布外の入力として検知するというイメージです。基本的には、距離ベースの手法の性能がよく、モデルより得られる特徴埋め込みを利用し、IDデータ（学習時に使用したクラスのデータ）のクラスタから、OODのサンプルが比較的離れていることを仮定してOODが行われます。この論文で論じている距離ベースの手法のみでは、どのくらいの閾値を設定すればよいのかなど、距離の正規化が困難であったりするため、[VOS: Learning What You Don't Know by Virtual Outlier Synthesis](https://openreview.net/forum?id=TW7d65uYu5M)のように、IDデータの各クラスの特徴埋め込みを正規分布と仮定し、OODの閾値設定を容易にする方法などもあります。

[VOS: Learning What You Don't Know by Virtual Outlier Synthesis の解説・実装](https://zenn.dev/kwashizzz/articles/ml-vos-ood-det) の記事で解説していますが、この方法は、論文の本筋とは関係ありません...

この論文で述べられている課題は、既存の手法(Contrastive Lossを用いた手法など)は、IDデータの分類には適しているが、OODには最適ではない埋め込みを学習していることが挙げられています。

> 例えば、SupCon損失を用いてCIFAR-10を学習した場合、IDとOODデータの平均角度距離　は、埋め込み空間において29.86度しかなく、効果的なIDとOODの分離を行うには小さすぎる

とのことです。

そこで、**OODに最適な表現学習方法はなにか？** ということ疑問に対し、**C**ompactness (コンパクト性)と、**D**isp**E**rsoin **R**egularized 学習のOOD用に設計されてフレームワークであるCIDEが提案されていました。CIDERでは、超球形埋め込みを、単位ノルムの球状ガウス分布に似ているvon Mises-Fisher(vMF)分布に近づくなるように損失関数を設計しています。

この設計により、

1. コンパクト損失: サンプルが不正クラスと比較して正しいクラスに割り当てられる確率が高い

```python: https://github.com/deeplearning-wisc/cider/blob/34501dfcaa65820ed2e7021dd2678b2aba90ed72/utils/losses.py#L153
class CompLoss(nn.Module):
    '''
    Compactness Loss with class-conditional prototypes
    '''
    def __init__(self, args, temperature=0.07, base_temperature=0.07):
        super(CompLoss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, prototypes, labels):
        # クラスプロトタイプ(クラス埋め込み平均)
        prototypes = F.normalize(prototypes, dim=1) 
        proxy_labels = torch.arange(0, self.args.n_cls).cuda()
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, proxy_labels.T).float().cuda() #bz, cls

        # compute logits　(z^T * u / t)
        feat_dot_prototype = torch.div(
            torch.matmul(features, prototypes.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
        logits = feat_dot_prototype - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss
```


2. Dispersion loss (分散損失): 異なるクラス平均間の角度距離を最大化する

```python: https://github.com/deeplearning-wisc/cider/blob/34501dfcaa65820ed2e7021dd2678b2aba90ed72/utils/losses.py#L244
class DisLoss(nn.Module):
    '''
    Dispersion Loss with EMA prototypes
    '''
    def __init__(self, args, model, loader, temperature= 0.1, base_temperature=0.1):
        super(DisLoss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.register_buffer("prototypes", torch.zeros(self.args.n_cls,self.args.feat_dim))
        self.model = model
        self.loader = loader
        self.init_class_prototypes()

    def forward(self, features, labels):    

        prototypes = self.prototypes
        num_cls = self.args.n_cls

        # 指数移動平均(EMA) でクラスごとの埋め込み平均を更新 (各クラスの全クラスサンプルの平均ベクトルによる更新だと計算量が多すぎるため)
        for j in range(len(features)):
            prototypes[labels[j].item()] = F.normalize(prototypes[labels[j].item()] *self.args.proto_m + features[j]*(1-self.args.proto_m), dim=0)
        self.prototypes = prototypes.detach()
        labels = torch.arange(0, num_cls).cuda()
        labels = labels.contiguous().view(-1, 1)

        mask = (1- torch.eq(labels, labels.T).float()).cuda()


        logits = torch.div(
            torch.matmul(prototypes, prototypes.T),
            self.temperature)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(num_cls).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask
        mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
        mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
        loss = self.temperature / self.base_temperature * mean_prob_neg.mean()
        return loss

    def init_class_prototypes(self):
        """Initialize class prototypes"""
        self.model.eval()
        start = time.time()
        prototype_counts = [0]*self.args.n_cls
        with torch.no_grad():
            prototypes = torch.zeros(self.args.n_cls,self.args.feat_dim).cuda()
            for i, (input, target) in enumerate(self.loader):
                input, target = input.cuda(), target.cuda()
                features = self.model(input)
                for j, feature in enumerate(features):
                    prototypes[target[j].item()] += feature
                    prototype_counts[target[j].item()] += 1
            for cls in range(self.args.n_cls):
                prototypes[cls] /=  prototype_counts[cls] 
            # measure elapsed time
            duration = time.time() - start
            print(f'Time to initialize prototypes: {duration:.3f}')
            prototypes = F.normalize(prototypes, dim=1)
            self.prototypes = prototypes
```

の損失が提案されています。コードは、[How to Exploit Hyperspherical Embeddings for Out-of-Distribution Detection?](https://github.com/deeplearning-wisc/cider/tree/master)

このクラス間の距離を大きくするように学習するのがSupConなどの既存手法に対する新規部分?だと思います。vMF分布とどう関係するかというと、正直あまりわかりませんでした。損失関数を見てわかるように、一応、角度として扱われています。

結果としては、これらの損失を追加すると、OODの性能があがり、加えて、分類性能も上がっています。詳細は、論文を見てみてください。

# まとめ

コンパクトネス性 + クラス間距離の最大化でOOD性能を上げた論文でした。
実装が簡単なので、クラス間距離の損失を追加するのは、ぜんぜんありだと思いました。

