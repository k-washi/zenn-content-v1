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

本記事では、文章生成のモデルの生成テキストの不自然さや、望ましくない単語の繰り返しを抑える手法として、[A Contrastive Framework for Neural Text Generation](https://arxiv.org/abs/2202.06417)で提案されたSimCTG(a **SIM**ple **C**ontrastive framework for neural **T**ext **G**eneration)を紹介します。

また、[論文の実装コード](https://github.com/yxuansu/SimCTG/tree/bb54480e5c43d62d5b660d5cdabaee7c7d7af442)を参考にし、Encoder-Decoder形式の文章生成モデルであるT5にSimCTGを適用してみたので、実装方法を紹介します。

※ [論文の実装コード](https://github.com/yxuansu/SimCTG/tree/bb54480e5c43d62d5b660d5cdabaee7c7d7af442)には、GPT-2に適用した実装があるので、GPT-2を使いたい人は、[こちら](https://github.com/yxuansu/SimCTG/tree/bb54480e5c43d62d5b660d5cdabaee7c7d7af442)を参考にしてください。

# なぜSimCTGが必要なのか?

> 昨今のDecoderを持つモデルでは、入力文＋過去に生成した単語列から次の単語列を予測するAuto-Regressiveな生成が行われています。この際に、計算量の関係から、Gready SearchやBeam Searchなどが用いられています。

参考: [テキスト生成における decoding テクニック: Greedy search, Beam search, Top-K, Top-p
](https://zenn.dev/hellorusk/articles/1c0bef15057b1d)

しかし、これらの探索手法を用いた場合、同じ単語を繰り返すなどの問題が発生します。そこで、`huggingface`の文章生成関数`generete`では、`no_repeat_ngram_size`や`repetition_penalty`という引数があり、このパラメータによって繰り返しを抑えることができますが、パラメータによっては、同じ単語列が全く出なくなり不自然な生成になってしまうことがあります。

下図(論文中Fig.1)の(a)は、GPT-2によって生成された各トークンにおける特徴ベクトル(Transformer最終層)のコサイン類似度行列です。見ての通り、文中のトークン間の類似度が0.95以上になっており、互いの特徴ベクトルが近いことがわかります。これは、異なるステップで繰り返しトークンを生成してしまう原因の一つと考えられます。論文中では、このように特徴ベクトルが偏っていることを異方性と言っています。

![Fig.1](https://storage.googleapis.com/zenn-user-upload/c51f3beef16d-20220313.png)

理想的には、生成されたトークンの特徴ベクトル間を識別可能にするため、上図(b)のようにトークン間の類似度が低くなるようにすべきと考えられます。つまり、各トークンの特徴ベクトルは偏らないようにする必要があります。（等方性にすべき。）

そこで、最尤推定（MLE）で言語モデルを学習し最も可能性の高いシーケンスをデコードしていくという従来手法に対して、識別可能で等方的な特徴ベクトルの学習を促すSimCTG損失を提案し、加えて、SimCTGによる学習を補完する新しい復号化手法であるContrastive Search (対照探索)が提案されています。

# SimCTG

SimCTGは、最尤推定の損失（クロスエントロピー損失)と対照損失(Constrastive Loss)からなります。

単語列を$x$とした場合、最尤推定の損失は、以下のようになります。一つ前までの単語列($x_{ < i}$)から生成される単語が$x_i$である確率分布 $p_{\theta}(x)$を最大化するような損失$\mathcal{L}_{MLE}$です。

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

予測確率が高い$k$個の単語集合を$V^k$とした場合のContastive Searchによる単語選択は、

$$
\begin{equation}

x_t = \mathrm{argmax}_{v \in V^k} \{ (1 - \alpha) \times p_{\theta}(v | x_{ < t}) 
- \alpha \times (\mathrm{max} \{ s(h_v, h_{x_j}) : 1 \leq j \leq t - 1 \}) \}

\tag{5}
\end{equation}
$$

で行われます。

# 結果

論文のTable.1を見てください。論文通りの結果だと、SimCTGに、Contrastive Searchを使った際、最強です。一方、両手法を使う必要があるとも見て取れます。

# T5におけるSimCTGの学習の実装

今回は、[【日本語モデル付き】2021年に自然言語処理をする人にお勧めしたい事前学習済みモデル](https://qiita.com/sonoisa/items/a9af64ff641f0bbfed44)で提供されているT5の転移学習例の[コード](https://github.com/sonoisa/t5-japanese)をベースに拡張させていただきました。[コード](https://github.com/sonoisa/t5-japanese)のニュース記事のタイトル生成（一種の文章要約）のリンクからGoogle Colabでの訓練コードが見れます。また、本記事では、簡単のため、修正したSimCTGの重要な部分のみ記載します。

まず、類似度計算の部分です。ここでは、T5のデコーダの最終層を特徴ベクトルとして使用しています。バッチ, 単語列数, 特徴ベクトルからなる行列なので、各単語ごとの特徴ベクトルの類似度を計算し、バッチ数×単語列数×単語列数の類似度行列を出力します。

```python
def t5_cosine_simirality(outputs):
    last_hidden_state = outputs.decoder_hidden_states[-1] # torch.Size([3, 64, 768])
    norm_rep = last_hidden_state / last_hidden_state.norm(dim=2, keepdim=True)
    cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1,2)) # torch.Size([3, 64, 64])
    return cosine_scores
```

事前学習済モデルの類似度行列は、以下のようになっています。(長いので最初の5行5列部分のみです。)
実際、対角部分以外の類似度も高いことが見て取れます。
```
[0.9999, 0.9807, 0.9654, 0.9587, 0.9528, 
[0.9807, 0.9999, 0.9805, 0.9755, 0.9687,
[0.9654, 0.9805, 0.9999, 0.9816, 0.9712,
[0.9587, 0.9755, 0.9816, 0.9999, 0.9878, 
[0.9528, 0.9687, 0.9712, 0.9878, 0.9999,
```

次に、対照損失の部分です。これは、[論文のコード](https://github.com/yxuansu/SimCTG/blob/bb54480e5c43d62d5b660d5cdabaee7c7d7af442/pretraining/simctg.py#L50)と同じです。論文コードのスターをポッチっと押しましょう！

```python
def compute_contrastive_loss(score_matrix, margin):
    '''
        margin: predefined margin to push similarity score away
        score_matrix: bsz x seqlen x seqlen; cosine similarity matrix
        input_ids: bsz x seqlen
    '''
    bsz, seqlen, _ = score_matrix.size()
    gold_score = torch.diagonal(score_matrix, offset=0, dim1=1, dim2=2) #対角成分を取り出す　bsz x seqlen
    gold_score = torch.unsqueeze(gold_score, -1) 
    assert gold_score.size() == torch.Size([bsz, seqlen, 1])

    difference_matrix = gold_score - score_matrix
    assert difference_matrix.size() == torch.Size([bsz, seqlen, seqlen])
    loss_matrix = margin - difference_matrix # bsz x seqlen x seqlen
    loss_matrix = torch.nn.functional.relu(loss_matrix)
    cl_loss = torch.mean(loss_matrix)
    return cl_loss
```

次に学習する部分です。参考にさせていただいた転移学習コードはPytorch Lightningを使用しているので、そのモデル部分の拡張です。
T5の隠れ層を出力するために、`output_hidden_states=True `を設定しています。`_step`部分で損失を追加で計算しています。

```python
class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # 最新版のplでは、hparamsのupdate方法が変更された。
        self.save_hyperparameters(hparams)
        # 事前学習済みモデルの読み込み
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        # トークナイザーの読み込み
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path, is_fast=True)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None):
        """順伝搬"""
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True # decoderの隠れ層を出力するためTrueに設定。
        )

    def _step(self, batch):
        """ロス計算"""
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            labels=labels,
        )

        mle_loss = outputs.loss
        cosine_scores = t5_cosine_simirality(outputs) # 類似度計算
        margin = 0.5 # margin
        cl_loss = compute_contrastive_loss(cosine_scores, margin) # 対照損失計算

        loss = mle_loss + cl_loss # SimCTG 損失
        loss = loss.mean()
        return loss
```

上記の変更を行なったコードで学習した結果、以下の類似度のようになりました。(同様に、長いので最初の5行5列部分のみです。)
対角成分は、類似度が高く、異なるトークン間の類似度である対角成分以外は、類似度が小さくなっていることがわかります。

```
[0.9999, 0.2982, -0.5032, -0.3650, -0.5413,
[0.2982, 0.9999, 0.2020, 0.2578, 0.2829,
[-0.5032, 0.2020, 0.9999, 0.6498, 0.6361,
[-0.3650, 0.2578, 0.6498, 1.0, 0.6117, 0.7059,
[-0.5413, 0.2829, 0.6361, 0.6117, 0.9999
```

# T5におけるContrastive Searchの実装

まず、推論の大まかな流れです。重要な部分は、前の単語列から次の単語を予測する`ContrastiveDecodingOneStepFast`という関数で、後で解説します。


```python
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
input_ids = next(iter(data_loader))["source_ids"] # データローダから読み込み
batch_size, seqlen = input_ids.size()

generated = [[] for _ in range(batch_size)] # バッチごとの生成文格納するリストです。
is_eos = [False for _ in range(batch_size)] # バッチごとの文章生成終了(EOS)判定リストです。
# past_key_valuesをモデルの入力とすることで、モデルの入力を一つ前の単語のみにできる(huggingfaceの実装のドキュメント見てください。)
past_key_values = None 
last_hidden_states = None
logits = None

decoding_len = 512 # 文章の最大長さ
beam_width = 3 # 実装では、ビームサーチの結果にContrastive Searchを行なっていた。
alpha = 0.5 # (1 - a) * 確率最大 + a * 類似度

input_ids.to(device)
trained_model.eval()
for step in range(decoding_len):
    next_ids, past_key_values, last_hidden_states, logits = ContrastiveDecodingOneStepFast(
        trained_model,
        input_ids,
        beam_width,
        alpha,
        past_key_values,
        last_hidden_states,
        tokenizer,
        logits,
        device,
        first_step=step == 0, #最初のステップのみ扱い違う
    )
    tokens = next_ids.squeeze(dim=-1).tolist()
    for idx, t in enumerate(tokens): #バッチ数分
        # EOSの場合スキップ
        if is_eos[idx]:
            continue
        if t == tokenizer.eos_token_id:
            is_eos[idx] = True
            continue
        generated[idx].append(t)
```

次に、`ContrastiveDecodingOneStepFast`の実装です。論文の実装は、GPT-2版ですが、T5はエンコーダ・デコーダモデルなので、デコーダの入力が必要になります。また、GPU使用していた際に、メモリーエラーになっていたので、その対処を入れています。[論文実装はこの部分](https://github.com/yxuansu/SimCTG/blob/bb54480e5c43d62d5b660d5cdabaee7c7d7af442/document_generation/utlis.py#L126)になります。ほとんど論文の実装と同じです。


```python
def ContrastiveDecodingOneStepFast(
    model, 
    ids, 
    beam_width, 
    alpha, 
    past_key_values,
    last_hidden_states,
    vocab,
    logit_for_next_step,
    device,
    first_step=False,
    
    ):
    # input_ids: [B, S]
    model.eval()
    if first_step:
        # T5では、最初のステップのみpad tokenをデコーダの入力とする。
        with torch.no_grad():
            bsz, _ = ids.size()
            ids = ids.to(device)
            decoder_inputs = torch.tensor([[0] for _ in range(bsz)]).to(device) # pad_token_idでstart
            output = model(
                input_ids=ids, 
                decoder_input_ids=decoder_inputs,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True
            )
            del decoder_inputs
        past_key_values = output.past_key_values
        last_hidden_states = output.decoder_hidden_states[-1].cpu()    # [B, S, E]
        logit_for_next_step = output.logits[:, -1, :].cpu()    # [B, V]
    bsz, seqlen, embed_dim = last_hidden_states.size()
    p = random.uniform(0, 1)

    next_probs = F.softmax(logit_for_next_step, dim=-1)

    # 最大確率のk候補を取得
    _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width) # [B, V:38000くらい] -> [B, K:beam_width]
    top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids) # 候補の確率を取得 [B, K] 
    
    # モデルの入力のpast_keyをバッチx候補のタプルに修正
    past_key_values = enlarge_past_key_values(past_key_values, beam_width)

    # 次の単語を予測
    input_ids = top_k_ids.view(-1, 1).to(device) # [B*K , 1]
    with torch.no_grad():
        output = model(
            input_ids=input_ids, 
            decoder_input_ids=input_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
    past_key_values = output.past_key_values
    logits = output.logits[:, -1, :].cpu()    # [B*K, V]
    next_hidden = output.decoder_hidden_states[-1].cpu()    # [B*K, 1, E]
    context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz*beam_width, seqlen, embed_dim)    # [B*K, S, E]

    # バッチごとの最大スコアの単語を選択
    # 実際の式の部分を計算 (下の関数)
    selected_idx = ranking_fast(
        context_hidden, 
        next_hidden, 
        top_k_probs,
        alpha,
        beam_width,
    )  # [B]

    # 次のステップの準備
    next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)    # [B, 1]
    next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))    # [B, K, E]
    next_hidden = next_hidden[range(bsz), selected_idx, :]    # [B, E]
    last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)    # [B, S, E]
    past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
    logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]    # [B, V]
    
    # GPUのメモリ解放
    if device != torch.device("cpu"):
        del model, input_ids, top_k_ids, top_k_probs
        torch.cuda.empty_cache()
    return next_id, past_key_values, last_hidden_states, logits 


def ranking_fast(context_hidden, next_hidden, next_top_k_probs, alpha, beam_width):
    '''バッチごとの最大スコアの単語を選択
        context_hidden: bsz*beam x seqlen x embed_dim
        next_hidden: bsz*beam x 1 x embed_dim
        next_top_k_probs: bsz x beam
    '''
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)    # [B*K, S]
    scores, _ = torch.max(cosine_matrix, dim=-1)    # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)    # [B*K]
    # 式の部分
    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores #
    scores = torch.stack(torch.split(scores, beam_width)) # バッチごとに戻す [B, K]
    selected_idx = scores.max(dim=-1)[1] # [B]
    return selected_idx

def enlarge_past_key_values(past_key_values, beam_width):
    # モデルの入力のpast_keyをバッチx候補のタプルに修正
    # from [B, num_head, seq_len, esz] to [B*K, num_head, seq_len, esz]
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            # item is the key and value matrix
            bsz, num_head, seq_len, esz = item.size()
            item = item.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(bsz*beam_width, num_head, seq_len, esz)    # [bsz*beam, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values

def select_past_key_values(past_key_values, beam_width, selected_idx):
    '''バッチx候補から最大スコアのpast_keyを選択
    select_idx: [B]
    '''
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            bsz = int(bsz_and_beam//beam_width)
            item = torch.stack(torch.split(item, beam_width, dim=0))    # [B, K, num_head, seq_len, esz] 
            item = item[range(bsz), selected_idx, :, :, :]   # [B, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values
```

以上、T5のConsrastive Search実装になります。

# まとめ

文章生成の改善手法を探していて、偶然見つけた論文ですが、簡単に実装でき、効果が高そうな手法な気がします。
再現実験までやったわけではありませんが、今後、どんどん使っていきたい手法です。

個人的に、Contrastive Learningは、好きな分野で、画像系ですがSimSiamの記事を書いています。今後も、この分野には着目していきたいですね。

