---
title: "OpenAIの音声認識モデル Whisperの解説 / Fine Tuning 方法" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

OpenAIから、かなりすごい音声認識モデル [Whisper](https://github.com/openai/whisper)が発表されました。特出すべき点は、68万時間という、かなりヤバめのデータ量で訓練しており、英語では商用の音声認識システムや人間の書き起こしに匹敵しているとのことです。

社内でも日本語、ブルガリア語、韓国語で試してみましたが、すごい精度でした。日本語の場合、漢字の間違いが多々ありましたが、発音は大体あってそうでした。ブルガリア語は、ロシア語で認識されていました。韓国語は、完璧でした。

しかし、[Github](https://github.com/openai/whisper)に公開されたコードを見てみると、訓練コードが含まれておらず、公開の予定もないそうです。そこで、本記事では、Whisperの解説に加えて、試しに作成してみたFine Tuningの方法を解説します。

※ Fine Tungingが上手くいってそうに見えるのですが、正確なコードではないと思います。気付いた点がありましたら、コメントください。

全てのコードは、[Whisper](https://github.com/openai/whisper)の[Discussion: Finetuning/Training code ?](https://github.com/openai/whisper/discussions/64#discussioncomment-3765117)をみてください。

# Whisperとは

1. 英語のみならず、日本語を含む多言語でSoTAに匹敵する性能
2. 音声の言語認識、音声区間検出、タイムスタンプの出力ができるらしい (試していない...)
3. MITライセンスで公開されている！

というすごく、ありがたい音声認識手法です。

これまでの音声認識の流れは、[wav2vec2.0](https://arxiv.org/abs/2006.11477)など教師なしのデータセットによる事前学習がメインでした。一方で、Whisperは、教師付きデータセットを68万時間(日本語:7054時間)にスケールさせることで、多くの言語/ドメインへのゼロショット転移を可能にしました。

LibriSpeechで学習したwav2vec2.0と比較した結果が、論文中のTable. 2の以下のテーブルです。各データセットに対するWER(word error rate)を示しています。

![](https://storage.googleapis.com/zenn-user-upload/e94a891042cd-20220930.png)

wav2vec2.0の学習対象であるLibriSpeechにおいて、Whisperはwav2vec2.0と同等の性能を示しており、他のドメインのデータセットでは、誤り率をかなり低減できていることがわかります。とてもすごいゼロショット性能です！


また、英語における人間による書き起こしとの比較ですが、以下の図になります。左側が、音声認識システム、右側が人間による書き起こしのWERです。これもすごい結果で、人間とWhisperはほとんど精度差がありません。

![](https://storage.googleapis.com/zenn-user-upload/7179ee64bd6a-20220930.png)

他にも、多言語における考察など[論文](https://cdn.openai.com/papers/whisper.pdf)に記載されているので、詳細を知りたい方は、要チェックです。

# 手法の概要

モデルのアーキテクチャは、以下の図のように単純で、音声を入力としたエンコーダ、デコーダ形式のTransformerです。
デコーダでは、エンコーダで抽出した音声特徴量をクロスアッテンションの入力としています。そして、SOT(Start of transcript)を最初のトークンとして、デコーダを繰り返すだけです。

実際、音声認識をする際には、以下の図のようにSOT、Langage Tag、Transcribe、No Timestampを組み合わせでスタートすれば良いです。

例えば、「こんにちは」という日本語の場合は、(SOT, Ja, Transcribe, No Timestamp)をはじめのデコーダ入力とし、結果として、(Ja, Transcribe, No Timestamp, こ)が推定され、その後、(SOT, Ja, Transcribe, No Timestamp, こ)を入力とし、(Ja, Transcribe, No Timestamp, こ, ん)が推定されていくみたいな感じです。「こ」、「ん」などの単語は、デコーダで各トークンのクラス確率を推定されるので、その最大値のIndexをとり、それをトークナイザーで文字や単語に直すという手順で取得できます。

![](https://storage.googleapis.com/zenn-user-upload/89b506b237a9-20220930.png)


# Fine Tuning

次に、WhisperのFine Tuning方法について解説します。今回は、簡単のため[JVSコーパス](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)をデータセットをOpenJtalkでカナに変換したものを使用しています。

また、実行場所は、Google Colabです。
重要な部分のみ記載するので、詳細な全てのコードは、[Whisper](https://github.com/openai/whisper)の[Discussion: Finetuning/Training code ?](https://github.com/openai/whisper/discussions/64#discussioncomment-3765117)をみてください。


## ライブラリのインストール

以下のコマンドで、Whisperをインストールします。今回は、`pytorch lightning`を使用します。CERの計算のため`evaluate`もインストールしています。

```s
%%capture
! pip install git+https://github.com/openai/whisper.git
! pip install jiwer
! pip install pyopenjtalk==0.3.0
! pip install pytorch-lightning==1.7.7
! pip install -qqq evaluate==0.2.2
```

## audioの読み込み

最近、音響ファイルの読み込みは、`torchaudio`を使用しているのですが、より早く読み込んで、sample rateを変換できる方法を知っている人は教えて欲しいです。

```python
def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform
```

## JVS データセットのダウンロード

JVSデータセットをgoogle driveからダウンロードします。

```s
%%capture
import gdown
gdown.download('https://drive.google.com/u/0/uc?id=19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt', 'jvs.zip', quiet=False)
!unzip jvs.zip -d ./jvs

import IPython.display
IPython.display.Audio("/content/jvs/jvs_ver1/jvs001/nonpara30/wav24kHz16bit/BASIC5000_0025.wav")
```

## データローダ関連

まずは、データセットの読み込みです。

```python
def text_kana_convert(text):
    text = pyopenjtalk.g2p(text, kana=True)
    return text

class JvsSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list # List[(audioのID, audioのパス, audioのテキスト)]
        self.sample_rate = sample_rate # 16_000
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.audio_info_list)
    
    def __getitem__(self, id):
        audio_id, audio_path, text = self.audio_info_list[id]

        # audioを読み込みめるスペクログラムに変換
        audio = load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten()) # 480000サンプル(30sec)に長さを揃える
        mel = whisper.log_mel_spectrogram(audio) # メルスペクログラムに変換

        text = text_kana_convert(text) # テキストをカナに変換

        # <|start of transcribe|><|ja|><|transcribe|><|notimestamps|>コンニチハ<|endoftext|>が最終的に得たい形式

        # デコーダの入力は <|start of transcribe|><|ja|><|transcribe|><|notimestamps|>コンニチハ
        # <|endoftext|> なし
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)

        # 正解ラベルは、<|ja|><|transcribe|><|notimestamps|>コンニチハ<|endoftext|>
        # <|start of transcribe|> なし
        labels = text[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "dec_input_ids": text,
            "labels": labels
        }
```

データをバッチ化する際に、トークン化したテキストの長さを合わせる必要があるため、`Collator`も実装しました。
実装は、[sanchit-gandhi/whisper-medium-switchboard-5k](https://huggingface.co/sanchit-gandhi/whisper-medium-switchboard-5k/blob/main/run_speech_recognition_whisper.py#L444)も参考にさせてもらいました。

```python
class WhisperDataCollatorWhithPadding:
    def __call__(sefl, features):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])
        
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids

        return batch
```

以下のような感じで、使用することができます。

```python
woptions = whisper.DecodingOptions(language="ja", without_timestamps=True)
wmodel = whisper.load_model("base")
wtokenizer = whisper.tokenizer.get_tokenizer(True, language="ja", task=woptions.task)

dataset = JvsSpeechDataset(eval_audio_transcript_pair_list, wtokenizer, SAMPLE_RATE)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())
```

## 学習コード

最後に学習コードです。これは、pytorch lightningに沿って実装しています。

まずは、`Config`です。

```python
class Config:
    learning_rate = 0.0005
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2
    batch_size = 16
    num_worker = 2
    num_train_epochs = 10
    gradient_accumulation_steps = 1
    sample_rate = SAMPLE_RATE
```

次にモデルやデータセットの設定です。

```python
class WhisperModelModule(LightningModule):
    def __init__(self, cfg:Config, model_name="base", lang="ja", train_dataset=[], eval_dataset=[]) -> None:
        super().__init__()
        # モデルやトークナイザーの設定です。
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="ja", task=self.options.task)

        # エンコーダによる音響特徴量の抽出部分の学習は行いません。
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        
        # Discussionに書かれてましたが、CrossEntropyを使っているそうです
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # WERとCERを計算する関数です。
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.cfg = cfg
        self.__train_dataset = train_dataset # List[(audioのID, audioのパス, audioのテキスト)]
        self.__eval_dataset = eval_dataset # List[(audioのID, audioのパス, audioのテキスト)]
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids) # ここは学習しない

        out = self.model.decoder(dec_input_ids, audio_features) # デコーダのみ学習
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()


        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        # 以下、トークンをカナ(テキスト)に変換
        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o, skip_special_tokens=True)) 
            l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)

        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.cfg.learning_rate, 
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps, 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):
        """初期設定"""

        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
    
    def train_dataloader(self):
        """訓練データローダーを作成する"""
        dataset = JvsSpeechDataset(self.__train_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

    def val_dataloader(self):
        """評価データローダーを作成する"""
        dataset = JvsSpeechDataset(self.__eval_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )
```

訓練メインコードです。

```python
log_output_dir = "/content/logs" # ログの出力先
check_output_dir = "/content/artifacts" # チェックポイントの出力先
train_name = "whisper"
train_id = "00001"
model_name = "base" # whisperのpratrainです
lang = "ja" # 日本語！

cfg = Config()

# 出力フォルダを作成
Path(log_output_dir).mkdir(exist_ok=True)
Path(check_output_dir).mkdir(exist_ok=True)

# tensor boardでロギングします
tflogger = TensorBoardLogger(
    save_dir=log_output_dir,
    name=train_name,
    version=train_id
)

# チェックポイントの出力設定です
# とりあえず、全て出力するようにしています
checkpoint_callback = ModelCheckpoint(
    dirpath=f"{check_output_dir}/checkpoint",
    filename="checkpoint-{epoch:04d}",
    save_top_k=-1 # all model save
)

callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]

# モデルモジュールの作成です
model = WhisperModelModule(cfg, model_name, lang, train_audio_transcript_pair_list, eval_audio_transcript_pair_list)

# 訓練開始です。fp16で学習するよう設定しています。
trainer = Trainer(
    precision=16,
    accelerator="gpu",
    max_epochs=cfg.num_train_epochs,
    accumulate_grad_batches=cfg.gradient_accumulation_steps,
    logger=tflogger,
    callbacks=callback_list
)

trainer.fit(model)
```

以上、Fine Tuningのコードでした。

# 推論

次に、実際に推論してみて評価します。

```python
checkpoint_path = "/content/artifacts/checkpoint/checkpoint-epoch=0009.ckpt"

# state_dictには、epochやoptimizerの情報も格納されているので、state_dictだけ取り出します。
state_dict = torch.load(checkpoint_path)
print(state_dict.keys())
state_dict = state_dict['state_dict']

# モデルを作成し、重みをロードします
whisper_model = WhisperModelModule(cfg)
whisper_model.load_state_dict(state_dict)

# トークナイザ作成
woptions = whisper.DecodingOptions(language="ja", without_timestamps=True)
wtokenizer = whisper.tokenizer.get_tokenizer(True, language="ja", task=woptions.task)

# データローダ作成
dataset = JvsSpeechDataset(eval_audio_transcript_pair_list, wtokenizer, SAMPLE_RATE)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())

# 推論処理です！
refs = []
res = []
for b in tqdm(loader):
    input_ids = b["input_ids"].half().cuda()
    labels = b["labels"].long().cuda()
    with torch.no_grad():
        results = whisper_model.model.decode(input_ids, woptions)
        for r in results:
            res.append(r.text)
        
        for l in labels:
            l[l == -100] = wtokenizer.eot
            ref = wtokenizer.decode(l, skip_special_tokens=True)
            refs.append(ref)
```

実際に、CERで評価してみると、CER: `0.0179`で、かなり良い気がします。タスクが簡単なので、性能の限界を見れている気はしませんが、学習はできているようです。

# まとめ

最新の、めっちゃ性能が良い音声認識モデルWhisperのFine Tuning方法を解説しました。正直なところ、正確な訓練方法か怪しいですが、一応、できてそうな気がします。もし、気付いた点があれば、コメントいただけると嬉しいです。

----

画像・自然言語・音声に関する機械学習の研究開発やMLOpsを行っています。もし、機械学習に関して、ご相談があれば、[@kwashizzz](https://twitter.com/kwashizzz)のアカウントまでDMしてください！

<iframe class="meety-embed" src="https://meety.net/embed/matches/DtCWdBaoxEoe?service=meety&color=pink"
  style="border: 0; display: block; max-width: 99%; width: 498px; height: 228px; padding: 0px; margin: 0px; position: static; visibility: visible;"></iframe>


これまでの、[機械学習記事のまとめ](https://zenn.dev/kwashizzz/articles/my-ml-articles-summary)です。
