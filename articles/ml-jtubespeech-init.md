---
title: "JTubeSpeech: YouTubeによる日本語音声コーパスの構築方法" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
---

# はじめに

こんにちは、[わっしー](https://twitter.com/kwashizzz)です。この記事では、[JTubeSpeech](https://github.com/sarulab-speech/jtubespeech)の日本語音声認識コーパスをローカル環境に構築する方法を解説します。

※ masterが一部動作しなかったため、実行を確認したコードを、[k-washi/jtubespeech/tree/dev_kwashi](https://github.com/k-washi/jtubespeech/tree/dev_kwashi)に置いています。もしかしたら、音声認識ライブラリなどのバージョンの違いで動作しなかったかもしれません。

※内容は、2022年8月11日に作成しました。

# JTubeSpeech とは

[JTubeSpeech: 音声認識と話者照合のために YouTube から構築される日本語音声コーパス](https://www.slideshare.net/ShinnosukeTakamichi/jtubespeech-youtube)のスライドがわかりやすいです。

1. 音声言語研究に必要なオープンな日本語音声コーパスです
2. YouTube動画をベースにコーパスを自動構築しています
3. 音声認識・話者照合用のコーパスです

多言語対応だが、本記事では、日本語のコーパス構築に関してのみ解説します。

# JTubeSpeechの環境構築

git lfsにより、**./data/ja/*.csv**をダウンロードする必要があります。
このcsvに、日本語音声コーパスのyoutubeのVideoIdが記載されています。
```
git clone https://github.com/sarulab-speech/jtubespeech.git
sudo apt install git-lfs
git lfs pull
```

pythonのライブラリをインストールします。以下、必要なライブラリです。

```txt:requiremnts.txt
espnet==202205
espnet-model-zoo==0.1.7
youtube-dl==2021.12.17
joblib==1.1.0
pydub==0.25.1
pandas==1.4.2
num2words==0.5.11
neologdn==0.5.1
romkan==0.2.1

```
**requirements.txt**に記載して、インストールしてください。

```
pip install -r requirements.txt
```

# Youtubeから音声と字幕をダウンロード

Youtubeから、`./data/ja/*.csv`に記載してあるVideoIdのビデオをダウンロードし、音声と字幕を抽出します。この操作は、READMEのstep4にあたります。

引数を適当に設定して、以下のコマンドで実行してください。

```sh
python scripts/download_video.py ja ./data/ja/202103.csv --outdir /audio/downloads
```

かなり時間がかかるので、注意してください。私の環境では、2週間ほどかかりました。途中実行が落ちないように、以下のようにコマンドを実行するのもおすすめです。

```
nohup python scripts/download_video.py ja ./data/ja/202103.csv --outdir /audio/downloads > ./nohop.out &
```

ダウンロードしたデータ数は、以下のコマンドで確認できます。

```sh
ls -1 /audio/downloads/ja/wav16k/*/ | wc -l
```

# 日本語音声認識コーパスデータセットの構築

ダウンロードした音声と字幕に対して、CTCスコアを計算することで、コーパスを作成します。この操作は、READMEのstep5にあたります。

動作確認済みのコードを、[k-washi/jtubespeech/tree/dev_kwashi](https://github.com/k-washi/jtubespeech/tree/dev_kwashi)に置いているので、コピーしてください。

1. まず、必要な音声認識モデルをダウンロードします。
`./scripts/align_model_download.py`でダウンロードします。こちらも上記の[github](https://github.com/k-washi/jtubespeech/tree/dev_kwashi)にコードを追加しています。

```
python ./scripts/align_model_download.py
```

2. 次に、字幕テキストと音声のアライメントによりCTCスコアを計算します。
`scripts/align.py`でエラーが発生したため、別のブランチに、[`scripts/align2.py`](https://github.com/k-washi/jtubespeech/blob/dev_kwashi/scripts/align2.py)を作成しています。

また、CUDAのメモリエラー対策のため、`longest_audio_segments=160`としています。(24564MiBの環境です。)

以下のように、処理を実行しています。

```
nohup python scripts/align.py --wavdir=/audio/downloads/ja/wav16k/ --txtdir=/audio/downloads/ja/txt/ --output=/audio/downloads/ja/asr/ --longest_audio_segments=160 --ngpu=1 > ./nohup/align.out & 
```

ログは以下で確認できます。

```sh
tail  /audio/downloads/ja/asr/segments.log 
```

また、実行結果は、以下で確認できます。

```sh
tail  /audio/downloads/ja/asr/segments.txt

# 以下のような出力です
# ファイル名_ID ファイル名 開始時間 終了時間 CTCスコア テキスト
LY9ewzIY8iQ_0114 LY9ewzIY8iQ 559.37 560.76 -1.8345 そうですね。今は
LY9ewzIY8iQ_0115 LY9ewzIY8iQ 560.76 564.70 -4.8862 ドゥーチェのお見合い相手、とでも言っておきますか
```

CTCスコアの閾値を`ms`として設定し、別の`segments.txt`ファイルに出力します。
```
awk -v ms=-0.3 '{ if ($5 > ms) {print} }' /audio/downloads/ja/asr/segments.txt > ./.data/segments.txt &
```
これが、音声認識コーパスとなります。


# 最後に

今回は、JTubeSpeechの日本語音声認識コーパスの作成方法を記事にしました。試行錯誤しながら実行したので、誤っている部分があったらご指摘のほど、お願いいたします。

また、音声合成などもやっていきたいので、話者式別用のコーパスの作成も、今後やっていければと思います。

---

画像・自然言語・音声に関する機械学習の研究開発やMLOpsを行っています。もし、機械学習に関して、ご相談があれば、[@kwashizzz](https://twitter.com/kwashizzz)のアカウントまでDMしてください！
これまでの、[機械学習記事のまとめ](https://zenn.dev/kwashizzz/articles/my-ml-articles-summary)です。