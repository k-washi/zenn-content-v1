---
title: "信号処理による話者性変換を用いた音声データ拡張" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習", "音声"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
publication_name: "fusic"
---

こんにちは！[Fusic](https://fusic.co.jp/) 機械学習チームの鷲崎です。機械学習モデルの開発から運用までなんでもしています。もし、機械学習で困っていることがあれば、気軽に[お問い合わせ](https://fusic.co.jp/contact/)ください。

最近、[so-vits-svc(SoftVC VITS Singing Voice Conversion)](https://github.com/svc-develop-team/so-vits-svc)という、性能の良いVoice Conversion (VC)が流行っている気がします。VCでは、入力音声から話者性を除去したコンテンツ情報の抽出が重要になります。コンテンツ情報に一貫性を持たせて、話者性を変更するデータ拡張では、イコライザーを用いたデータ拡張に加えて、ピッチを変更するデータ拡張がよく利用されます。

イコライザーを用いた音声データ拡張に関しては、[前回](https://zenn.dev/fusic/articles/ml-content-vec-equalizer-aug)解説しました。

