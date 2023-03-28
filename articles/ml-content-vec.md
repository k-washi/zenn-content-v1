---
title: "【論文読み】ContentVec: 話者情報の切り離しによるSpeech表現の自己教師あり学習の改善" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
publication_name: "fusic"
---

こんにちは！鷲崎([@kwashizzz](https://twitter.com/kwashizzz))です。最近、[so-vits-svc(SoftVC VITS Singing Voice Conversion)](https://github.com/svc-develop-team/so-vits-svc)という、性能の良いVoice Conversion (VC)が流行っている気がします。構造的には、普通のVCとほとんど変わらないと思いますが、4.0で、本記事で紹介するContentVecが使用されていたりと、実用的な工夫がおもしろいです。

そこで、私が開発中のVCにも、ContentVecを取り入れるべく、論文を読んでみたので、その内容を解説します。

[ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers](https://arxiv.org/abs/2204.09224)


# ざっくりまとめ

音声の自己教師学習(SSL)は、大規模な未注釈音声コーパスに対して音声表現ネットワークを学習させ、学習した表現を下流のタスクに適用するものである。SSL学習の下流タスクの大半は、音声の内容情報に主眼を置いているため、最も望ましい音声表現は、話者のバリエーションなどの不要なバリエーションを内容から切り離すことができるものでなければならない。しかし、話者の情報を取り除くことは、容易に内容も失うことになり、後者の損害は前者の利益をはるかに上回るのが普通であるため、話者の切り離しは非常に困難である。本論文では、コンテンツの深刻な損失なしに話者の分離を達成することができる新しいSSL手法を提案する。我々のアプローチはHuBERTフレームワークから採用され、教師（マスクされた予測ラベル）と生徒（学習された表現）の両方を正則化するディスエンタングリング機構を組み込んでいる。我々は、コンテンツに関連した一連の下流タスクで話者分離の利点を評価し、話者分離された表現が一貫して顕著なパフォーマンス上の利点を持つことを観察した。

