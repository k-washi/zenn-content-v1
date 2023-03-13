---
title: "WSL2で同一環境を複数インストール" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["初心者"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
publication_name: "fusic"
---

こんにちは！鷲崎([@kwashizzz](https://twitter.com/kwashizzz))です。
最近、Windowsに乗り換えましたが、WSL環境を使い回せれば便利だと思い、Ubuntu22.04を別の環境としてインストールしてみました。そこで、詰まった部分があったので、記載しておきます。

基本的には、管理者権限で、[WSL 上で同一ディストリビューションの環境を複数インストール･管理する](https://qiita.com/souyakuchan/items/9f95043cf9c4eda2e1cc)を実行すればうまくいきます。

しかし、Ubuntu22.04をインストールした際に、ユーザを作成しましたが、rootユーザで入ってしまい、`update`
などがうまく動作しませんでした。

そこで、`wsl --distribution 作成したディストリビューション名 --user <ユーザ名>` で環境に入ると、うまく動作しました。

また、環境が壊れた際に、`wsl --unregister <削除するディストリビューション>`　としたところ、wslが動作しなくなしました。

その際には、[起動しなくなったWSL2を復活させるまでにしたあれこれ](https://zenn.dev/karaage0703/articles/e30c9614a55bdb)の、`WSL2無効化→有効化`を試すと動作するようになりました。この辺も管理者権限で行っていなかったから動作しなくなった気がします...

