---
title: "spaCy/GiNZAの形態素解析処理 Sudachiの設定ファイルを変更する方法" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習", "nlp"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
---

GiNZAの形態素解析のライブラリとして、Sudachiが使用されています。そのため、GiNZAの解析にユーザ定義辞書を追加したい場合は、Sudachiにユーザ定義辞書を追加する必要があります。

前回、[Sudachiにユーザ定義辞書を追加する方法](https://zenn.dev/kwashizzz/articles/nlp-sudachi-user-dic)を解説する記事を書きました。
この時は、Sudachiのデフォルトの設定ファイルにユーザ辞書のパスを追記していました。しかし、ユーザごとに辞書を切り分けたい場合があります。そこで、Sudachi設定ファイルをGiNZAから指定し、切り替える方法を解説します。


使用ライブラリー
```
spacy.__version__, ja_ginza.__version__, sudachipy.__version__
('3.2.2', '5.1.0', '0.6.3')
```

# 例

例えば、ユーザA専用の辞書専用の辞書を作成したとします。その結果、[Sudachiにユーザ定義辞書を追加する方法](https://zenn.dev/kwashizzz/articles/nlp-sudachi-user-dic) にならい、`/workspace/.sudachi/user.dic`が作成されたとします。

Sudachi関連のデフォルトの設定ファイルや必要なリソースは、私の環境では、`/root/.pyenv/versions/3.9.7/lib/python3.9/site-packages/sudachipy/resources`にあるので、`.sudachi`にコピーします。その`.sudachi/resource`の中に、設定ファイル`sudachi.json`や`char.def`などの設定ファイルに記載されたファイル群があります。

この`sudachi.json`設定ファイルをユーザA専用とし、[Sudachiにユーザ定義辞書を追加する方法](https://zenn.dev/kwashizzz/articles/nlp-sudachi-user-dic)の記事のように、ユーザ定義辞書のパスを追記します。

この設定ファイルを読み込んだSudachiを形態素解析器として、spaCy/GiNZAで解析を行う方法を紹介します。

#　初期設定

ここでは、ユーザ専用の初期設定を行う方法を記載します。
まず、[Sudachiにユーザ定義辞書を追加する方法](https://zenn.dev/kwashizzz/articles/nlp-sudachi-user-dic)に沿って、ユーザ定義辞書(`user.dic`)を作成してください。

## ユーザ専用設定

ここでは、ユーザ専用のsudachiリソース群を作成します。まずは、必要なリソースのコピーです。

以下のプログラムで、リソースが入っているであろうディレクトリがわかります。

```python
import sys

site_package = ""
for p in sys.path:
    if "site-packages" in os.path.basename(p):
        site_package = p
print(site_package)
# /root/.pyenv/versions/3.9.7/lib/python3.9/site-packages
```

`{site_packages}/sudachipy/resources`に入っているので、ディレクトリごと `.sudachi`にコピーします。

```s
!cp -r {site_packages}/sudachipy/resources ./.sudachi/
```

`.sudachi/resources/sudachi.json`に`"userDict" : ["/workspace/.sudachi/user.dic"]`を追記します。

これで、初期設定は終わりです。

# Spacy/GiNZAでユーザ定義辞書を設定

以下のプログラムで、Spacy/GiNZAの形態素解析器であるSudachiの辞書をユーザ定義辞書を用いた設定に変更できます。



```python
from spacy.lang.ja import JapaneseTokenizer

# https://github.com/megagonlabs/ginza/blob/61cb655f2e5c85980f1a1bbc7d833623931e4235/ginza/analyzer.py
# を参考に、config_pathを設定できるように修正しました。
def try_sudachi_import(split_mode="A", config_path=None):
    try:
        from sudachipy import dictionary, tokenizer

        split_mode = {
            None: tokenizer.Tokenizer.SplitMode.A,
            "A": tokenizer.Tokenizer.SplitMode.A,
            "B": tokenizer.Tokenizer.SplitMode.B,
            "C": tokenizer.Tokenizer.SplitMode.C,
        }[split_mode]

        tok = dictionary.Dictionary(config_path=config_path).create(mode=split_mode)
        return tok
    except ImportError:
        raise ImportError(
            "Japanese support requires SudachiPy and SudachiDict-core "
            "(https://github.com/WorksApplications/SudachiPy). "
            "Install with `pip install sudachipy sudachidict_core` or "
            "install spaCy with `pip install spacy[ja]`."
        ) from None

class MySudachiTokenizer(JapaneseTokenizer):
    def __init__(self, nlp, config_path=None) -> None:
        self.nlp = nlp
        self.vocab = nlp.vocab
        self.split_mode= nlp.tokenizer.split_mode
        self.tokenizer = try_sudachi_import(
            split_mode=self.split_mode, config_path=config_path
        )
        
        self.need_subtokens = not (self.split_mode is None or self.split_mode == "A")
```

上記の関数を使用し、辞書の設定ファイルを変更します。実際に変更して、解析する例です。

```python
import ginza

nlp = spacy.load('ja_ginza')
ginza.set_split_mode(nlp, "A")

config_path = "/workspace/.sudachi/resources/sudachi.json" # 設定ファイルのパス　(userDictを追記済み)
nlp.tokenizer = MySudachiTokenizer(nlp, config_path=config_path) # 設定ファイルの辞書に変更したトークナイザーに変更します。

# 以下、Spacyによる解析です。
text = "ルイズ・フランソワーズは、ゼロの使い魔のヒロインです。"
doc = nlp(text)
for sent in doc.sents:
    for token in sent:
        print(token.i, token.orth_, token.tag_)
```

以下のように、ユーザ定義辞書に記載した固有名詞の区切りで出力されると思います。

```s
0 ルイズ・フランソワーズ 名詞-固有名詞-人名-一般
1 は 助詞-係助詞
2 、 補助記号-読点
3 ゼロの使い魔 名詞-固有名詞-一般
4 の 助詞-格助詞
5 ヒロイン 名詞-普通名詞-一般
6 です 助動詞
7 。 補助記号-句点
```

以上、spaCy/GiNZAで、ユーザ定義辞書の設定ファイルに変更する方法でした。



