---
title: "NLPのはじめの一歩！Mecab+Neologdのインストール・使用方法" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
---


普段は画像処理を行っている私ですが、本日、初めてNLPのプログラムを書きました。その際、Mecabの使用から躓いてしまったので、メモを残したいと思います。

以下、Google Colabで試して動いた方法を紹介します！


# Mecabのインストール

```bash
!apt install aptitude swig
!aptitude install mecab libmecab-dev mecab-ipadic-utf8
!pip install mecab-python3
```

# Neologdのインストール
Neologdは、比較的に新しい言葉も含んだ辞書で、NLPをやっているチームの同僚におすすめされました！
今回は、この辞書の導入方法も紹介します。
``` bash
!git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
!./mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -a -y
```

インストールされた場所は、以下のコマンドで確認しました。

```bash
!echo `mecab-config --dicdir`"/mecab-ipadic-neologd"

#/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd
```

# Mecabで分かち書きをしてみた

インストールも終わったので、Mecabで言葉の区切りに空白を入れる分かち書きをやってみました。  
しかし、このままでは、以下のエラーが発生してしまいました。

```
error message: [ifs] no such file or directory: /usr/local/etc/mecabrc 
```

そこで、mecabrcを探索すると、以下の場所にありました。
```bash
!find / -iname mecabrc
#/etc/mecabrc
```

上記のエラーの対処も含めて、以下が、分かち書きを行うコードです。

```python
import re
import MeCab
import os
os.environ["MECABRC"] = '/etc/mecabrc'

def remove_brackets(inp):
    """
    記号とかを除く
    https://tech.fusic.co.jp/posts/2021-04-23-bert-multi-classification/
    """
    brackets_tail = re.compile('【[^】]*】$')
    brackets_head = re.compile('^【[^】]*】')
    output = re.sub(brackets_head, '', re.sub(brackets_tail, '', inp))
    return output

class MecabTokenizer():
    """
    分かち書きを行うクラス
    https://qiita.com/sudo5in5k/items/f89d9dc1bec1ed221ede
    """
    def __init__(self, dic_path):
        self._mecab = MeCab.Tagger(f'-Owakati -d {dic_path}')
       
    def parse(self, line):
        """
        args:
            line: "私は猫"
        output:
            r: ["私", "は", "猫"]
        """
        line = remove_brackets(line) # いらない文字は除去
        r = []

        self._mecab.parse('') #バグ対策??
        node = self._mecab.parseToNode(line)
        while node:
            # 単語を取得
            if node.feature.split(",")[6] == '*':
                word = node.surface
            else:
                word = node.feature.split(",")[6]

            # 品詞を取得
            part = node.feature.split(",")[0]
            if part in ["名詞", "形容詞", "動詞", "記号", "助詞"]:
                r.append(word)
            node = node.next
        return r

```

```python
dic_path = "/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd" # 辞書の場所
wakati = MecabTokenizer(dic_path)

s = "私は犬が好き。"
m = wakati.parse(s)
print(m)
#['私', 'は', '犬', 'が', '好き', '。']
```

きれいに分割できています！！

少し、補足しますと、MecabToknizerの中で仕事するmecabの出力は、以下の通りになっています。

```bash
BOS/EOS,*,*,*,*,*,*,*,*
名詞,代名詞,一般,*,*,*,私,ワタシ,ワタシ
助詞,係助詞,*,*,*,*,は,ハ,ワ
名詞,一般,*,*,*,*,犬,イヌ,イヌ
助詞,格助詞,一般,*,*,*,が,ガ,ガ
名詞,形容動詞語幹,*,*,*,*,好き,スキ,スキ
記号,句点,*,*,*,*,。,。,。
BOS/EOS,*,*,*,*,*,*,*,*
```
この情報があると、品詞を限定したりすることもできます。

# 参考文献

- [Google colabでBERTを使ってライブドアニュースコーパスを多クラス分類をする](https://tech.fusic.co.jp/posts/2021-04-23-bert-multi-classification/)

- [mecab + NEologd + python3 で形態素解析](https://qiita.com/sudo5in5k/items/f89d9dc1bec1ed221ede)