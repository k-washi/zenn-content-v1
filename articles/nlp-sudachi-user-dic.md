---
title: "GiNZAの形態素解析処理 Sudachiにユーザ定義辞書を追加する方法" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習", "nlp"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
---

GiNZAの形態素解析のライブラリとして、Sudachiが使用されています。そのため、GiNZAの解析にユーザ定義辞書を追加したい場合は、Sudachiにユーザ定義辞書を追加する必要があります。

ここでは、Sudachiにユーザ定義辞書を追加する方法を解説します。


[sudachi ユーザ辞書 公式のドキュメント](https://github.com/WorksApplications/Sudachi/blob/develop/docs/user_dict.md)

使用ライブラリー
```
spacy.__version__, ja_ginza.__version__, sudachipy.__version__
('3.2.2', '5.1.0', '0.6.3')
```

# 例

例えば、`ルイズ・フランソワーズは、ゼロの使い魔のヒロインです。` という文を形態素解析したとします。
デフォルトの辞書では、以下のように結果になります。

```s
0 ルイズ 名詞-固有名詞-人名-一般
1 ・ 補助記号-一般
2 フランソワーズ 名詞-固有名詞-人名-一般
3 は 助詞-係助詞
4 、 補助記号-読点
5 ゼロ 名詞-数詞
6 の 助詞-格助詞
7 使い魔 名詞-普通名詞-一般
8 の 助詞-格助詞
9 ヒロイン 名詞-普通名詞-一般
10 です 助動詞
11 。 補助記号-句点
```

ユーザ定義辞書に追加し、以下のように人物名や作品のタイトルを、固有名詞として認識したいということが、本記事の目的です。

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

# ユーザ辞書の読み込み

実際は、csvなどに記載しておき、pandasなどで読み込みます。
今回は、リスト形式で格納しています。

```python
user_dic_list = [
    ["名詞-固有名詞-人名-一般", "ルイズ・フランソワーズ"],
    ["名詞-固有名詞-一般", "ゼロの使い魔"],
]
```

# 正規化処理

形態素解析する際には、正規化された文字である必要があります。
以下、簡単な例ですが、実際には、より複雑な処理(スペース除去など)が必要になります。

```python
def normalize_text(text):
    text = text.replace("\n", "").replace("\r", "").replace("\t", " ")
    text = text.lower()
    return text
```

# Sudachi辞書の形式に形成

Sudachiの辞書形式である、以下の形に形成します。
```s
 ["見出し", "左連接ID", "右連接ID", "コスト", "見出し", "品詞1", "品詞2", "品詞3", "品詞4", 
           "品詞 (活用型)", "品詞 (活用形)", "読み", "正規化表記", "辞書形ID", "分割タイプ", "A単位分割情報", "B単位分割情報", "未使用"]
```

例に合わせると、以下になります。

```s
['ルイズ・フランソワーズ', 4788, 4788, -4091, 'ルイズ・フランソワーズ', '名詞', '固有名詞', '人名', '一般', '*', '*', 'ルイズ・フランソワーズ', 'ルイズ・フランソワーズ', '*', '*', '*', '*', '*']
```

以下のプログラムで`new_user_dic`に格納していきます。


```python
new_user_dic = []

for hinshi, hyouki in user_dic_list:
    # 本当は、より細かい正規化が必要
    hyouki = normalize_text(hyouki)

    cost = -5000 + int(10000 / len(hyouki)) # 名詞の場合5000 ~ 9000が推奨らしいが、かなり低めに設定
    hinshi, hinshi_t1, hinshi_t2, hinshi_t3, rensa_id = hinshi_format(hinshi)
    
    sudashi_form =  [hyouki, rensa_id, rensa_id, cost, hyouki, hinshi, hinshi_t1, hinshi_t2, hinshi_t3,"*","*", hyouki, hyouki,"*","*","*","*","*"]
    new_user_dic.append(sudashi_form)
print(new_user_dic[0])
print(new_user_dic[-1])
```

コストは、名詞の場合5000 ~ 9000が推奨らしいですが、基本を-5000とかなり低く設定し、文字数が長いほど低くなるようにもしています。
品詞に関しては、以下のように品詞ごとに、品詞1, 品詞2や連接IDを割り当てています。

```python
def hinshi_format(hinshi):
    if hinshi == "名詞-固有名詞-人名-一般":
        hinshi = "名詞"
        hinshi_t1 = "固有名詞"
        hinshi_t2 = "人名"
        hinshi_t3 = "一般"
        rensa_id = 4788
    elif hinshi == "名詞-一般":
        hinshi = "名詞"
        hinshi_t1 = "普通名詞"
        hinshi_t2 = "一般"
        hinshi_t3 = "*"
        rensa_id = 5146
    elif hinshi == "名詞-サ変接続":
        hinshi = "名詞"
        hinshi_t1 = "普通名詞"
        hinshi_t2 = "サ変接続"
        hinshi_t3 = "*"
        rensa_id = 5133
    elif hinshi == "名詞-固有名詞-一般":
        hinshi = "名詞"
        hinshi_t1 = "固有名詞"
        hinshi_t2 = "一般"
        hinshi_t3 = "*"
        rensa_id = 4786
    elif hinshi =="名詞-固有名詞-地名-一般":
        hinshi = "名詞"
        hinshi_t1 = "固有名詞"
        hinshi_t2 = "地名"
        hinshi_t3 = "一般"
        rensa_id = 4792
    elif hinshi == "名詞-固有名詞-人名-姓":
        hinshi = "名詞"
        hinshi_t1 = "固有名詞"
        hinshi_t2 = "人名"
        hinshi_t3 = "姓"
        rensa_id = 4790
    elif hinshi == "名詞-固有名詞-人名-名":
        hinshi = "名詞"
        hinshi_t1 = "固有名詞"
        hinshi_t2 = "人名"
        hinshi_t3 = "名"
        rensa_id = 4789
    elif hinshi == "形容詞":
        # IPAでは、形容動詞は名詞の形容動詞語幹として含まれ、 形容詞には含まれない
        # sudachiでは、Juman => 形容詞
        hinshi = "形容詞"
        hinshi_t1 = "一般"
        hinshi_t2 = "*"
        hinshi_t3 = "*"
        rensa_id = 5161
    else:
        print(hinshi)
        raise ValueError(f"ルールにない品詞が含まれいます。{hinshi}")
    return hinshi, hinshi_t1, hinshi_t2, hinshi_t3, rensa_id
```

# ユーザ定義辞書を保存し、sudachi辞書にビルド

以下の処理で、`.sudachi`ディレクトリを作成し、`user_dic.csv`としてユーザ定義辞書を保存します。

```python

import os
import pandas as pd

# .sudachi ディレクトリ作成
cwd = os.getcwd()
print(cwd)
dic_dir = f"{cwd}/.sudachi"
os.makedirs(dic_dir, exist_ok=True)

# 以下のcolumnsで.sudachiディレクトリにユーザ定義辞書(user_dic.csv)を保存する
columns = ["見出し", "左連接ID", "右連接ID", "コスト", "見出し", "品詞1", "品詞2", "品詞3", "品詞4", 
           "品詞 (活用型)", "品詞 (活用形)", "読み", "正規化表記", "辞書形ID", "分割タイプ", "A単位分割情報", "B単位分割情報", "未使用"]

df = pd.DataFrame(new_user_dic, columns=columns)
df.head()
df.to_csv(f"{dic_dir}/user_dic.csv", header=None, index=False)
```

保存したユーザ定義辞書をsudachi辞書にビルドします。

```python
import sys

site_package = ""
for p in sys.path:
    if "site-packages" in os.path.basename(p):
        site_package = p
print(site_package)
# /root/.pyenv/versions/3.9.7/lib/python3.9/site-packages

# jupyter上で実行したとします。
# user dicの作成 (.sudachi/user.dicが作成される)
!sudachipy ubuild -s '{site_package}/sudachidict_core/resources/system.dic' ./.sudachi/user_dic.csv -o ./.sudachi/user.dic
```

# sudachiのデフォルトの設定ファイルを書き換える

sudachiは、実行時に設定ファイルが読み込まれており、それは、以下のパスに格納されています。

```s
{site_package}/sudachipy/resources/sudachi.json

# 私の環境の場合
/root/.pyenv/versions/3.9.7/lib/python3.9/site-packages/sudachipy/resources/sudachi.json
```

この設定ファイルは、以下のようになっています。この設定ファイルに、ユーザ定義辞書のパスを追加します。


```s
{
    "systemDict" : null,
    "characterDefinitionFile" : "char.def",
    "inputTextPlugin" : [
        { "class" : "com.worksap.nlp.sudachi.DefaultInputTextPlugin" },
        { "class" : "com.worksap.nlp.sudachi.ProlongedSoundMarkPlugin",
          "prolongedSoundMarks": ["ー", "-", "⁓", "〜", "〰"],
          "replacementSymbol": "ー"},
	    { "class": "com.worksap.nlp.sudachi.IgnoreYomiganaPlugin",
          "leftBrackets": ["(", "（"],
          "rightBrackets": [")", "）"],
          "maxYomiganaLength": 4}
    ],
    "oovProviderPlugin" : [
        { "class" : "com.worksap.nlp.sudachi.MeCabOovPlugin",
          "charDef" : "char.def",
    ...
```

私の環境の場合、 `/workspace/.sudachi/user.dic`なので、以下のようになります。

> `"userDict" : ["/workspace/.sudachi/user.dic"],` を追加

```s
{
    "systemDict" : null,
    "characterDefinitionFile" : "char.def",
    "userDict" : ["/workspace/.sudachi/user.dic"],
    "inputTextPlugin" : [
        { "class" : "com.worksap.nlp.sudachi.DefaultInputTextPlugin" },
        { "class" : "com.worksap.nlp.sudachi.ProlongedSoundMarkPlugin",
          "prolongedSoundMarks": ["ー", "-", "⁓", "〜", "〰"],
          "replacementSymbol": "ー"},
	    { "class": "com.worksap.nlp.sudachi.IgnoreYomiganaPlugin",
          "leftBrackets": ["(", "（"],
          "rightBrackets": [")", "）"],
          "maxYomiganaLength": 4}
    ],
    "oovProviderPlugin" : [
        { "class" : "com.worksap.nlp.sudachi.MeCabOovPlugin",
          "charDef" : "char.def",
```

以上、ユーザ定義辞書の追加方法でした。

# 形態素解析

あとは、以下のように形態素解析すれば良いです。

```python
import spacy
import ginza

nlp = spacy.load('ja_ginza')
ginza.set_split_mode(nlp, "A")

text = "ルイズ・フランソワーズは、ゼロの使い魔のヒロインです。"
text = normalize_text(text)
doc = nlp(text)

for sent in doc.sents:
    for token in sent:
        print(token.i, token.orth_, token.tag_)
```
