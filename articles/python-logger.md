---
title: "Pythonのログング:loggingを用いたロガー作成方法" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["python"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
---


個人的にPythonでログを取る方法は、常に迷っています。簡単のため、余計なライブラリは使用せず、デフォルトで入っているloggingを使用したり、ファイルごとに分けたロガーを設定したいなどやりたいことはたくさんあります。

今回は、直近、1年くらい使っているロガーの紹介をしたいと思います。

# ファイル構成

今回は、以下のような構成にし、```src/app.py```, ```src/module.py```にて、```utils/logger.py```からログの設定を読み出して使用することを想定します。

```
-src
 |- app.py
 |- module.py

-utils
 |-logger.py

```

# ロガー生成関数

以下の```get_logger```関数を各ファイルで呼び出して、使用します。その時、ロガーに名前を付けたい場合は、引数のnameに名前を与えます。この名前ごとに、```loggers```辞書にロガーが格納されていく形式になっています。基本的には、分ける必要がないですが、ファイル出力を分けたいなど複雑な設定が必要になったときに、修正が楽です。

ロガーに関する設定は、簡単にしています。もし、ファイルに出力したい場合は、```FileHandler```を使用すれば良いですし、ログレベルを設定したい場合は、環境変数からレベルを読み込み、場合わけで設定すれば良いです。

また、基本的には、機械学習で使用していますが、アプリ内で使用する場合は、jsonフォーマットにして、Amazon CloudWatchで記録すれば、なんとかなる気がしています！

```python:utils/logger.py
import logging

loggers = {}

def get_logger(name=None):
    global loggers
    if name is None:
        # デフォルトの名前
        name = "library-name"

    if loggers.get(name):
        # すでにnameのロガーが格納されているなら、格納されているロガーを返す
        return loggers.get(name)

    # 新しいロガーを作成
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(levelname)-8s: %(asctime)s | %(filename)-12s - %(funcName)-12s : %(lineno)-4s -- %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    loggers[name] = logger

    return logger
```

# 使用例

上記のロガーを使用すると以下のような出力を得ることができます。
左から、ログレベル、日時、ファイルネーム、関数名、ライン番号、メッセージです。これだけあれば、ある程度場所の特定は、可能だと思います。

```bash
INFO    : 2021-05-05 16:41:20 | app.py       - main         : 16   -- appログテスト
ERROR   : 2021-05-05 16:41:20 | app.py       - main         : 17   -- appエラー
INFO    : 2021-05-05 16:41:20 | module.py   - module       : 14   -- moduleログテスト
```

## 使用方法

以下に、```src/app.py```と```src/module.py```のコードを載せています。基本的には、```utils```や```src```にパスを通す処理を行い、ロガーを生成しています。ロガーを生成するために、パスを通す処理をしており、この記述が面倒に思えるかもしれません。しかし、この記述は、脳死で、同じ階層や上の階層のライブラリを呼ぶことができるようになるため、個人的にかなり、重宝しています。

```python:src/app.py
import sys
import pathlib

# パスを通す
module_path = pathlib.Path(__file__).parent.parent.resolve()
if module_path not in sys.path:
    sys.path.append(str(module_path))

from utils import get_logger
logger = get_logger()

from src.modules import module

def main():
    logger.info("appログテスト")
    logger.error("appエラー")

    module()

if __name__ == "__main__":
    main()
```

```python:src/module.py
import sys
import pathlib

# パスを通す
module_path = pathlib.Path(__file__).parent.parent.resolve()
print(module_path)
if module_path not in sys.path:
    sys.path.append(str(module_path))

from utils import get_logger
logger = get_logger("モジュール")

def module():
    logger.info("moduleログテスト")
```

以上が、Pythonのデフォルトであるloggingライブラリを使用したログの使用例です。loggingの設定は、ファイルに設定を書いたり、他にも設定を楽にする方法がたくさんあります。個人的に、現状の方法で重宝していますが、より良い方法を知っている方は、コメントいただければ嬉しいです！

