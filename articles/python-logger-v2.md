---
title: "Pythonのロギング:loggingを用いたロガー作成方法" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["python"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
---

こんにちは、[わっしー](https://twitter.com/kwashizzz)です。
個人的にPythonでログを取る方法は、常に迷っています。簡単のため、余計なライブラリは使用せず、デフォルトで入っているloggingを使用したり、ファイルごとに分けたロガーを設定したいなどやりたいことはたくさんあります。この記事では、現在使用しているロギングの方法を紹介します。

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

以下の```get_logger```関数を各ファイルで呼び出して、使用します。その時、ロガーに名前を付けたい場合は、引数のnameに名前を与えます。この名前ごとに、```_log_initialized```辞書にロガーが格納されていく形式になっています。基本的には、分ける必要がないですが、ファイル出力を分けたいなど複雑な設定が必要になったときに、修正が楽です。

ロガーに関する設定は、簡単にしています。
もし、ファイル出力を行いたい場合は、```filename```引数にlogファイル名を設定してください。また、その際、ファイルサイズやバックアップ数も、```file_max_bytes```, ```file_backup_count```で設定できます。

この`logger`では、```INFO```と```DEBUG```を設定できるようにしています。デフォルトではINFOですが、```debug = True```でDEBUGモードに変更できます

また、基本的には、機械学習で使用していますが、アプリ内で使用する場合は、jsonフォーマットの機能などを追加し、Amazon CloudWatchで記録するのが楽かもしれません。
formatに関しては、適宜修正してください。

```python:utils/logger.py
import logging
import logging.handlers # handlersを使用するため呼び出し必須
from pathlib import Path
from typing import Dict, Optional



_log_initialized: Dict[str, logging.Logger] = {}


def get_logger(
    debug: bool = False,
    filename: Optional[str] = None,
    name: str = "main",
    add_stream_handler: bool = True,
    file_max_bytes=100000, 
    file_backup_count=1
) -> logging.Logger:
    """loggerを取得
    関数を読み込む前に実行
    Args:
        debug (bool): デバッグモードにするか?, Falseの場合、INFO
        filename (str, optional): ログのファイル出力先
        name (str, optional): ログの名前
        add_stream_handler (bool, optional): ストリーム出力
    Returns:
        logging.Logger: Logger instance.
    """
    global _log_initialized
    logger = _log_initialized.get(name, None)
    if logger is not None:
        return logger
    
    format = "'%(levelname)-8s: %(asctime)s | %(filename)-12s - %(funcName)-12s : %(lineno)-4s -- %(message)s', datefmt='%Y-%m-%d %H:%M:%S'"
    logger = logging.getLogger(name)
    
    #ログレベルの設定
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if add_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(format))
        logger.addHandler(stream_handler)
    
    # ログファイルに関する設定
    if filename is not None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename, 
            maxBytes=file_max_bytes, 
            backupCount=file_backup_count,
            mode="a+", # 開くか、新しいテキストファイルを作って最後から更新
            encoding="utf-8"
            )
        file_handler.setFormatter(logging.Formatter(format))
        logger.addHandler(file_handler)
        
    _log_initialized[name] = logger
    return logger

if __name__ == "__main__":
    logger = get_logger(debug=False)
    logger.info("info message")
    logger.debug("debug message")

```

# 使用例

上記のロガーを使用すると以下のような出力を得ることができます。
左から、ログレベル、日時、ファイルネーム、関数名、ライン番号、メッセージです。これだけあれば、ある程度場所の特定は、可能だと思います。

```bash
2021-05-05 16:41:20 (app.py:10)       - main         : 16   -- appログテスト
ERROR   : 2021-05-05 16:41:20 | app.py       - main         : 17   -- appエラー
INFO    : 2021-05-05 16:41:20 | module.py   - module       : 14   -- moduleログテスト
```

## 使用方法

以下に、```src/app.py```と```src/module.py```のコードを載せています。

基本的には、```utils```や```src```にパスを通す処理を行い、ロガーを生成しています。パスを通す記述が面倒に思えるかもしれません。これは、```setup.py```を作成し、```pip install -e .``` を行えば必要ないはずです。しかし、この記述を行っておくことで、脳死で、同じ階層や上の階層のライブラリを呼ぶことができるようになるため、個人的にかなり、重宝しています。

```python:src/app.py
import sys
import pathlib

# パスを通す (setup.pyを作成し、pip install -e . を行えば必要ないはず)
module_path = pathlib.Path(__file__).parent.parent.resolve()
if module_path not in sys.path:
    sys.path.append(str(module_path))

# 絶対にmoduleを呼び出す前にloggerの設定が必要
# 後に設定した場合、moduleの方のlogger設定になってしまう。
from utils import get_logger
logger = get_logger(
    debug=False,
    filename=".log/sample.log",
)


from src.modules import module

logger.info("info message")
logger.debug("debug message") # debugなしなので、出力されない
logger.error("error message") 


module()
```

```python:src/module.py
import sys
import pathlib

# パスを通す
module_path = pathlib.Path(__file__).parent.parent.resolve()
print(module_path)
if module_path not in sys.path:
    sys.path.append(str(module_path))

# もし、app.pyと同様のログを使用する場合は、ログレベルなどを気にしなくて良い
from utils import get_logger
logger = get_logger()

def module():
    logger.info("module log message")
```

以上が、Pythonのデフォルトであるloggingライブラリを使用したログの使用例です。loggingの設定は、ファイルに設定を書いたり、他にも設定を楽にする方法がたくさんあります。個人的に、現状の方法で重宝していますが、より良い方法を知っている方は、コメントいただければ嬉しいです！

----

画像・自然言語・音声に関する機械学習の研究開発やMLOpsを行っています。もし、機械学習に関して、ご相談があれば、[@kwashizzz](https://twitter.com/kwashizzz)のアカウントまでDMしてください！
これまでの、[機械学習記事のまとめ](https://zenn.dev/kwashizzz/articles/my-ml-articles-summary)です。