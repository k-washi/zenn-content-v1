---
title: "Hydraを用いたPython・機械学習のパラメータ設定方法" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["python", "機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

Pythonで設定ファイルを書く方法は、多くあります。以前は、.envファイルに書いたりしていましたが、最近は、Hydraというライブラリを用いてパラメータを管理しています。
実際、機械学習のプロダクトで、Hydraを使用して、実験から、プロダクトの管理までを行っています。本記事では、Hydraの簡単な使用方法を解説します！

# インストール

```
pip install hydra-core==1.0.4
```

# ファイル構成

設定ファイルには、読み込みたいパラメータを書いています。この設定パラメータを```config.py```に記載しているクラスで読み込みます。このライブラリを読み込んで実際に実行されるのは、src配下の実行ファイルです。

```yaml
# 設定ファイル
-conf 
 |- default.yaml
 |-test_param
    |-default.yaml
    |-test.yaml

# 実行ファイル
-src
 |-hydra_test.py

# 設定を読み込みライブラリ
-utils
 |-config.py

```

# 設定ファイル

今回は、以下のような辞書を返すような設定にしたいと思います。
```
{'param': 'paramです！', 'test_param': {'param1': 0, 'param2': 'test0'}}
```

まず、conf配下の```default.yaml```です。

```yaml:default.yaml
param: "paramです！"
defaults:
  - test_param: default
```

これは、test_paramディレクトリの```default.yaml```を読み込む設定も書かれています。なので、ディレクトリを分けてパラメータを記載することができます。今回は、以下のように記載しています。

```yaml: test_param/default.yaml
# @package _group_

param1: 0
param2: "test0"
```

ここで、重要なのは、ファイルの開始時に、```@package _group_```のように書くことです。

## 設定ファイルの変更

もし、機械学習の実験などで、パラメータを変更したいとします。このような時は、```test_param/test.yaml```など、別のファイルを読み込めば良いです。

設定は簡単で、以下のように、```conf/default.yaml```を変更するだけです。

```yaml:default.yaml
param: "paramです！"
defaults:
  - test_param: test
```

例えば、```test_param/test.yaml```を、以下のように記載しているとします。

```yaml:test_param/test.yaml
# @package _group_

param1: 1
param2: "test1
```

そうした時、以下のように、異なる設定を読み込めます。
```
{'param': 'paramです！', 'test_param': {'param1': 1, 'param2': 'test1'}}
```

# 設定を読み込むファイル

次に実際にhydraを用いてパラメータを読み込む、```utils/config.py```について説明します。
プログラムを見た方が早いと思います。このファイルで、conf配下の設定を読み込みます。

```python: utils/config.py
import sys
import os
from hydra.experimental import compose, initialize_config_dir


class Config():
    """
    hydraによる設定値の取得 (conf)
    """
    @staticmethod
    def get_cnf(conf_path: str):
        """
        設定値の辞書を取得
        @param
            conf_path: str
        @return
            cnf: OmegaDict
        """
        conf_dir = os.path.join(os.getcwd(), "conf")
        if not os.path.isdir(conf_dir):
            print(f"Can not find file: {conf_dir}.")
            sys.exit(-1)

        with initialize_config_dir(config_dir=conf_dir):
            cnf = compose(config_name="default.yaml")
            return cnf
```

# 実行ファイル

上記の設定を読み込む関数を呼ぶ、実行ファイルに関して説明します。

まず、実行ファイルから```utils```へのパスを通す処理を行っています。ラリブラリの読み込みで迷ったら、とりあえず書いておくのが良いと思います。

次に、Configクラスを読み込んで、実行しています。```get_conf```関数は、```staticmethod```としているので、直接実行可能です。

実際、読み込んだパラメータの結果もプログラム内に記載しています。```param.test_param```で、```test_param```配下のパラメータにもアクセスできています。

```python:src/hydra_test.py
import sys
import pathlib

# utilsのパスを通す
module_path = pathlib.Path(__file__, "..", "..").resolve()
if module_path not in sys.path:
    sys.path.append(str(module_path))

# 設定を読み込む
from utils.config import Config
param = Config.get_cnf("./conf")

print(param)
#{'param': 'paramです！', 'test_param': {'param1': 0, 'param2': 'test0'}}
print(param.test_param)
#{'param1': 0, 'param2': 'test0'}
```

とても、便利なので是非使ってみてください！