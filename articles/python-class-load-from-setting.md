---
title: "Python 設定ファイルで処理パイプラインを作成する方法" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["python", "機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

[MMDetection](https://github.com/open-mmlab/mmdetection)や[EasyCV](https://github.com/alibaba/EasyCV)では、データ処理のパイプラインを設定ファイルに書くことで、処理の内容、順序を定義することができます。

例えば、データ読み込み後の処理パイプラインは、以下のように書かれています。

```python: https://github.com/alibaba/EasyCV/blob/master/configs/detection/common/dataset/autoaug_coco_detection.py
train_pipeline = [
    dict(type='MMRandomFlip', flip_ratio=0.5),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='MMPad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'ori_img_shape',
                   'img_shape', 'pad_shape', 'scale_factor', 'flip',
                   'flip_direction', 'img_norm_cfg'))
]
```

実際、自前で、この処理を実装する場合の例を紹介します。

`Register`クラスを用いて作成された`SAMPLE`という処理辞書に、処理クラスを事前に登録するとします。

```python
from src.util.registry import Registry

# sampleという処理辞書を作成
SAMPLE = Registry("sample")

# sample処理辞書にSampleClsを登録
@SAMPLE.register_module()
class SampleCls():
    def __init__(self, a) -> None:
        self.a = a
    def forward(self):
        return self.a

print(SAMPLE.name) # sample
print(SAMPLE.module_dict) # {'SampleCls': <class '__main__.SampleCls'>}
```

この作成した処理辞書から、設定ファイルを用いて、`SampleCfg`を読み出し、インスタンスを作成します。

```python
# 設定辞書
cfg = {
        "type": "SampleCls",
        "a": "sample_str"
    }

# SAMPLEという処理辞書から設定辞書を用いて処理クラスを呼び出し、type以外の引数を用いて、インスタンスを作成
s = build_from_cfg(
    cfg,
    SAMPLE
)

print(s.forward()) # sample_str
```

# 処理辞書の作成と呼び出し

`Register`と`build_from_cfg`は、以下のようなコードになります。これは、[EasyCV]の[registry.py](https://github.com/alibaba/EasyCV/blob/b737027aa46b41ba9074d3c3cc11b22de06dd88a/easycv/utils/registry.py#L1)を参考にしています。

```python:src/util/registry.py
import inspect
from functools import partial

class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, force=False):
        """Register a module.
        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls=None, force=False):
        if cls is None:
            return partial(self.register_module, force=force)
        self._register_module(cls, force=force)
        return cls

def build_from_cfg(cfg, registry:Registry, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if obj_type in registry.module_dict:
        obj_cls = registry.get(obj_type)
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)
```

# ファイル分割して使用する場合

処理辞書を設定するファイルと処理クラスを実装するファイルを分割したいことが多々あります。その実装について解説します。

例えば、`src/util/repo.py`で処理辞書を作成します。

```python: src/util/repo.py
from src.util.registry import Registry
SAMPLE = Registry("sample")
```

次に、処理に使用するクラスの実装を登録します。

```python: src/sample/a.py
from src.util.repo import SAMPLE

@SAMPLE.register_module()
class SampleCls():
    def __init__(self, a) -> None:
        self.a = a
    def forward(self):
        return self.a
```

これを、`src/test.py`で使用するとします。

```python: src/test.py
from src.util.repo import SAMPLE
from src.util.registry import build_from_cfg

print(SAMPLE.name)
print(SAMPLE.module_dict)
s = build_from_cfg(
    {"type": "SampleCls"},
    SAMPLE,
    {"a": "sample_str"}
)

print(s.forward())
```

もし、`import`のパスが通らない場合は、[sys.path.append() を使わないでください](https://qiita.com/siida36/items/b171922546e65b868679)を参考にしてみてください。

さて、これで上手くいくかと思いますが、実際は、`SampleCls`が登録されていません。

しかし、

```python
from src.sample.a import SampleCls
```

のように呼び出しを追加するのは、処理クラスの数が増えた時、管理できなくなります。


そこで、`__init__.py`を作成します。

まず、`src/util/__init__.py`を作成します。

```python: src/util/__init__.py
from src.util.repo import SAMPLE

__all__ = ["SAMPLE"]
```

次に、`src/sample/__init__.py`を作成します。

```python: src/sample/__init__.py
from src.sample.a import SampleCls

__all__ = [
    "SampleCls"
]
```

最後に、`src/__init__.py`を作成します。

```python: src/__init__.py
from src import sample
from src import util
```

これで、`SampleCls`が登録されていると思います。

以上、設定ファイルから、処理のパイプラインを作成する方法でした。