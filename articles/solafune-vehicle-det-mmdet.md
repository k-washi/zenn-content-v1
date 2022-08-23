---
title: "Solafuneのマルチ解像度画像の車両検出コンペ - MMDetectionでやってみた" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["python", "ml"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
---

こんにちは！鷲崎([@kwashizzz](https://twitter.com/kwashizzz))です。今回は、[Solafune](https://solafune.com/)という衛星データ解析コンテストプラットフォームのマルチ解像度画像の車両検出のコンペティションに挑戦してみました。本記事では、データのダウンロードから、結果の提出まで解説しています。物体検出には[MMDetection](https://github.com/open-mmlab/mmdetection)を使用し、今回のコンペの評価指標である[OC-cost(Optimal Correction Cost)](https://arxiv.org/abs/2203.14438)の導入方法も解説しています。

OC-Costに関しては、日本語の[この記事](https://cyberagent.ai/blog/research/computervision/16366/)がわかりやすかったです。

なんと8月22日時点で1位(スコア:0.3090)です！提出しているのが一人なので、当たり前ですが...  
この記事でコンペに興味を持ってもらい、どんどんコンペの参加者が増え、接戦になれば嬉しいです。

>cf. @solafune (https://solafune.com)コンテストの参加以外の目的とした利用及び商用利用は禁止されています。商用利用・その他当コンテスト以外で利用したい場合はお問い合わせください。(https://solafune.com)

# 実装の準備

必要なライブラリのインストール方法を書いています。環境によって異なると思いますが、参考にしてください。

`pip install -r ./requirements.txt` で必要なライブラリをインストール

```txt:requirements.txt
tensorboard>=2.8.0
tqdm>=4.62.3
numpy>=1.22.2
Pillow>=9.0.1
pytest>=7.0.1
PuLP==2.6.0
```

### MMDetectionのインストール

物体検出ライブラリ[MMDetection](https://github.com/open-mmlab/mmdetection)を以下の手順でインストールします。
とりあえず、最新っぽいバージョンをインストールしています。

```sh
# install mmdetection
pip install -U openmim==0.2.1
mim install mmcv-full==1.6.1

git clone https://github.com/open-mmlab/mmdetection.git -b v2.25.1
cd mmdetection
pip install -v -e .

# verify install
mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
```

### OC-Costのインストール

Solafuneさんが準備された?[OC-Cost](https://github.com/Solafune-Inc/OC-cost)をインスールします。

```
git clone git@github.com:Solafune-Inc/OC-cost.git
cd ./OC-cost
pip install -e .
```



# データセットの前準備

ここでは、データセットのダウンロードと、MMDetectionで使用するための前準備について解説します。

まず、データセットのダウンロードです。今後、データセットは`./data/vdet`配下にあるとします。コンペホームページより、`train.json`, `train_images.zip`, `evaluation_images.zip`をダウンロードしてください。そして、以下のように解凍します。

```sh
unzip -o ./data/vdet/evaluation_images.zip -d ./data/vdet/
unzip -o ./data/vdet/train_images.zip -d ./data/vdet/
```

次に、MMDetectionで使用するため、

1. tif形式の画像をpngに変換する
2. アノテーションを訓練、評価用に分割
3. coco形式にアノテーションを変換

を行います。

まずは、tif形式の画像をpngに変換する処理です。

```python:./src/dataset/preprocess.py
from typing import Union
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from tifffile import TiffFile
import numpy as np

def load_tiff(fp:Union[str, Path]) -> np.ndarray:
    with TiffFile(str(fp)) as tif:
        if len(tif.pages) > 1:
            raise ValueError(f"{fp} pages: {tif.pages} > 1")
        return tif.asarray()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    
    fp = Path("./data/vdet/evaluation_images/evaluation_0.tif")
    if not fp.exists():
        raise FileExistsError(f"{fp}は、存在しません。script/dataset/solafune-vdet-download.sh を実行し、データを準備してください。")
    tif_array = load_tiff(fp)
    print(f"image shape: {tif_array.shape}, min: {tif_array.min()}, max: {tif_array.max()}, size: {tif_array.__sizeof__() / (1024*1024)} MB")
    plt.imshow(tif_array)
    
    os.makedirs("./data/tmp/", exist_ok=True)
    plt.savefig("./data/tmp/tiff_load.png")

def tiff_to_png(tiff_path:Union[str, Path], output_dir:Union[str, Path], save=True):
    """tiffファイルを読み込み、output_dirにpngとして出力
    """
    tiff_path = Path(tiff_path)
    tiff_arr = load_tiff(tiff_path)
    if save:
        output_dir = Path(output_dir)
        output_path = output_dir / f"{tiff_path.stem}.png"
        
        img = Image.fromarray(tiff_arr)
        img.save(output_path)
    return tiff_arr
   

def convert_png_from_tiff_dataset(dataset_dir: Union[str, Path], output_dir:Union[str, Path]):
    """tiffデータセットをpngの画像に変換
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise ValueError(f"{dataset_dir}はディレクトリではありません。")
    
    # 出力ディレクトリの作成
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fplist = sorted(list(dataset_dir.glob("*")))
    for fp in tqdm(fplist):
        tiff_to_png(fp, output_dir)

def __train_preprocess(dataset_dir, output_dir):
    convert_png_from_tiff_dataset(dataset_dir, output_dir)
    
def __test_preprocess(dataset_dir, output_dir):
    convert_png_from_tiff_dataset(dataset_dir, output_dir)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--train_input", default="./data/vdet/train_images")
    parser.add_argument("-o1", "--train_output", default="./data/vdet/trainset")

    parser.add_argument("-i2", "--test_input", default="./data/vdet/evaluation_images")
    parser.add_argument("-o2", "--test_output", default="./data/vdet/testset")
    
    args = parser.parse_args()
    
    __train_preprocess(args.train_input, args.train_output)
    __test_preprocess(args.test_input, args.test_output)
    
    
```

次にアノテーション`train.json`をtrain, evalに分割し、coco形式に変換する処理です。cocoのフォーマットに関しては、[MS COCO datasetのフォーマットまとめ](https://qiita.com/kHz/items/8c06d0cb620f268f4b3e)の記事を参考にしました。引数の`eval_rate`でevalをどの程度にするか変更してください。

```python:./src/dataset/ano_json_to_coco.py
import json
from typing import Union
from pathlib import Path
from PIL import Image
import collections as cl
from operator import itemgetter
from sklearn.model_selection import train_test_split

__CAR_CATEGORY_ID = 1
__TMP_LICENCE_ID = 0

import collections as cl
from dataclasses import dataclass

@dataclass
class COCOKEY:
    info:str = "info"
    license:str = "licenses"
    images:str = "images"
    annotations:str = "annotations"
    categories:str = "categories"

@dataclass
class CocoBBox():
    x: int
    y: int
    width: int
    height: int
    
    def cocoformat(self):
        return [self.x, self.y, self.width, self.height]

def coco_info():
    t = cl.OrderedDict()
    t["description"] = "multi resolution veacle detection dataset"
    t["url"] = "https://solafune.com/competitions/25012781-b1e8-499e-9c8c-1f9b284d483e"
    t["version"] = "0.01"
    t["year"] = 2022
    t["contributor"] = "k-washi"
    t["data_created"] = "2022/08/19"
    return t

def coco_licenses(index):
  tmp = cl.OrderedDict()
  tmp["id"] = index
  tmp["url"] = ""
  tmp["name"] = ""
  return [tmp]

def coco_image(index, file_name, width, height, licene_id):
    tmp = cl.OrderedDict()
    tmp["license"] = licene_id
    tmp["id"] = index
    tmp["file_name"] = file_name
    tmp["width"] = width
    tmp["height"] = height
    tmp["date_captured"] = "2022-08-01 12:34:56"
    tmp["coco_url"] = ""
    tmp["flickr_url"] = ""
    return tmp

def coco_anotation(index:int, image_index:int, category_id:int, bbox:CocoBBox):
    
    tmp = cl.OrderedDict()
    tmp["segmentation"] = []
    tmp["area"] = 10
    tmp["image_id"] = image_index
    tmp["iscrowd"] = 0
    tmp["bbox"] = bbox.cocoformat()
    tmp["id"] = index
    tmp["category_id"] = category_id
    tmp["caption"] = ""
    return tmp

def coco_categories_car_only(category_id):
    # assert category_id > 0, f"カテゴリID{category_id} > 0"
    tmps = []
    
    tmp = cl.OrderedDict()
    tmp["id"] = category_id
    tmp["supercategory"] = "car"
    tmp["name"] = "car"
    tmps.append(tmp)
    return tmps

def anolist_to_cocoformt(image_list:list, dataset_dir:Union[str, Path], start_image_uid:int=0, start_ano_uid:int=0, scale:int=1):
    image_uid = start_image_uid
    ano_uid = start_ano_uid
    dataset_dir = Path(dataset_dir)
    
    js = cl.OrderedDict()
    js[COCOKEY.info] = coco_info()
    js[COCOKEY.license] = coco_licenses(__TMP_LICENCE_ID)
    js[COCOKEY.images] = []
    js[COCOKEY.annotations] = []
    js[COCOKEY.categories] = coco_categories_car_only(__CAR_CATEGORY_ID)
    
    for imdata in image_list:
        image_uid += 1
        file_name = imdata["name"].replace("tif", "png").replace("tiff", "png")
        img_path = dataset_dir / file_name
        _img = Image.open(str(img_path))
        imw, imh = _img.size
        ano_list = imdata["annotation"]
        
        js[COCOKEY.images].append(
            coco_image(image_uid, file_name, imw, imh, __TMP_LICENCE_ID)
        )
        
        for ano in ano_list:
            ano_uid += 1
            bbox = CocoBBox(
                x=int(ano["lefttop_x"] * scale),
                y=int(ano["lefttop_y"] * scale),
                width=int((ano["rightbottom_x"] - ano["lefttop_x"]) * scale),
                height=int((ano["rightbottom_y"] - ano["lefttop_y"]) * scale)
            )
            js[COCOKEY.annotations].append(
                coco_anotation(ano_uid, image_uid,  __CAR_CATEGORY_ID, bbox)
            )

    return js, image_uid, ano_uid

def convert_ano_to_json(train_ano_json_path:Union[str, Path], output_dir:Union[str, Path], dataset_dir:Union[str, Path], eval_rate:float=0.0, scale:int=1, random_seed:int=3407):
    assert (1 > eval_rate >= 0), f"eval rate:{eval_rate}を0以上から1未満に設定してください。"
    assert Path(dataset_dir).is_dir(), f"データセッt:{dataset_dir}はディレクトリではありません。"
    assert scale in (1, 2, 4, 6, 8), f"画像Scale:{scale}は、(1, 2, 4, 6, 8)のいずれかで設定してください"
    with open(train_ano_json_path, "r") as f:
        ano_data = json.load(f)
    
    image_list = ano_data.get("images")
    assert image_list is not None, f"{train_ano_json_path}からアノテーションデータを読み込めていません。"

    print(f"Dataset num: {len(image_list)}")
    
    # datasetを取り出しtrainとevalに分割
    image_indexes = list(range(0, len(image_list)))
    if eval_rate == 0:
        train_image_indexes, eval_image_indexes = image_indexes, []
    else:
        train_image_indexes, eval_image_indexes = train_test_split(image_indexes, test_size=eval_rate, random_state=random_seed, shuffle=True)
    print(f"train: {len(train_image_indexes)}, eval: {len(eval_image_indexes)}")
    train_image_list = itemgetter(*train_image_indexes)(image_list)
    eval_image_list = [] if len(eval_image_indexes) == 0 else itemgetter(*eval_image_indexes)(image_list)

    # cocoformatに修正
    output_dir = Path(output_dir)
    train_coco, image_uid, ano_uid = anolist_to_cocoformt(train_image_list, dataset_dir, start_image_uid=0, start_ano_uid=0)
    with open(output_dir / f"train_coco_s{scale}.json", "w", encoding='utf-8') as f:
        json.dump(train_coco, f, ensure_ascii=False, indent=2)

    eval_coco, image_uid, ano_uid = anolist_to_cocoformt(eval_image_list, dataset_dir, start_image_uid=image_uid, start_ano_uid=ano_uid)
    with open(output_dir / f"eval_coco_s{scale}.json", "w", encoding="utf-8") as f:
        json.dump(eval_coco, f, ensure_ascii=False, indent=2)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_ano_json_path", default="./data/vdet/train.json")
    parser.add_argument("-o", "--output_coco_dir", default="./data/vdet/")
    parser.add_argument("-r", "--dataset_dir", default="./data/vdet/trainset")
    parser.add_argument("-s", "--scale", default=1)
    parser.add_argument("-er", "--eval_rate", default=0.05)
    parser.add_argument("-rs", "--random_seed", default=3407)
    
    args = parser.parse_args()
    
    convert_ano_to_json(
        args.train_ano_json_path, 
        args.output_coco_dir, 
        args.dataset_dir, 
        eval_rate=float(args.eval_rate), 
        scale=int(args.scale), 
        random_seed=int(args.random_seed)
    )
```

# MMDetectionによる物体検出の学習 with OC-cost

ここでは、YOLOXを用いた車両の物体検出の学習について解説します。
基本的には、MMDetectionに合わせた設定ファイルの作成と、OC-Costの計算の追加となります。

まずは、YOLOX-lの重みファイルをダウンロードします。

```
wget https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth -P ./checkpoints
```

次に、`./mmdetection`から必要な設定ファイルをコピーします。

```
cp -r ./mmdetection/configs/_base_ ./src/mmdet
```

YOLOX-lの設定ファイルを`./src/mmdet/yolox/yolox_l_8x8_300e_coco_occost.py`として置くとします。基本的には、`mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py`を参考に修正しています。

設定ファイルは以下になります。ほとんど変更ないですが、OC-Costの計算のため、`custom_imports`として`CocoDataset`を拡張した`CocoOtcDataset`を追加しています。パラメータとして、`evaluation`にOC-Costのパラメータ(`otc_params`)と計算に使用する予測BBoxのスコアの閾値`score_th`を設定しています。今回はクラス分類がないため、`alpha`(論文の$\lambda$に相当)は、$1$で良いと思います。

```python:./src/mmdet/yolox/yolox_l_8x8_300e_coco_occost.py
_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'src.mmdet.occost.coco', 
    ],
    allow_failed_imports=False
)

img_scale = (640, 640)  # height, width
load_from = 'checkpoints/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(10, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3),
    bbox_head=dict(
        type='YOLOXHead', num_classes=1, in_channels=256, feat_channels=256),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
data_root = './data/vdet/'
dataset_type = 'CocoOtcDataset'
scale=1

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        max_rotate_degree=30,
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', direction=['horizontal', 'vertical'], flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

classes = ('car', "empty")

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + f'train_coco_s{scale}.json',
        img_prefix=data_root + 'trainset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]


test_dataset = dict(
    type=dataset_type,
    classes=classes,
    ann_file=data_root + f'eval_coco_s{scale}.json',
    img_prefix=data_root + 'trainset/',
    pipeline=test_pipeline
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=test_dataset,
    test=test_dataset
)
# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr= 1e-2/64,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

max_epochs = 300
num_last_epochs = 15
resume_from = None
interval = 3

# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    otc_params=[("alpha", 1.0), ("beta", 0.5)],
    score_th=0.6,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='bbox')

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)

```

[Optimal Correction Cost for Object Detection Evaluation](https://github.com/mayu-ot/oc-cost)を参考にし、`CocoOtcDataset`を`src/mmdet/occost/coco.py`に実装しています。

```python:src/mmdet/occost/coco.py
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.builder import DATASETS
import numpy as np
from oc_cost import OC_Cost
from oc_cost.Annotations import Annotations, predBBox

import time


def bboxes_to_anos(bboxes, image_id, score_th=0.5):
    bboxs = list()
    for b in bboxes:
        if b[4] < score_th:
            continue
        ins = predBBox(
            0, (b[0], b[1]), b[2] - b[0], b[3] - b[1], b[4]
        )
        bboxs.append(ins)
        
    return Annotations(image_id, bboxs) 

def eval_ot_costs(gts, results, occost:OC_Cost, beta, score_th=0.5):
    occost_list = []
    for i, (gt, res) in enumerate(zip(gts, results)):
        gt, res  = gt[0], res[0]
        gt_anos = bboxes_to_anos(gt, i)
        res_anos = bboxes_to_anos(res, i)
        
        c_matrix = occost.build_C_matrix(gt_anos, res_anos)
        pi_tilde_matrix = occost.optim(float(beta))
        cost = np.sum(np.multiply(pi_tilde_matrix, occost.opt.cost))
        occost_list.append(cost)
    return occost_list


@DATASETS.register_module()
class CocoOtcDataset(CocoDataset):
    def __init__(
        self,
        ann_file,
        pipeline,
        classes=None,
        data_root=None,
        img_prefix="",
        seg_prefix=None,
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=True,
    ):

        super().__init__(
            ann_file,
            pipeline,
            classes=classes,
            data_root=data_root,
            img_prefix=img_prefix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
        )

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        classwise=False,
        proposal_nums=(100, 300, 1000),
        iou_thrs=None,
        metric_items=None,
        eval_map=True,
        otc_params=[("alpha", 1), ("beta", 0.5)],
        score_th=0.5
    ):
        """Evaluate predicted bboxes. Overide this method for your measure.

        Args:
            results ([type]): outputs of a detector
            metric (str, optional): [description]. Defaults to "bbox".
            logger ([type], optional): [description]. Defaults to None.
            jsonfile_prefix ([type], optional): [description]. Defaults to None.
            classwise (bool, optional): [description]. Defaults to False.
            proposal_nums (tuple, optional): [description]. Defaults to (100, 300, 1000).
            iou_thrs ([type], optional): [description]. Defaults to None.
            metric_items ([type], optional): [description]. Defaults to None.
            eval_map (bool): Whether to evaluating mAP
            otc_params (list): OC-cost parameters.
                                alpha (lambda in the paper): balancing localization and classification costs.
                                beta: cost of extra / missing detections.
                                Defaults to [("alpha", 0.5), ("beta", 0.6)]

        Returns:
            dict[str, float]: {metric_name: metric_value}
        """
        if eval_map:
            eval_results = super().evaluate(
                results,
                metric=metric,
                logger=logger,
                jsonfile_prefix=jsonfile_prefix,
                classwise=classwise,
                proposal_nums=proposal_nums,
                iou_thrs=iou_thrs,
                metric_items=metric_items,
            )
        else:
            eval_results = {}

        otc_params = {k: v for k, v in otc_params}
        self.occost = OC_Cost(float(otc_params["alpha"]), False)
        
        mean_otc = self.eval_OTC(results, **otc_params, score_th=score_th)
        eval_results["mOTC"] = mean_otc

        return eval_results

    def get_gts(self):
        gts = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            gt = self._ann2detformat(ann_info)
            if gt is None:
                gt = [
                    np.asarray([]).reshape(0, 5)
                    for _ in range(len(self.CLASSES))
                ]
            gts.append(gt)
        return gts

    def eval_OTC(
        self,
        results,
        alpha=1.0,
        beta=0.5,
        score_th=0.5,
        get_average=True,
    ):
        gts = self.get_gts()
        tic = time.time()
        ot_costs = eval_ot_costs(gts, results, self.occost, beta, score_th)
        toc = time.time()
        print("OTC DONE (t={:0.2f}s).".format(toc - tic))

        if get_average:
            mean_ot_costs = np.mean(ot_costs)
            return mean_ot_costs
        else:
            return ot_costs

    def evaluate_gt(
        self,
        bbox_noise_level=None,
        **kwargs,
    ):

        gts = self.get_gts()
        n = len(gts)
        for i in range(n):
            gt = gts[i]
            if gt is None:
                gts[i] = [np.asarray([]).reshape(0, 5) for _ in self.CLASSES]
                continue
            for bbox in gt:
                if len(bbox) == 0:
                    continue

                w = bbox[:, 2] - bbox[:, 0]
                h = bbox[:, 3] - bbox[:, 1]
                shift_x = (
                    w * bbox_noise_level * np.random.choice((-1, 1), w.shape)
                )
                shift_y = (
                    h * bbox_noise_level * np.random.choice((-1, 1), h.shape)
                )
                bbox[:, 0] += shift_x
                bbox[:, 2] += shift_x
                bbox[:, 1] += shift_y
                bbox[:, 3] += shift_y
        return self.evaluate(gts, **kwargs)

    def _ann2detformat(self, ann_info):
        """convert annotation info of CocoDataset into detection output format.

        Parameters
        ----------
        ann : list[dict]
            ground truth annotation. each item in the list correnponds to an instance.
            >>> ann_info[i].keys()
            dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'])
        Returns
        -------
        bboxes : list[numpy]
            list of bounding boxes with confidence score.
            bboxes[i] contains bounding boxes of instances of class i.
        """
        if len(ann_info) == 0:
            return None

        bboxes = [[] for _ in range(len(self.cat2label))]

        for ann in ann_info:
            if ann.get("ignore", False) or ann["iscrowd"]:
                continue
            c_id = ann["category_id"]
            x1, y1, w, h = ann["bbox"]

            bboxes[self.cat2label[c_id]].append([x1, y1, x1 + w, y1 + h, 1.0])

        np_bboxes = []
        for x in bboxes:
            if len(x):
                np_bboxes.append(np.asarray(x, dtype=np.float32))
            else:
                np_bboxes.append(
                    np.asarray([], dtype=np.float32).reshape(0, 5)
                )
        return np_bboxes

```

## 訓練の実行

以下のコマンドで訓練を実行できます。

```
python ./mmdetection/tools/train.py ./src/mmdet/yolox/yolox_l_8x8_300e_coco_occost.p
```

# テスト

ここでは、学習した重みを使用して、`evaluation_images`の画像群に対し推論し、提出用のjsonファイル形式に変換する処理を解説します。


```python:src/dettest/dettest.py
from ast import arg
from typing import Union
from pathlib import Path
import collections as cl
import json
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm
import torch

__JSON_IMAGE_KEY = "images"
__IMAGE_NAME_KEY = "name"
__ANNOTATION_KEY = "annotation"

def get_device(gpu_id=-1):
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    else:
        return torch.device("cpu")

def car_det_test(
    config_file:Union[str, Path], 
    model_checkpoint:Union[str, Path], 
    test_image_dir:Union[str, Path], 
    output_dir:Union[str, Path], 
    cuda:bool=False
):
    if cuda:
        gpu_device = get_device(0)
    else:
        gpu_device = get_device(-1)
    print(f"USE CUDE: {cuda}: {gpu_device}")
    assert Path(config_file).exists(), f"{config_file}が存在しません。"
    assert Path(model_checkpoint).exists(), f"{model_checkpoint}が存在しません。"
    test_image_dir = Path(test_image_dir)
    assert Path(test_image_dir).is_dir(), f"{test_image_dir}はディレクトリではありません。"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    image_save_dir = output_dir / "valid"
    image_save_dir.mkdir(exist_ok=True)
    # モデルの初期化
    model = init_detector(config_file, model_checkpoint, device=gpu_device)
    
    imlist = list(test_image_dir.glob("*.png"))
    js = cl.OrderedDict()
    js[__JSON_IMAGE_KEY] = []
    
    for i, fp in tqdm(enumerate(imlist)):
        _result = inference_detector(model, str(fp))
        result = _result[0]
        
        anotations = []
        for ano in result:
            anotations.append({
                "x": float(ano[0]),
                "y": float(ano[1]),
                "width": float(ano[2] - ano[0]),
                "height": float(ano[3] - ano[1]),
                "score": float(ano[4])
            })
            
        js[__JSON_IMAGE_KEY].append(
            {
                __IMAGE_NAME_KEY: fp.name,
                __ANNOTATION_KEY: anotations
            }
        )
        if i < 5:
            model.show_result(
                fp,
                _result, 
                out_file=image_save_dir / fp.name
            )

    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(
            js, f, ensure_ascii=False, indent=2
        )
    return js

def convert_solafune_publish(input_json:Union[str, Path], output_json:Union[str, Path], score_th:float = 0.5):
    input_json = Path(input_json)
    assert input_json.exists, f"{input_json}が存在しません。"
    output_json = Path(output_json)
    
    with open(input_json, "r") as f:
        js = json.load(f)
    
    image_list = js[__JSON_IMAGE_KEY]
    r = cl.OrderedDict()
    r[__JSON_IMAGE_KEY] = []
    for im in image_list:
        fn = im[__IMAGE_NAME_KEY]
        anos = im[__ANNOTATION_KEY]
        r_anos = []
        for ano in anos:
            score = float(ano["score"])
            if score < score_th:
                continue
            r_anos.append(
                {
                    "class": "car",
                    "lefttop_x": int(ano["x"]),
                    "lefttop_y": int(ano["y"]),
                    "rightbottom_x": int(ano["x"]+ano["width"]),
                    "rightbottom_y": int(ano["y"]+ano["height"])
                }
            )
        r[__JSON_IMAGE_KEY].append(
            {
                __IMAGE_NAME_KEY: fn.replace("png", "tif"),
                __ANNOTATION_KEY: r_anos
            }
        )
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", default="./work_dirs/yolox_l_8x8_50e_coco/yolox_l_8x8_50e_coco.py")
    parser.add_argument("-m", "--model_checkpoint", default="./work_dirs/yolox_l_8x8_50e_coco/best_bbox_mAP_epoch_48.pth")
    parser.add_argument("-i", "--test_image_dir", default="./data/vdet/testset")
    parser.add_argument("-o", "--output_dir", default="./data/vdet/yolox_l_8x8_50e_coco")
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("-s", "--score_th", default=0.5, type=float)
    args = parser.parse_args()
    
    js = car_det_test(args.config_file, args.model_checkpoint, args.test_image_dir, args.output_dir, args.cuda)
    
    output_dir = Path(args.output_dir)
    convert_solafune_publish(output_dir / "result.json", output_dir / "publish.json", score_th=args.score_th)
```

以下のコマンドで実行できます。引数は、`-c=設定ファイル`、`-m=重みファイル`、`-o=出力ディレクトリ`、`-s=BBoxの推論閾値`、`--cuda`でGPU実行になります。出力ディレクトリに出力される`publish.json`が提出用のファイルになります。

```
python ./src/dettest/dettest.py -c=./work_dirs/yolox_l_8x8_300e_coco_occost/yolox_l_8x8_300e_coco_occost.py -m=./work_dirs/yolox_l_8x8_300e_coco_occost/best_xxxxxxx.pth -o=./data/vdet/yolox_l_8x8_50e_coco_occost --cuda -s=0.5
```
## 参考

- [MMDetection – YOLOXを使って自作データセットで物体検出してみた](https://dev.classmethod.jp/articles/mmdetection-train-with-customize-dataset/)

# 最後に

以上、簡単にですが、環境構築やデータの準備から、結果の提出まで解説しました。MMDetectionを使用することで、簡単に一連の処理を実装できました。これから、どのようにして性能を向上させていくか悩み中です。ここでモデルを変えるとか、アンサンブルをするなどは面白くないので、何か普段やらない工夫ができれば良いと思いますw

---

画像・自然言語・音声に関する機械学習の研究開発やMLOpsを行っています。もし、機械学習に関して、ご相談があれば、[@kwashizzz](https://twitter.com/kwashizzz)のアカウントまでDMしてください！
これまでの、[機械学習記事のまとめ](https://zenn.dev/kwashizzz/articles/my-ml-articles-summary)です。