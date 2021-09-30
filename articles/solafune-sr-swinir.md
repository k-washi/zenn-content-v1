---
title: "Solafuneの市街地画像の超解像化コンペのまぁまぁ高精度な解法" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["python", "ml"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

こんにちは！鷲崎([@kwashizzz](https://twitter.com/kwashizzz))です。

普段は、スポーツ分析や音声処理、AWSでの推論環境構築などやっています。
今回は、[Solafune](https://solafune.com/)という衛星データ解析コンテストプラットフォームの市街地画像の超解像化のコンペティションに挑戦してみました。

9月30日時点で6位(スコア：0.869977428102021)ですが、ここで、現在の解法を公開したいと思います。
個人的には、ここから、精度上げるのは、方向転換する必要がある気がしています。

以下のコード等を参考にしつつ、どんどんコンペが接戦になれば嬉しいです。

# モデルに関して

今回使用したモデルは、[SwinIR](https://arxiv.org/abs/2108.10257)です。
2021年の8月に公開された手法で、かなり新しいです。特徴は、Transformerを使用している超解像モデルであることです。精度としても、現在、多くの超解像タスクでSoTAです。
個人的には、とりあえず、性能良いやつを選択しとけ〜と思っています。しかし、コンペで使っていて思ったのが、Transformer系って、もしかして、学習が遅いのではという感想です...

カラー画像ノイズ除去やJPEG圧縮のノイズ除去など多くのタスクごとに事前学習済みモデルが公開されています。しかし、これらのモデルは、Google Colabで`Cuda out of memory`になってしまったので、同時に公開されている軽量版を使用しました。

学習なしで、使用した時点で、Public リーダボードのスコアが、0.853935412634738と恐ろしいスコアを出しています。
Fine Tuningなしで、このスコアは凄すぎます。

しかし、もう少し精度を上げたいので、FineTuningし、現在のスコア(0.869977428102021)となります。
もし、良かったら、軽量版以外で、やってみてください...


# 試したこと

ここでは、FineTuningのために、試したことを紹介したいと思います。

## 現状

モデルは、紹介したSwinIRを使用しています。軽量版のため`32x32`の画像を`128x128`にアップスケールする学習をしています。
損失関数は、モデルの評価指標である、SSIMを使用しています。
Optimizerは、[MADGRAD](https://arxiv.org/abs/2101.11075)を使用しています。これも、新しいのを、とりあえず使っとけという心情です。

学習時には、learning rateを2e-4 => 1e-4 => 2e-5と段階的に低くしています。
データ拡張は色々しています。実装を参考にしてください。

めちゃくちゃ、シンプルですね！個人的に、これ以上学習を複雑にすると、コード管理難しくなり、案件に使うのに躊躇してきます...

## 試したけど上手くいかなかったこと
1. SRGANにならい、GANの形式で学習してみましたが、思ったよりスコアが伸びませんでした。
2. 超解像のデータ拡張である、[CutBlur](https://arxiv.org/abs/2004.00448)を使ってみたかったのですが、モデルの入出力(入力が出力と同じサイズでない)ということで、使用できませんでした...

# 実際の実装

ここから、実際の実装を、羅列していきます。参考にしてください。
注意点として、lrを変化させる部分は、めっちゃコードが複雑なバージョンに書いていたのですが、面倒なので省きました。

[KAIR](https://github.com/cszn/KAIR)の実装をめちゃくちゃ参考にしました。

## データのダウンロード

Google Driveに一時保存しているので、ダウンロード

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%%capture
!mkdir data
%cp -r /content/drive/MyDrive/data/solafune_sresolution /content/data
!mkdir /content/train
!unzip /content/data/solafune_sresolution/train.zip -d /content/train
!mkdir /content/eval
!unzip /content/data/solafune_sresolution/evaluation.zip -d /content/eval
```

## ライブラリ等のダウンロード、import

```python
!pip install timm==0.4.12
!pip install madgrad

# https://arxiv.org/pdf/2108.10257v1.pdf
!git clone https://github.com/JingyunLiang/SwinIR.git
!wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth
```

```python
import torch
import numpy as np
import random
import torch.utils.data as data
import cv2
from glob import glob
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import math
%matplotlib inline


from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
from math import exp
from madgrad import MADGRAD
```


もし、抜けているものがあったら、追加してください。

## データローダー

```python
def augment_img(img, mode):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def rotate_augment(img, mode, deg):
    if mode == 0:
        return img
    elif mode == 1:
        return ndimage.rotate(img, deg, reshape=False)

def adjust(img1, img2, mode):
    if mode == 0:
        return img1, img2
    
    alpha = random.uniform(0.9, 1.1)
    beta = random.uniform(-10, 10)
    img1 = alpha * img1 + beta
    img2 = alpha * img2 + beta
    return np.clip(img1, 0, 255), np.clip(img2, 0, 255)

def hsvshift(img1, img2, mode):
    if mode == 0:
        return img1, img2
    
    ang = random.randint(-45, 45)
    img1 = hsvshift_proc(img1, ang)
    img2 = hsvshift_proc(img2, ang)
    #print("Shift")
    return img1, img2

```

データローダです

```python
class DatasetSR(data.Dataset):
    def __init__(self, l_paths, h_paths, phase="train"):
        super(DatasetSR, self).__init__()
        self.sf = 4 # scale
        self.window_size = 8
        self.patch_size = 128
        self.L_size = self.patch_size // self.sf

        self.paths_H = h_paths
        self.paths_L = l_paths
        self.imgs_H = []
        for h_path in self.paths_H:
            img_H = cv2.imread(h_path, cv2.IMREAD_UNCHANGED)
            img_H = cv2.cvtColor(img_H, cv2.COLOR_BGR2RGB)
            
            H, W, C = img_H.shape
            H_r, W_r = H % self.sf, W % self.sf
            img_H = img_H[:H - H_r, :W - W_r, :]
            self.imgs_H.append(img_H)
        self.imgs_L = []
        for l_path in self.paths_L:
            img_L = cv2.imread(l_path, cv2.IMREAD_UNCHANGED)
            img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
            
            self.imgs_L.append(img_L)

        self.phase = phase

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))
    def __len__(self):
        return len(self.paths_H)
    def __getitem__(self, index):
        H_path = self.paths_H[index]
        L_path = self.paths_L[index]
        img_H = self.imgs_H[index]
        img_L = self.imgs_L[index]

        if self.phase == 'train':

            H, W, C = img_L.shape

            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]


            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # rotate
            mode = random.randint(0, 1)
            deg = random.randint(-30, 30)
            img_L, img_H = rotate_augment(img_L, mode, deg), rotate_augment(img_H, mode, deg)

            mode = random.randint(0, 1)
            img_L, img_H = hsvshift(img_L, img_H, mode)

            # random color
            mode = random.randint(0, 1)
            img_L, img_H = adjust(img_L, img_H, mode)

            mode = random.randint(0, 7)
            img_L, img_H = augment_img(img_L, mode=mode), augment_img(img_H, mode=mode)

        img_H = np.float32(img_H/255.)
        img_L = np.float32(img_L/255.)
        img_H = torch.from_numpy(np.ascontiguousarray(img_H)).permute(2, 0, 1).float()
        img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float()

        _, h_old, w_old = img_L.size()
        h_pad = math.ceil(h_old / self.window_size) * self.window_size - h_old
        w_pad = math.ceil(w_old / self.window_size) * self.window_size - w_old
        img_L = torch.cat([img_L, torch.flip(img_L, [1])], 1)[:, :h_old + h_pad, :]
        img_L = torch.cat([img_L, torch.flip(img_L, [2])], 2)[:, :, :w_old + w_pad]

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}
```

データを読み込みます

```python
eval_num = 5
h_paths = sorted(glob("/content/train/*high.tif"))
l_paths = sorted(glob("/content/train/*low.tif"))
t_h_paths = h_paths[:-eval_num]
t_l_paths = l_paths[:-eval_num]
e_h_paths = h_paths[-eval_num:]
e_l_paths = l_paths[-eval_num:]

test_l_paths = sorted(glob("/content/eval/*low.tif"))

print(len(t_h_paths), len(t_l_paths), len(e_h_paths), len(e_l_paths), len(test_l_paths))
```

```python
BATCH_SIZE = 24
train_set = DatasetSR(t_l_paths, t_h_paths, phase="train")
train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=3,
    drop_last=True,
    pin_memory=True)
test_set = DatasetSR(e_l_paths, e_h_paths, phase="eval")
test_loader = DataLoader(
    test_set,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    drop_last=False,
    pin_memory=True)
```

## utilの作成や設定値など

```python
%cd SwinIR/
!mkdir /content/models
```

```python
from models.network_swinir import SwinIR as net
from main_test_swinir import define_model, setup, get_image_pair
```

```python
sf = 4
window_size = 8
loss_type = "ssim"
G_optimizer_lr = 2e-5
G_weight_decay = 1e-6

batch_size = BATCH_SIZE # 8
device = torch.device('cuda' if [0] is not None else 'cpu')
print(device)
```

```python
def get_bare_model(network):
    if isinstance(network, (DataParallel, DistributedDataParallel)):
        network = network.module
    return network
def load_network(load_path, network, strict=True, param_key='params'):
    network = get_bare_model(network)
    state_dict = torch.load(load_path)
    if param_key in state_dict.keys():
        state_dict = state_dict[param_key]
    network.load_state_dict(state_dict, strict=strict)
    return network
def save_network(save_dir, network, network_label, iter_label):
    save_filename = '{}_{}.pth'.format(iter_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    network = get_bare_model(network)
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())
def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)
```

## loss

SSIM損失です。[SSIM Loss を PyTorch で実装](https://zenn.dev/taikiinoue45/articles/bf7d2314ab4d10)を参考にしました。ぜひ、いいねを押してください！！
実際には、ここの損失を少し拡張しましたが、基本は、これです。

```python
class SSIMLoss(nn.Module):
    def __init__(self, kernel_size: int = 16, sigma: float = 1.5) -> None:

        """Computes the structural similarity (SSIM) index map between two images.

        Args:
            kernel_size (int): Height and width of the gaussian kernel.
            sigma (float): Gaussian standard deviation in the x and y direction.
        """

        super().__init__()
        
        self.kernel_size = 8
        self.sigma = sigma
        self.gaussian_kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)


    def forward(self, x: Tensor, y: Tensor, as_loss: bool = True) -> Tensor:

        if not self.gaussian_kernel.is_cuda:
            self.gaussian_kernel = self.gaussian_kernel.to(x.device)
        
        ssim_map = self._ssim(x, y, self.gaussian_kernel, self.kernel_size)

        if as_loss:
            return 1 - ssim_map.mean() 
        else:
            return ssim_map

    def _ssim(self, x: Tensor, y: Tensor, gaussian_kernel, kernel_size) -> Tensor:
        if not gaussian_kernel.is_cuda:
             gaussian_kerne = gaussian_kernel.to(x.device)

        # Compute means
        ux = F.conv2d(x, gaussian_kernel, padding=kernel_size // 2, groups=3)
        uy = F.conv2d(y, gaussian_kernel, padding=kernel_size // 2, groups=3)

        # Compute variances
        uxx = F.conv2d(x * x, gaussian_kernel, padding=kernel_size // 2, groups=3)
        uyy = F.conv2d(y * y, gaussian_kernel, padding=kernel_size // 2, groups=3)
        uxy = F.conv2d(x * y, gaussian_kernel, padding=kernel_size // 2, groups=3)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denominator = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
        return numerator / (denominator + 1e-12)

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> Tensor:

        start = (1 - kernel_size) / 2
        end = (1 + kernel_size) / 2
        kernel_1d = torch.arange(start, end, step=1, dtype=torch.float)
        kernel_1d = torch.exp(-torch.pow(kernel_1d / sigma, 2) / 2)
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)

        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
        kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size).contiguous()
        return kernel_2d
```

```python
# 色々な損失を試していたので、ここでロスをまとめます。
class GeneratorLoss(nn.Module):
    def __init__(self, device):
        super(GeneratorLoss, self).__init__()
        self.ssim_loss = SSIMLoss().to(device)

    def forward(self, out_images, target_images):
        image_loss = self.ssim_loss(out_images, target_images)
        return image_loss
```

## モデルなどを定義

まず、モデルを定義し、事前学習済みの重みを読み込みます。

```python
model_path = "/content/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"
trained_current_step = 0

G_model = net(upscale=sf, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
G_model = load_network(model_path, G_model, strict=True, param_key='params')
```

```python
# 確認用
test_input=torch.ones(1,3,32,32)
if device == "cuda":
    test_input=test_input.cuda()
    G_model.cuda()
out=G_model(test_input)
print(out.size())
```

次にOptimizerを定義します。
```python
G_optim_params = []
for k, v in G_model.named_parameters():
    if v.requires_grad:
        G_optim_params.append(v)
    else:
        print('Params [{:s}] will not optimize.'.format(k))
G_optimizer = MADGRAD(
    G_optim_params,
    lr=G_optimizer_lr,
    weight_decay=G_weight_decay
)

```

## 最後に訓練です！

1000000ステップとしていますが、実際には、70000程度でストップしました。

```python
!mkdir /content/models
save_dir = "/content/models"
!mkdir /content/imgs
save_img_dir = "/content/imgs"
```

```python
checkpoint_num = 500
log_num = 20
```

```python
G_model.to(device)
g_loss = GeneratorLoss(device).to(device)

current_step = trained_current_step
prev_g_loss_value = 0

for epoch in tqdm(range(current_step, 1000000)):  # keep running
    total_batch_num = 0
    g_loss_value = 0
    current_step += 1
    for i, train_data in enumerate(train_loader):
        G_model.train()
        if train_data['L'].size()[0] != batch_size:
            break
        total_batch_num += batch_size
        
        L_img = train_data['L'].to(device)
        H_img = train_data['H'].to(device)

        # -------------------------------
        # 3) optimize parameters
        # -------------------------------

        fake_image = G_model(L_img)

        # Generetor
        G_optimizer.zero_grad()
        G_loss=g_loss(fake_image, H_img)

        G_loss.backward()
        G_optimizer.step()
        g_loss_value += G_loss.item()
    prev_g_loss_value = g_loss_value / total_batch_num
    # -------------------------------
    # 4) training information
    # -------------------------------
    if current_step % log_num == 0:
        message = f'<epoch:{epoch:3d}, iter:{current_step:8,d}> G_loss: {prev_g_loss_value:<.7f}'
        print(message)

    # -------------------------------
    # 5) save model
    # -------------------------------
    if current_step % checkpoint_num== 0:
        print('Saving the model.')
        save_network(save_dir, G_model, "G", current_step)
        save_optimizer(save_dir, G_optimizer, "optimizerG", current_step)

    # -------------------------------
    # 6) testing
    # -------------------------------
    if current_step % checkpoint_num == 0:
        G_model.eval()
        avg_ssim = 0.0
        idx = 0

        for test_data in test_loader:
            idx += 1
            image_name_ext = os.path.basename(test_data['L_path'][0])
            img_name, ext = os.path.splitext(image_name_ext)

            img_dir = os.path.join(save_img_dir, img_name)
            os.makedirs(img_dir, exist_ok=True)

            data_L = test_data["L"]
            data_H = test_data["H"]
            
            _, _, h_old, w_old = data_L.size()
            h_pad = math.ceil(h_old / window_size) * window_size - h_old
            w_pad = math.ceil(w_old / window_size) * window_size - w_old
            data_L = torch.cat([data_L, torch.flip(data_L, [2])], 2)[:, :, :h_old + h_pad, :]
            data_L = torch.cat([data_L, torch.flip(data_L, [3])], 3)[:, :, :, :w_old + w_pad]
            with torch.no_grad():
                fake_image = G_model(data_L.to(device))

                fake_image = fake_image.detach()[0].float().cpu()
            data_H = data_H.detach()[0].float().cpu()
                
            E_img = tensor2uint(fake_image)
            H_img = tensor2uint(data_H)
            H, W = 1200, 1500
            E_img = E_img[:H, :W, :]
            H_img = H_img[:H, :W, :]

            # -----------------------
            # save estimated image E
            # -----------------------
            save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
            imsave(E_img, save_img_path)
            current_ssim = structural_similarity(E_img, H_img, multichannel=True)

            print('{:->4d}--> {:>10s} | {:<4.7f}dB'.format(idx, image_name_ext, current_ssim))

            avg_ssim += current_ssim

        avg_ssim = avg_ssim / idx
        # testing log
        print('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.7f}dB\n'.format(epoch, current_step, avg_ssim))


```

## 推論です。

モデルを保存していると仮定して、ダウンロード

```python
%cp /content/drive/MyDrive/result/solafune_resolution/swinir_train_v3/39000_G.pth /content/
```

モデルを設定

```python
model_path = "/content/39000_G.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
window_size = 8
scale = 4

model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv').to(device).eval()

param_key_g = 'params'
pretrained_model = torch.load(model_path)
model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
```

その他、使用する関数(上にも書いています。)

```python
def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())
```

推論

```python
test_l_paths = sorted(glob("/content/eval/*low.tif"))
print(len(test_l_paths))
result_dir = "/content/sample_eval"

!mkdir $result_dir
```

```python
for path_L in test_l_paths:
    base_name = os.path.basename(path_L).replace("low.tif", "answer.tif")
    img_L = cv2.imread(path_L, cv2.IMREAD_UNCHANGED)
    img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
    img_L = np.float32(img_L/255.)
    img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float().unsqueeze(0)

    _, _, h_old, w_old = img_L.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    img_L = torch.cat([img_L, torch.flip(img_L, [2])], 2)[:, :, :h_old + h_pad, :]
    img_L = torch.cat([img_L, torch.flip(img_L, [3])], 3)[:, :, :, :w_old + w_pad]
    img_L = img_L.to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_L).detach()[0].float().cpu()
    output = tensor2uint(output)
    output = output[:h_old * scale, :w_old * scale, :]
    print(output.shape)
    save_path = os.path.join(result_dir, base_name)
    print(save_path)
    imsave(output, save_path)
```

コンペに提出するzipファイルにします。
```python
zip_path = result_dir + ".zip"
print(result_dir)
print(zip_path)
!zip -j /content/sample_eval_stage1.zip /content/sample_eval/*.tif
```

# まとめ

超解像コンペで、これをベースラインとして、どんどん、参加して、盛り上がっていくと面白いと思い、公開しました。
まだまだ、やれることはあると思いますが、私は、一旦、このコンペをはなれ、また、最後あたりに戻ってきたいと思います。

この超解像により、衛星画像でも物体検出の精度が上がってくれば面白いと思いました。
衛星画像系の仕事面白そうと思える良いコンペです！

まだ、2ヶ月ほどあるので、みなさん、一緒に頑張っていきましょう！！