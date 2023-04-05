---
title: "イコライザーを用いた音声データ拡張" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習", "音声"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
publication_name: "fusic"
---

こんにちは！[Fusic](https://fusic.co.jp/) 機械学習チームの鷲崎です。機械学習モデルの開発から運用までなんでもしています。もし、機械学習で困っていることがあれば、気軽に[お問い合わせ](https://fusic.co.jp/contact/)ください。[TwitterのDM](https://twitter.com/kwashizzz)でも大歓迎です！！

最近、[so-vits-svc(SoftVC VITS Singing Voice Conversion)](https://github.com/svc-develop-team/so-vits-svc)という、性能の良いVoice Conversion (VC)が流行っている気がします。VCでは、入力音声から話者性を除去したコンテンツ情報の抽出が重要になります。コンテンツ情報に一貫性を持たせて、話者性を変更する際に、イコライザーによるデータ拡張も行うことがあります。このデータ拡張を実際試したところ、スペクトロうグラム上での見た目はかわりませんでしたが、実際聞いた際に、若干、声を張り上げているように変換されていたりしました。

本記事では、イコライザーを用いた音声データ拡張について実装の解説します。
イコライザーの基礎に関しては、[エフェクターの基礎知識①：イコライザー＆フィルターの基礎を理解しよう！](https://kensukeinage.com/effector_eq_and_filter/) という記事がとても参考になりました。

# 概要

この記事から引用すると、イコライザーとは、

> 音に含まれる周波数バランスをコントロールするエフェクター

とのことです。

高周波成分や、低周波成分の増幅や、一部周波数帯の減少を行うことで、音を聞きやすくしたり、ハウリングを抑えたりされるらしいです。

まず、単語の解説をします。

音声からのコンテンツ抽出器として有名な[ContentVec](https://arxiv.org/abs/2204.09224)の学習で使用されたデータ拡張用のイコライザーのフィルターとして、`Peak Filter`, `High Shelf Filter`, `Low Shelf Filter`があります。

- `Peak Filter`は、指定した周波数を中心として、増幅や減少を行うフィルタです。
- `High Shelf Filter`は、指定した周波数以上の帯域にある信号の増幅や減少を行うフィルターです。
- `Low Shelf Filter`は、指定した周波数未満の帯域にある信号の増幅や減少を行うフィルターです。

これらのフィルターを形成するパラメータとして、$F$, $G$, $Q$があります。

- $F$は、基準となる周波数です。
- $G$は、イコライザーによる周波数増減の強さです。
- $Q$は、基準周波数$F$を中心とした帯域幅です。

# 実装の流れ

実装に関しては、[ContentVecのgithub](https://github.com/auspicious3000/contentvec)を参考に紹介します。

イコライザーでデータ拡張を行う実装を切り出しています。

```python: https://github.com/auspicious3000/contentvec/blob/main/contentvec/data/audio/contentvec_dataset.py
import numpy as np
from scipy.signal import sosfilt
from fairseq.data.audio.audio_utils_1 import params2sos
Qmin, Qmax = 2, 5

...

    def random_eq(self, wav, sr):
        z = self.rng.uniform(0, 1, size=(10,))
        Q = Qmin * (Qmax / Qmin)**z
        G = self.rng.uniform(-12, 12, size=(10,))
        sos = params2sos(G, self.Fc, Q, sr)
        wav = sosfilt(sos, wav)
        return wav
```

`uniform`関数でランダムな値を`size`分作成しています。これは、以下のようなジェネレータオブジェクトから生成されます。このように、ジェネレータを作成している理由として、乱数生成処理の高速化があるそうです。

```python
self.rng = np.random.default_rng()
```

次に、ランダム値と、$Q$の最小値、最大値を用いてランダムな$Q$を定義しています。ゲイン$G$も$-12~12$の範囲でランダムな値を生成しています。

その後、sos(second-order sections; 2次セクション型) IIRデジタルフィルターを波形データ`wav`に適用することで、データ拡張をしています。`params2sos`は、上記の`Peak Filter`などの係数をパラメータから計算する関数で、その係数を`sosfilt`関数で適用します。

[`matlabのsosfilt`に関するドキュメント](https://jp.mathworks.com/help/signal/ref/sosfilt.html)を参考に説明します。
デジタル フィルターの設計を行う場合、伝達関数$H$を設計することになになります。この伝達関数を信号と掛け合わせることで、イコライズした結果を得ることができます。sosは、以下の行列のように、1行6列の係数で1つのフィルターを表し、$L$行（個）のフィルターに関する行列からなります。そして、この行列から、下の式で、伝達関数$H$を表現しています。

$$
\operatorname{sos}=\left[\begin{array}{cccccc}
b_{01} & b_{11} & b_{21} & 1 & a_{11} & a_{21} \\
b_{02} & b_{12} & b_{22} & 1 & a_{12} & a_{22} \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
b_{0 L} & b_{1 L} & b_{2 L} & 1 & a_{1 L} & a_{2 L}
\end{array}\right]
$$

$$
H(z)=\prod_{k=1}^L H_k(z)=\prod_{k=1}^L \frac{b_{0 k}+b_{1 k} z^{-1}+b_{2 k} z^{-2}}{1+a_{1 k} z^{-1}+a_{2 k} z^{-2}}
$$

この伝達関数を作成するための各フィルターの係数は、各々のフィルター特性によって決まっています。これらの式は、[Cookbook formulae for audio equalizer biquad filter coefficients](https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html)に記載されています。

例えば、`Peak Filter`の場合、以下のようなコードになります。式を単純に書き起こしただけです。

```python
def make_peaking(g, fc, Q, fs=44100):
    """Generate filter coefficients for 2nd order Peaking EQ.
    Args:
        g  (float): Gain factor in dB.
        fc (float): Cutoff frequency in Hz.
        Q  (float): Q factor.
        fs (float): Sampling frequency in Hz.
    Returns:
        tuple: (b, a) filter coefficients 
    """
    # convert gain from dB to linear
    g = np.power(10,(g/20))

    # initial values
    A = np.max([0.0, np.sqrt(g)])
    omega = (2 * np.pi * np.max([fc, 2.0])) / fs
    alpha = np.sin(omega) / (Q * 2)
    c2 = -2 * np.cos(omega)
    alphaTimesA = alpha * A
    alphaOverA = alpha / A

    # coefs calculation
    b0 = 1 + alphaTimesA
    b1 = c2
    b2 = 1 - alphaTimesA
    a0 = 1 + alphaOverA
    a1 = c2
    a2 = 1 - alphaOverA
    
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
```

他にも、`High shelf filter`や`Low Shelf filter`など[ontentvec/blob/main/contentvec/data/audio/audio_utils_1.py](https://github.com/auspicious3000/contentvec/blob/main/contentvec/data/audio/audio_utils_1.py)にコードが記載されています。

各フィルタを重ねたsosの行列を作成する`params2sos`関数は、以下のコードとなします。

```python
def params2sos(G, Fc, Q, fs):
    """Convert 5 band EQ paramaters to 2nd order sections.
    Args:
        x  (float): Gain factor in dB.       
        fs (float): Sampling frequency in Hz.
    Returns:
        ndarray: filter coefficients for 5 band EQ stored in (5,6) matrix.
        [[b1_0, b1_1, b1_2, a1_0, a1_1, a1_2],  # lowshelf coefficients
         [b2_0, b2_1, b2_2, a2_0, a2_1, a2_2],  # first band coefficients
         [b3_0, b3_1, b3_2, a3_0, a3_1, a3_2],  # second band coefficients
         [b4_0, b4_1, b4_2, a4_0, a4_1, a4_2],  # third band coefficients
         [b5_0, b5_1, b5_2, a5_0, a5_1, a5_2]]  # highshelf coefficients
    """
    # generate filter coefficients from eq params
    c0 = make_lowshelf(G[0], Fc[0], Q[0], fs=fs)
    c1 = make_peaking (G[1], Fc[1], Q[1], fs=fs)
    c2 = make_peaking (G[2], Fc[2], Q[2], fs=fs)
    c3 = make_peaking (G[3], Fc[3], Q[3], fs=fs)
    c4 = make_peaking (G[4], Fc[4], Q[4], fs=fs)
    c5 = make_peaking (G[5], Fc[5], Q[5], fs=fs)
    c6 = make_peaking (G[6], Fc[6], Q[6], fs=fs)
    c7 = make_peaking (G[7], Fc[7], Q[7], fs=fs)
    c8 = make_peaking (G[8], Fc[8], Q[8], fs=fs)
    c9 = make_highself(G[9], Fc[9], Q[9], fs=fs)

    # stuff coefficients into second order sections structure
    sos = np.concatenate([c0,c1,c2,c3,c4,c5,c6,c7,c8,c9], axis=0)

    return sos
```

最終的には、この行列を`sosfilt`で音声データに適応すると、データ拡張され、少し変化した音声が出力されます。



# まとめ

ContentVecで使用されていたデータ拡張で、有用そうなので、調べてみました。実際使用すると、スペクトログラム上だと、ほとんど変化していないように見えますが、実際に聞くと、変化が分かるので、記事を参考に試してみてください。実際には、この処理の前に、F0を変化させるデータ拡張なども行っていたので、のちほどこちらに関しても記事にしようと思います。

最後に宣伝になりますが、機械学習でビジネスの成長を加速するために、[Fusic](https://fusic.co.jp/)の機械学習チームがお手伝いしています。機械学習のPoCから運用まで、すべての場面でサポートした実績があります。もし、困っている方がいましたら、ぜひ[Fusic](https://fusic.co.jp/)にご相談ください。[お問い合わせ](https://fusic.co.jp/contact/)から気軽にご連絡いただけますが、[TwitterのDM](https://twitter.com/kwashizzz)からでも大歓迎です！