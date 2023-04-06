---
title: "信号処理による話者性変換を用いた音声データ拡張" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習", "音声"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
publication_name: "fusic"
---

こんにちは！[Fusic](https://fusic.co.jp/) 機械学習チームの鷲崎です。機械学習モデルの開発から運用までなんでもしています。もし、機械学習で困っていることがあれば、気軽に[お問い合わせ](https://fusic.co.jp/contact/)ください。

最近、[so-vits-svc(SoftVC VITS Singing Voice Conversion)](https://github.com/svc-develop-team/so-vits-svc)という、性能の良いVoice Conversion (VC)が流行っている気がします。VCでは、入力音声から話者性を除去したコンテンツ情報の抽出が重要になります。そのため、コンテンツ情報に一貫性を持たせて、話者性を変更するデータ拡張が使用されることがあります。例えば、ピッチを変更するデータ拡張やイコライザーを持ちいたデータ拡張です。

イコライザーを用いた音声データ拡張に関しては、[前回](https://zenn.dev/fusic/articles/ml-content-vec-equalizer-aug)記事にしました。スペクトログラム上では、あまり変化していないように見えましたが、実際聞いた際に、若干、声を張り上げているよう聞こえたりし、環境が変化したような音声に拡張されていました。


本記事では、ピッチを変更するデータ拡張について解説します。コンテンツ情報を抽出するように学習された[ContentVec](https://arxiv.org/abs/2204.09224)では、フォルマント周波数と基本周波数をランダムにスケールさせて話者性を変化させた結果に対して対照学習を行うことで、抽出コンテンツ情報の一貫性を保たせるという工夫をしています。実際、ピッチを変更するデータ拡張を行うと、男性の声を女性っぽく変更したり、さらに声を低くしたりといった拡張が達成できていました。

[ContentVec](https://arxiv.org/abs/2204.09224)の論文では、フォルマント周波数と基本周波数をスケールと書いていますが、実際の実装では、ピッチを遷移させることで、話者性の遷移を達成しています。(これは、等価なのでしょうか？)

# Change Gender

[ContentVec](https://arxiv.org/abs/2204.09224)では、以下のような関数でピッチのデータ拡張を行っています。

```python
import parselmouth
def change_gender(x, fs, lo, hi, ratio_fs, ratio_ps, ratio_pr):
    s = parselmouth.Sound(x, sampling_frequency=fs)
    f0 = s.to_pitch_ac(pitch_floor=lo, pitch_ceiling=hi, time_step=0.8/lo)
    f0_np = f0.selected_array['frequency']
    f0_med = np.median(f0_np[f0_np!=0]).item()
    ss = parselmouth.praat.call([s, f0], "Change gender", ratio_fs, f0_med*ratio_ps, ratio_pr, 1.0)
    return ss.values.squeeze(0)
```

ここで重要な部分は、`praat.call`で`Change gende`を実行している部分です。引数の詳細などは、[Sound: Change gender...](https://www.fon.hum.uva.nl/praat/manual/Sound__Change_gender___.html)を参考にしてください。ここで、ピッチを遷移させており、以下の式で計算されます。

`finalPitch = newPitchMedian + (pitch * newPitchMedian / oldPitchMedian - newPitchMedian) * pitchRangeScaleFactor`

ここの、`oldPitchMedian`は、関数中で`to_pitch_ac`で`f0`を抽出し、`median`を取っている部分です。

`Change gende`の引数は、以下になります。

- `ratio_fs`は、周波数の遷移割合を表しており、$1.1$にした場合、声が高くなり女性っぽい声になり、$1/1.1$に設定した場合は、声が低くなり男性っぽい声になります。
- `f0_med*ratio_ps`は、新しいピッチの中央値（`newPitchMedian`）で、元の音声のF0の中央値を`ratio_ps`でスケールさせており、`0`Hzで、オリジナルの音声と同じピッチになるとのことです。
- `ratio_pr`は、上式の`pitchRangeScaleFactor`で、最終的なピッチを決定する際に、どの程度`newPtichMedian`を重要視するかというスケール値です。
- 最後の引数$1$は、`Duration factor`で、時間方向のスケールを行うと時間ごとのコンテンツ情報が変化してしまうため、固定で1にしています。



# データ拡張の実装

実際に`change gender`を用いてデータ拡張を行う実装は、以下になります。詳細は、[auspicious3000/contentvec/blob/main/contentvec/data/audio/contentvec_dataset.py](https://github.com/auspicious3000/contentvec/blob/main/contentvec/data/audio/contentvec_dataset.py)を参考にしてください。

音声波形`wav`と、その周波数、そして、`spk2info`から情報を抽出するための話者index`spk`が引数となっています。

最小pitch`lo`と最大pitch`hi`の設定ですが、話者ごとに設定するのがベストです。しかし、作者にきいたところ、基本的には、男性、女性の属性ごとに設定しており、男性の場合、`lo=50, hi=250`, 女性の場合、`lo=100, hi=400`としているとのことです。メソッドの中では、一部修正しており、男性の場合`lo=50`を$75$に変更しています。

`change_gender`の他の引数に関しては、ランダムに生成しています。

```python: contentvec/blob/main/contentvec/data/audio/contentvec_dataset.py
...
    def random_formant_f0(self, wav, sr, spk):
        #s = parselmouth.Sound(wav, sampling_frequency=sr)
        _, (lo, hi, _) = self.spk2info[spk]
        
        if lo==50:
            lo=75
        if spk=="1447":
            lo, hi = 60, 400
        
        ratio_fs = self.rng.uniform(1, 1.4)
        coin = (self.rng.random() > 0.5)
        ratio_fs = coin*ratio_fs + (1-coin)*(1/ratio_fs)
        
        ratio_ps = self.rng.uniform(1, 2)
        coin = (self.rng.random() > 0.5)
        ratio_ps = coin*ratio_ps + (1-coin)*(1/ratio_ps)
        
        ratio_pr = self.rng.uniform(1, 1.5)
        coin = (self.rng.random() > 0.5)
        ratio_pr = coin*ratio_pr + (1-coin)*(1/ratio_pr)
        
        ss = change_gender(wav, sr, lo, hi, ratio_fs, ratio_ps, ratio_pr)
        
        return ss
```

このメソッドを用いることで、ランダムにピッチを変更し話者性を信号処理的にですが、変更できていました。

# まとめ

簡単にですが、ピッチに関するデータ拡張について解説しました。思った以上に、簡単に実装できるので、音声処理のデータ拡張において、もっと一般的になればと思います。

最後に宣伝になりますが、機械学習でビジネスの成長を加速するために、[Fusic](https://fusic.co.jp/)の機械学習チームがお手伝いしています。機械学習のPoCから運用まで、すべての場面でサポートした実績があります。もし、困っている方がいましたら、ぜひ[Fusic](https://fusic.co.jp/)にご相談ください。[お問い合わせ](https://fusic.co.jp/contact/)から気軽にご連絡いただけますが、[TwitterのDM](https://twitter.com/kwashizzz)からでも大歓迎です！