---
title: "【機械学習】動画にない視点の画像を作成してみた! NeRFを時間方向に拡張したNSFF:Nural Scene Flow Fieldの解説" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

こんにちは、鷲崎です。2020年、新しい3次元空間の表現手法として、[Neural Radiance Fields(NeRF)](https://arxiv.org/abs/2003.08934)が登場しました。この手法は、下図のように複数の画像からNeRFを学習することで、任意の視点(位置と角度)から見えるである画像を表現できるようになります。

![nerf overview](https://storage.googleapis.com/zenn-user-upload/2fe7647e5aba44957979a711.png)

この手法の面白いところは、学習に使用した画像の中に存在しない新しい視点の画像が生成できることです。そのため、以下の動画([論文公式ページより参照](https://www.matthewtancik.com/nerf))のように滑らかな視点変化の作成もできます。

![nerf example](https://lh3.googleusercontent.com/x-yjek206ZdhcNa2Op8DwFoh4x_-HJmtNv-tzyecTD-LAe62e3hhNyGVsLxuyolnNATNai60ImRZNMBgpgcb8JmDxl8bfEz_6ueHm-P31W9aWYo61Tw9UkWHDYDuF3Smyh-P4wvTQg=w600)

また、NeRFは、3D空間をニューラルネットワークで表現できるので、より研究が進めば、SLAMなどロボット研究の進歩につながるのではと期待しています。

# NeRF

実際には、NeRFは、画像を表現しているわけではありません。Radiance Fieldsと言う、ある点の物体の色と密度をニューラルネットワークで表現しており、これを基に画像が生成されています。

NeRFの詳細に関しては、[弊社機械学習チームの濱野が書いた記事](https://tech.fusic.co.jp/posts/2020-03-29-read-nelf-paper/)や、[【論文読解】NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections](https://qiita.com/takoroy/items/53e62d303b9743b06801)など、わかりやす記事が多く書かれているので参考にしてください。

# NeRFの欠点

一般的に、インターネット上に公開されている動画は、人間や動物、車などの多様な動的なコンテンツを含んでいます。しかし、NeRFを表現する手法の多くは、学習するシーンが静的であることを仮定しています。そのため、下図の右端のように動的な物体を含む動画では、不確かなNeRFを獲得してしまい、適用が困難です。

そこで、空間に加え時間方向も表現した[Neural Scene Flow Fields (NSFF)](https://www.cs.cornell.edu/~zl548/NSFF/)が提案されました。

![nerf prop](https://storage.googleapis.com/zenn-user-upload/1212075dbcd9726eea3f0e29.png)

# NSFFを実際に試してみました！

空間に加え時間方向も表現したNSFFの詳細な解説の前に、どんな手法であるかイメージを持ってもらうため、実際に試してみた結果を紹介したいと思います！

学習に使用した動画は、以下のものになります。  
![origin](https://lh3.googleusercontent.com/nhxW-hkpLHMFmUwUEg07wBVYIYfL7KwiFCmh6BQV0i9fWHDNdVRUhjSNGY9nGNPad36a6Xp0UqUMidncutGAF4cSFgsMx4GUwWqOtxN9aSt7WrjHaFHwIhozB1kxVliZQ91QcYcW3Q=w600)


この動画から、10FPSで画像を切り出し、それを学習データとして使用しました。

学習には、この動画の他に、カメラパラメータと言われる撮影条件のようなものが必要になります。これは、[COLMAP](https://colmap.github.io/)という、複数の画像から3Dモデルを作成する三次元再構成ライブラリを用いて取得します。COLMAPにより今回の動画から得られる3Dモデルは、以下のようなものになりました。

机の上の静的な物体は、3Dモデルになっていますが、人物やモニターなど動いているものは、あまり結果がよくありませんでした。

![colmap 3d](https://lh3.googleusercontent.com/sk0cZlY326ZogW9MBdmq0q7Q9wIb1ns2qjUCuyrP66IUGOGqvjwgGKf6_z3NyV_qA2SNSbJCgVeydghYldf64j_AFWgc-Ve9cRIpCHdXKfHmSyg4U6rzCiEPB2X7JJ8VHqZkBaHKEg=w600)

さて、COLMAPにより取得した複数枚の画像とカメラパラメータを用いて、[オフィシャルのコード](https://github.com/zhengqili/Neural-Scene-Flow-Fields)を参考にしつつ、NSFFを学習させてみました。学習には、GeForce RTX 3090を使用したのですが、4日ほどかかりました。結果は、下の動画のようになり、かなり良い結果が得られていると思います。 視点を固定したり、同じ視点付近で視点を回転させています。もっと動きがあれば、分かりやすいかもしれませんが、再度、別の動画で学習労力がありませんでした。[公式のサイト](https://www.cs.cornell.edu/~zl548/NSFF/)では、より面白い動画を見ることができます。

![loccam](https://lh3.googleusercontent.com/ofZSgKJw5rCOrDcyfUdY6KoUuOCmwduFLiN8hVrkKQmbkQdaiZ0X59xyBStjYGsE3tk6V9uM-Ryvj4j3TOPp0aSWGFmd5Mo9poZvu3W-GwIwRrN92QD7ehE3gHFcPMDUuqViVmmrbA=w600)


![](https://lh3.googleusercontent.com/XMIjhkgiOKPMBeUzdZamIKM2EifClAcsV3-58I_jTjMI2oynzCtP47RzHy7oANEab6d2qkO_I6sqxcOqU3c9b339YF7Jt5UbfJhqeGkKrLisUFj8wDmiplmVK7y6hsDCcrMgVmnvGg=w600)

では、解説パートに移りたいと思います。

# NSFFとは？

論文中には、「動的なシーンを空間と時間の連続した関数として表現し、その出力に色と密度だけでなく、3Dシーンの動きも含まれていること」を提案したと書かれています。つまりNSFFの概念としては、NeRFに時間方向の情報を加えたものになります。

そして、特に重要な提案部分は、手法の名前にもなっている3Dの密なScene Flow Fieldsをモデル化することです。このモデル化により、空間と時間の両方の変化に対して補間することを可能にしています。一方で、このモデル化は、かなりチャレンジングな問題で、レンダリング品質を向上するために、様々な工夫が行われています。

# 動的な環境におけるScene Flow Fields

下図は、Scene flow fieldsの概要図(論文中Fig.2)です。ある時刻$i$において、あるピクセルの色($\hat{C}_i$)を考えたとき、カメラ中心からそのピクセルを通る線をベクトル$r_i$で表現しています。そして、そのベクトル上の3次元の点は色$c_i$と密度$\sigma_i$で表現されています。

時刻$j$において、あるピクセルに先ほど考えた時刻$i$のピクセルと同じものが写っているとすると、$r_j$上に同じ色と密度の点が存在すると考えられます。そのため、$r_j$上で、時刻$i$における色$c_i$と密度$\sigma_i$に一致する点を探索することで、色$c_j$と密度$\sigma_j$の位置を推測できます。

このとき、時刻$i$から$j$で、3次元空間を移動した点の移動量をScene flow $f_{i \rightarrow j}$と言います。もし、静的な物体だと、移動しないのでScene flowは小さいです。一方で、動的な物体は、時刻と共に変化するので、Scene flowも大きくなります。

実際の処理では、時刻$i$に対応したScene Flowとして時刻$i-1$と時刻$i+1$のものを計算します。

![scine flow filed wrap](https://storage.googleapis.com/zenn-user-upload/ca7c5f39e69e593c410c1d69.png)


## Scene Flow の最適化

Scene Flowの損失関数は、多くの曖昧さに対応するため、以下のような3つの工夫により構成されています。

1. 輝度一貫性  
上図の通り、Scene Flowを用いると、時刻$j$における3次元表現を時刻$i$にワープさせることができ、時刻$i$におけるピクセルの色$\hat{C}_{j \rightarrow i}$になります。そのため、$\hat{C}_{j \rightarrow i}$と$\hat{C}_i$が一貫しているという仮定を置くことができます。
しかし、この一貫性は、オクルージョンなどにより、物体の境界で曖昧になることがあります。下図(論文中 Fig. 3)を見てください。一番下のワープさせた画像(Warped Image)では、実際の画像(Rendered Image)と一致しない部分(?の部分)が発生しています。
この曖昧さに起因するエラーを軽減するため、Scene Flowに対応する重み$w_{i \rightarrow j}$を追加で予測するようにしています。この重みは、最適化時に、輝度一貫性のエラーをどの程度の強さで適用すべきかという情報を与えます。

![Fig3](https://storage.googleapis.com/zenn-user-upload/a8298ac36af90acb174aa09d.png)

2. サイクル一貫性  
時刻$i$で予測されたScene flow $f_{i \rightarrow j}$は、時刻$j$におけるScene flow $f_{j \rightarrow i}$に対応します。そのため、時刻$i$の3次元点$x_i$に対するScene Flow $f_{i \rightarrow j}$ と 時刻$i$の点$x_i$をScene Flow$f_{i \rightarrow j}$でワープさせた点 $x_{i \rightarrow j}$に対するScene flow $f_{j \rightarrow i}$は対応したベクトルであるため(向きは逆)、$||f_{i \rightarrow j} + f_{j \rightarrow i}||_1$が小さいほど、一貫性があることになります。

3. 幾何学的整合性  
   この幾何学的整合性は、三次元再構成の分野では、よく適応される手法で、隣接するフレーム間の正確な対応関係を構築します。
   例えば、隣接したフレーム間のピクセルの移動量$u_{i \rightarrow j}$をオプティカルフローで計算するとします。この時、ピクセル$p_{i \rightarrow j} = p_{i} + u_{i \rightarrow j}$の関係が成り立ちます。一方で、別ルートでこのピクセルの推定値$\hat{p}_{i \rightarrow j}$を計算することができます。まず、scene flow $f_{i \rightarrow j}$と$r_i$上を通る3次元点$x_i$を計算し、$x_{i \rightarrow j}$に射影した点を2次元ピクセルに変換すると$\hat{p}_{i \rightarrow j}$が計算できます。これらの誤差は、再投影誤差と呼ばれ、良く使用されています。

   また、安定性のため単視点深度項も設けられています。この項は、別途用意した単眼深度推定ネットワークにより推定された深度と、予測した密度$\sigma_i$から計算される深度の差を考えています。

    この幾何学的整合性はノイズが多い(試した動画でもノイズが多かった)ため、学習の初期化のみに使用され、学習中は重みを0にして無視しています。

# 静的なシーン表現との統合

上記のScene Flow Fieldを用いることで、動的な画像群に対して、最先端の性能になります。しかし、静的なシーンにおいて、以前のNeRFを用いて表現した方が性能が向上することが論文内で示されています。そこで、各ピクセルにおいて、多層パーセプトロンを用い、動的なシーン表現(色、密度)と静的なシーン表現、これらのブレンド率を出力し、線形補間的に統合することが提案されています。もし、動的な部分の場合、動的なシーン表現が大きな割合で割り当てられ、一方で静的な部分の場合、静的なシーン表現が割り当てられます。

# 時間情報の補正

動的なシーンの場合、時刻$i$と$i+1$の間の情報を補間する必要があります。基本的には、3次元点$x_i + \delta_i f_{i \rightarrow 1}$で線形補間されます。すると、時刻$i + \delta_i$の色と密度表現は、時刻$i$, $i+1$の色と密度表現を$\delta_i f_{i}$、$(1 - \delta_i) f_{i+1}$で変異させ、$\delta_i$, $\delta_{i+1}$の割合でブレンドしたものになります。

# まとめ

最後に、実際の比較画像(論文中Fig.8)を示します。明らかに破綻が少なく、動的なシーンも表現できています。NSFFも含めNeRFは、比較的に新しい分野で、まだまだ研究すべき部分はたくさんあります。例えば、今回、試した際の学習に4日ほどかかりました。たった、数秒の表現を学習するためにこれだけ時間がかかるのは、大きな欠点だと思います。しかし、[KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs](https://arxiv.org/abs/2103.13744)のように学習速度を早くする研究が行われていたり、汎用的にする研究なども行われています。今後、この研究が進むことで、ゲームのように好きな視点で、見たい風景を見る日が来ると思います。

![Fig8](https://storage.googleapis.com/zenn-user-upload/ab2c4cd686dca8794ca8a400.png)

