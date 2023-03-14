---
title: "ROS2 Tutorialのメモ1" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["ROS", "ROS2"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
publication_name: "fusic"
---

こんにちは！鷲崎([@kwashizzz](https://twitter.com/kwashizzz))です。
最近、Windowsに乗り換え、WSLでROSを動かすため試行錯誤しています。ほぼやることないですが...

WSL環境で環境作成ミスると面倒なので、WSL2のディストリビューションをコピーして作成するようにして、自由に使える環境を作成しています。
方法に関しては、[WSL2で同一環境を複数インストール](https://zenn.dev/fusic/articles/wsl-multi-dist) で解説しています。

# ROS2 Humbleのインストール方法

ROS2のHubmle DistributionがLTSででたので、現在(23/03/15)だと、Humbleをインストールしたほうが良いと思います。

ダウンロードは、基本的に以下の記事を参考にすればよいです。

[Ros2 Humble install in ubuntu (offical)](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)

[ubuntu22.04にROS2 Humbleをインストールする](https://qiita.com/porizou1/items/5dd915402e2990e4d95f)

初心者の場合は、`ros-humble-desktop`を入れておけば問題ないと思います。

毎回、`humble/setup.sh`の実行が面倒なので、

```s
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

として、`.bashrc`に追記しておけばよいです。

ROS1では、ビルドツールとして、`catkin`が使われ、ROS2では、`ament`が使われていましたが、これらを統一した`colcon`がでました。
また、ロボットのシミュレーション環境としてよく使用されるGazeboや、QTベースのGUIツールであるrqtもよく使用するのでインストールします。

```s
sudo apt install python3-colcon-common-extensions
sudo apt -y install gazebo
sudo apt install ros-humble-gazebo-*
sudo apt install ros-humble-rqt-*
```

pythonのインストールが必要な場合、別途行ってください。

# Tutorial

ROS2 Humbleの[Tutorials](https://docs.ros.org/en/humble/Tutorials.html)をやってみます。
ざっくり、内容をまとめつつ、やっていきます。

# [Beginner: CLI tools](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools.html)

## [Configuring environment](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Configuring-ROS2-Environment.html)

>“Workspace”は、ROS 2で開発するシステム上の場所を意味する。ROS 2で開発する場合、通常、複数のワークスペースを組み合わせて、同時に使用することになる。  
ワークスペースを組み合わせることで、ROS2の異なるバージョン、または異なるパッケージに対する開発が容易になる。また、同じコンピュータに複数のROS2ディストリビューションをインストールし、それらを切り替えることができる。これは、新しいシェルを開くたびにセットアップファイルを`source`するか、シェルのスタートアップスクリプトにソースコマンドを一度追加することで実現します。

これは、インストール時に`bashrc`に追加したものですね。
新しいシェルでは、`source /opt/ros/humble/setup.bash`を実行するように書かれています。

### **3. Check environment variables**

`printenv | grep -i ROS`で、環境変数が正しく設定されているか確認します。
以下の環境変数が設定されていればよいです。

```
ROS_VERSION=2
ROS_PYTHON_VERSION=3
ROS_DISTRO=humble
```

### **3.1 The ROS_DOMAIN_ID variable**

ROS2では、ネットワーク内のnode(ROSの処理単位?)を自動的に検知するため、複数のPCでnodeを実行させて分散処理が可能です。その際に、複数人で同じnodeを実行させると問題がおきるため、`ROS_DOMAIN_ID`を設定する必要があります。`ROS_DOMAIN_ID`を設定すれば、同一の値が設定された場所同士でしかnodeを検知できなくなります．

```
echo "export ROS_DOMAIN_ID=1" >> ~/.bashrc
```

`ROS_DOMAIN_ID=0`は、全公開？？

### **3.2 The ROS_LOCALHOST_ONLY variabl**

> デフォルトでは、ROS2の通信はlocalhostに限定されていません。`ROS_LOCALHOST_ONLY`環境変数により、ROS2の通信をlocalhostのみに制限することができます。これは、ROS2システム、およびそのトピック、サービス、アクションが、ローカルネットワーク上の他のコンピュータに表示されないことを意味します。`ROS_LOCALHOST_ONLY`を使用すると、教室など、複数のロボットが同じトピックにパブリッシュして奇妙な動作を引き起こす可能性がある特定の環境で役立ちます。

```
echo "export ROS_LOCALHOST_ONLY=1" >> ~/.bashrc
```

## [Using turtlesim and rqt](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Introducing-Turtlesim/Introducing-Turtlesim.html)

Goal: 今後のチュートリアルに備えて、turtlesimパッケージとrqtツールをインストールし、使用する。

>Turtlesimは、ROS 2を学習するための軽量シミュレータです。rqtは、ROS 2のGUIツールです。rqtで行うことはすべてコマンドラインで行うことができますが、ROS 2の要素をより簡単に、よりユーザーフレンドリーに操作する方法を提供します。ROSの基本概念の説明をします。

### **1 Install turtlesim**

Turtlesimをインストール

```s
sudo apt update

sudo apt install ros-humble-turtlesim
```

パッケージの確認

```
ros2 pkg executables turtlesim
```

出力

```
turtlesim draw_square
turtlesim mimic
turtlesim turtle_teleop_key
turtlesim turtlesim_node
```


### **2 Start turtlesim**

Turtlesimの実行

```
ros2 run turtlesim turtlesim_node
```

### **3 Use turtlesim**

別のターミナルを開いて、以下を実行

```
ros2 run turtlesim turtle_teleop_key
```

実行して、矢印で移動させようとしたが、反応しなかった。（wslだからか？）

しかし、以下のようにnodeなどの確認すると、起動しているようでした。

```s
~/ros-vio-tutorial# ros2 node list
/teleop_turtle

~/ros-vio-tutorial# ros2 topic list
/parameter_events
/rosout
/turtle1/cmd_vel

~/ros-vio-tutorial# ros2 service list
/teleop_turtle/describe_parameters
/teleop_turtle/get_parameter_types
/teleop_turtle/get_parameters
/teleop_turtle/list_parameters
/teleop_turtle/set_parameters
/teleop_turtle/set_parameters_atomically

~/ros-vio-tutorial# ros2 action list
/turtle1/rotate_absolute
~/ros-vio-tutorial# 
```

動作しないなら、無理やりnodeにデータを流し込んでみます。
チュートリアルをそのまま進めて、rqtで実行します。

### 4 Install rqt & 5 Use rqt

```
sudo apt install ~nros-humble-rqt*
```

rqtを実行して、Tutorial上の図のように行っていきます。(Plugins > Services > Service Callerの画面です。)
```
rqt
``

`/turtle1/teleport_absolute`で、`x,y`の値を変更すると、亀が移動すると思います。また、Tutorial通りに、`'turtle2'`を作成し、`ros2 run turtlesim turtle_teleop_key --ros-args --remap turtle1/cmd_vel:=turtle2/cmd_vel`を実行すると、矢印キーで亀が移動しました。