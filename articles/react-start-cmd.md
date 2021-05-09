---
title: "Reactに関するメモ" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["react"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

# nodenvでnpm環境設定

nodenvのインストール
```bash
brew install anyenv
echo 'eval "(anyenv init -)"' >> ~/.zshrc
exec $SHELL -l
anyenv install nodenv
exec $SHELL -l 
```

anyenv, nodenvプラグインのインストール
```
mkdir -p $(anyenv root)/plugins
git clone https://github.com/znz/anyenv-update.git $(anyenv root)/plugins/anyenv-update
mkdir -p "$(nodenv root)"/plugins
git clone https://github.com/nodenv/nodenv-default-packages.git "$(nodenv root)/plugins/nodenv-default-packages"
touch $(nodenv root)/default-packages
```
以下のライブラリを```default-packages```に記載。
```
yarn
typescript
ts-node
typesync
```

anyenvのアップデート
```
anyenv update 
```

nodeのインストール
```
nodenv install -l   
nodenv install 14.4.0
nodenv global 14.4.0    
```

# React環境作成

```
npx create-react-app hello-world --template=typescript
```

# Yarn

```bash
yarn (install) -- package.jsonの記述に従ってインストール
yarn add <PACKAGE_NAEM> -- 指定パッケージインストール
yarn remove <PACKAGE_NAME> -- 指定パッケージ アンインストール
yarn upgrade <PACKAGE_NAME> -- 指定パッケージを最新バージョンに更新
yarn info <PACKAGE_NAME>
```

package.json内にもコマンドが記載されている。

最新版のパッケージを確認する
```
yarn upgrade-interactive
```

# config
追加項目
package.json
```
"type": "module",
```

tsconfig.json  
```
strict: いろいろなstrict関係をtrueにする
target: es6にして良いかも(IEがなければ)
```

追加したい設定
```
"baseUrl": "src",
"downlevelInteration": true,
```

baseUrlで絶対パス指定が可能になる。しかし、VSCodeで認識してくれないことが偶にあるので、以下の対処法を行う。
- shift + command + P（※ Windows では Shift + Ctrl + P）または F1 キーでコマンドパレットを 開き、>type を入力するとリストアップされる「TypeScript: Reload Project」を選択・実行 
- コンソールから touch tsconfig.json を実行

# Javascript

- 第一級関数とクロージャをサポート
- プロパティを随時追加できる柔軟なオブジェクト
- 表現力の高いリテラル記法

条件演算子
```
const judge = rand % 2 === 0 ? 'even': 'odd';
```

クラス
```
class A{
    constructor(a) {
        this.a = a;
    }
    b = () => {};
    static c = () => {};
}

class D extends A {}

const aa = new A("test");
A.c();
```

分割代入
```
const res = {
    data: [
        {id:1},
        {id:2}
    ]
};

const {data: users = []} = res; //　デフォルトは空配列
console.log(users);
//[{id:1}, {id:2}]
```

オブジェクトのコピー
```
const data = {a:1, b:2}
const coyp = {...data} // shallow copy

//deep copy; Dateオブジェクトやundefinedがある場合うまくいかない
const deep_copy = JSON.parse(JSON.stringify(data)) 
```

Nullish, Optional Chaining
```
//data?.name => nameがないなら、undefinedを返す
// data?.nameがnull, undefinedなら ?? 以降が評価され処理される

const a = data?.name ?? {name: 'other'} 
```

sortは破壊的なメソッド
```
const list = [5, 7, 1, 3];

lst.slice().sort((n, m) => n < m ? -1 > 1); # sliceを使用する
```

高階関数 => 入出力に関数を使用  

カーリー化
```
const withMultiple = (n) => (m) => n * m;
withMultiple(3)(5)
const triple = withMultiple(3);
triple(5); //常に3の倍数を返す関数になっている
```

# typescript

インターフェース
```
interface Color {
    readonly rgb: string //読み込みだけ
    opacity: number;
    name?: string // ?で省略可能
    [attr: string]: number;
}
const tt: Color = {rgb: '00afcc', opacity: 1, id: 1};
tt.name = 'a';
```

列挙 enum
```
enum Pet {Cat, Dog, Rabbit}
Pet.Cat // 0
Pet.Dog // 1
let Tom: Pet = Pet.Cat; // 0
```

リテラル型
```
let Mary: 'Cat' | 'Dog'| 'Rabbit' = 'Cat'
Mary = 'Dog'
```

タプル型
```
const userAttrs: [number, string, boolean] = [7, 'name', true];
const [id, username, isAdmin] = userAttrs;
```

クラス  
クラスは型であり関数でもある！
クラスの継承はいけてない！
```
interface Shape {
    readonly name: string;
    getArea: () => number;
}

interface Quadrangle {
    sideA: number;
    sideB?: number;
}

class Rectangle implements Shape, Quadrangle {
    readonly name = 'rectangle';
    sideA: number;
    sideB: number;

    constructor(sideA:number, sideB: number) {
        this.sideA = sideA;
        this.sideB = sideB;
    }

    getArea = (): number => this.sideA * this.sideB;
}
```

型エイリアス  
拡張不可などinterfaceより安全性が高い  
interfaceより型エイリアスを優先的に使用した方が良い  
```
typs Shape {
    readonly name: string;
    getArea: () => number;
}

//交差型を用いて、extendのようなことができる
type a = Shape & {
    data: int[];
}
```

Nullも使用したいばあい
```
let foo: string | null = 'foo';
```

テンプレートリテラル型
```
type DataFormat = `${number}-${number}-${number}`

const date1: DateFormat = '2020-12-05'
```

```ts
const tables = ['users', 'posts', 'comments'] as const; 
type Table = typeof tables[number]; type AllSelect = `SELECT * FROM ${Table}`; 
type LimitSelect = `${AllSelect} LIMIT ${number}`; 
const createQuery = (table: Table, limit?: number): AllSelect | LimitSelect => 
    limit ? `SELECT * FROM ${table} LIMIT ${limit}` as const 
    : `SELECT * FROM ${table}` as const; 
const query = createQuery('users', 20);
```

オーバーライド
```ts
class Brooch {
  pentagram = 'Silver Crystal';
}

type Compact = {
  silverCrystal: boolean;
};

class CosmicCompact implements Compact {
  silverCrystal = true;
  cosmicPower = true;
}

class CrisisCompact implements Compact {
  silverCrystal = true;
  moonChalice = true;
}

type Transform = {
  (): void;
  (item: Brooch): void;
  (item: Compact): void;
};

const transform: Transform = (item?: Brooch | Compact): void => {
  if (item instanceof Brooch) {
    console.log('Moon crystal power💎, make up!!');
  } else if (item instanceof CosmicCompact) {
    console.log('Moon cosmic power💖, make up!!!');
  } else if (item instanceof CrisisCompact) {
    console.log('Moon crisis🏆, make up!');
  } else if (!item) {
    console.log('Moon prisim power🌙, make up!');
  } else {
    console.log('Item is fake...😖');
  }
};

transform();
transform(new Brooch());
transform(new CosmicCompact());
transform(new CrisisCompact());
transform({pengram: '***'}) // Item is fake... 構造で判断される

```

アサテーション

```
const a = name as string

// or 型ガード
if (typeof a == 'string') {
    a.split(',')
}
```