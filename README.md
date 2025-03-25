# ゼロから作る Deep Learning 3 を Rust で実装してみる。
* 本プロジェクトは、O'Reilly Japan の書籍『ゼロから作る Deep Learning 3』を基に、Rust で実装することを目的としています。
* https://www.oreilly.co.jp/books/9784873119069/

## 基本的な使い方
* [main.rs](https://github.com/t2hk/RustZeroDeepLearning/blob/main/src/main.rs) を参照

## 実行方法
* ステップ19まで

```
cargo run --bin [RSファイル]
```

* ステップ20以降

```
cargo run
```

## 開発履歴

### ステップ1 箱としての変数
* Python の numpy の代替として Rust の ndarray を使用する。
* 変数は Array::from_elem(IxDyn(&[]), f64) 固定とした。

### ステップ2 変数を生み出す関数
* Function をトレイトで定義する。
* Function トレイトの実装として Square や Exp を実現する。

### ステップ3 関数の連結
* call メソッドで関数を連結して実行できるようした。
* ただし、Python と異なり、call メソッドを明示的に呼び出している。

### ステップ4 数値微分
* 書籍通り数値微分を実装した。

### ステップ5 バックプロパゲーションの理論
* 机上学習

### ステップ6 手作業によるバックプロパゲーション
* 逆伝播のために入力値をどのように保持させるか悩む。
* Square や Exp など関数用の構造体に入力値を内包させ、ライフタイムも使用してみる。
* この方法は次の ステップ7 で見直すことになる。

### ステップ7 バックプロパゲーションの自動化
* ステップ9までの仕組みを検証しながら実装。
* 計算グラフを実現するため、Rc, RefCell, Weak を導入する。
* ステップ6 で使ったライフタイムは不要となった。
* 関数の入出力値を Weak で関数用の構造体に内包させてみたり、専用の構造体に内包させてみたり。

### ステップ8 再帰からループへ
* ステップ7の実現と合わせて検討・検証しつつ対応した。

### ステップ9 関数を便利に
* ステップ7の実現と合わせて検討・検証しつつ対応した。

### ステップ10 テストを行う
* Rust のテストモジュールとして実装した。

### ステップ11 可変長の引数 (順伝播編)
* 順伝播について、vec で可変長の引数に対応し、加算関数を追加した。
* 逆伝播については一時的にコメントアウトしている (ステップ13 で対応予定)。
* 書籍では関数とそのの出力結果をリンクさせ、出力から関数へのリンクを辿って逆伝播を自動化している (リンクノード)。
* Rust で同様の仕組みを実現する方法がわからず、関数をベクタで保持して繰り返し処理する方法で実現してみた。

### ステップ12 可変長の引数 (改善編)
* add メソッドや square メソッドを追加し、計算を行いやすく改善した。

### ステップ13 可変長の引数 (逆伝播編)
* 逆伝播も可変長に対応した。
* FunctionExecutor という関数のラッパー構造体を追加し、書籍と同様に、関数とその出力結果をリンクさせ、出力から関数へのリンクを辿って逆伝播を自動化できるようにした。
* FunctionExecutor の役割
  * 関数の入力値、出力値、及び関数自身のトレートオブジェクト(出力値の creator) を保持する。
  * forward, backward を実装する。
  * 逆伝播の自動化のため、自身の入力値とその creator を繰り返し辿ることで逆伝播の計算グラフを再現できるようにした。

### ステップ14 同じ変数を繰り返し使う
* 書籍通り微分の加算やクリアに対応した。

### ステップ15 複雑な計算グラフ (理論編)
* 机上学習

### ステップ16 複雑な計算グラフ (実装編)
* 世代を追加した。
* 世代順に計算グラフを構築するため、優先度付きキューを採用した。
* FunctionExecutor トレートオブジェクトを世代順に並べるため、以下のように工夫した。
  * FunctionExecutor に対して PartialEq、Eq、PartialOrd、Ord を実装した。
    * PartialEq: FunctionExecutor のトレートオブジェクトのポインタが一致する場合、同一と判断する。
    * PartialOrd, Ord: FunctionExecutor の世代の大小比較により優先度を判断する。
  * BinaryHeap を使って世代順にソートする。
  * 同一の変数を複数回使用する場合、その creator を重複して検出してしまうため、creator のトレートオブジェクトのポインタが一致する場合は同一と判断して計算グラフから除外する。

### ステップ17 メモリ管理と循環参照
* FunctionExecutor 構造体で保持していた関数の出力値を弱参照に変更した。

### ステップ18 メモリ使用量を減らすモード
* 不要な微分を保持しないモードや逆伝播の有効・無効の切り替えに対応した。
* モードの設定を管理する構造体を追加し、切り替えるメソッドを実装した。
* 設定値はスレッドローカルの static 変数とし、各処理で参照するように対応した。

### ステップ19 変数を使いやすく
* 書籍通り、変数に名前を付けられるように対応した。
* これまで ndarray の Array<f64, IxDyn> にのみ対応していたが、num_traits トレイトを追加して他の数値にも対応できるように変更した。
* 変数を生成する際は Array::from_elem(IxDyn(&[]), *self) のみだったが、from_shape_vec にも対応し、任意の次元の配列にも対応した。
* 変数について、size, ndim, dtype, shape を取得できるように対応した。

### ステップ20 演算子のオーバーロード (1)
* 書籍の内容ではないが、1つのファイルで開発するのが煩雑になったためプロジェクトの構成を見直し、モジュール分割した。
* 加算と乗算についてオーバーロードに対応した。
  * オーバーロードは順伝播となるため、入出力値や関数を保持する必要がある。
  * Rust の典型的なオーバーロードの実装例は状態を保持しない「関数」の例が多いが、今回は順伝播の状態を保持しなければならないため工夫が必要であった。
    * オーバーロードの実装として参照を受け付ける
    * Variable 構造体自体の参照共有や内部可変に対応するため、Variable を RawVariable に変更し、そのラッパーとなる構造体 Variable を追加して、Rc<RefCell<Variable>> として保持するように変更した。
    * 上記の対応のため、コードを全面的に修正した。

### ステップ21 演算子のオーバーロード (2)
* Variable 以外の数値や Array とのオーバーロードに対応した。
* 様々な数値やオペランド左右を入れ替えての対応はマクロを時使って実現している。

### ステップ22 演算子のオーバーロード (3)
* 負数 neg、引き算、割り算、累乗のオーバーロードに対応した。
* 演算に関するコードが巨大がしてきたため、個別のモジュールに分割した。

### ステップ23 パッケージとしてまとめる
* これまでのステップで構成を見直していたため書籍とは異なるが、現状のまま続けることとし、適宜見直す。

### ステップ24 複雑な関数の微分
* Sphere 関数、matyas 関数、Goldstein-Price 関数を追加した。
* 逆伝播実行時に、計算グラフ上の関数を世代の高い順に取得しなければならないが、BinaryHeap の格納順に取得してしまっていたため修正した。

### ステップ２５ 計算グラフの可視化 (1)
* graphviz のインストールと簡易なグラフ作成
