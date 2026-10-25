//! MNIST の手書き数字を MLP で学習・推論するサンプル

// ライブラリを一括でインポート
use rust_zero_deeplearning::*;

use ndarray::{Array, Array2, Axis, IxDyn};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

/// MNIST データの格納ディレクトリ
fn mnist_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mnist")
}

/// idx 形式のファイルを読み込む。
///
/// Arguments:
/// * path (&Path): ファイルのパス
///
/// Return:
/// * (Vec<usize>, Vec<u8>): 各次元の要素数とデータ本体
fn read_idx(path: &Path) -> std::io::Result<(Vec<usize>, Vec<u8>)> {
    let bytes = std::fs::read(path)?;
    // マジックナンバー: 0x00 0x00 [型(0x08 = u8)] [次元数]
    assert_eq!(0x08, bytes[2], "unsupported idx data type: {:?}", path);
    let ndim = bytes[3] as usize;
    let dims = (0..ndim)
        .map(|i| u32::from_be_bytes(bytes[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize)
        .collect();
    let body = bytes[4 + 4 * ndim..].to_vec();
    Ok((dims, body))
}

/// MNIST を読み込む。
/// 画像は 0.0〜1.0 に正規化し、(枚数, 784) の形状にする。
///
/// Arguments:
/// * train (bool): 訓練データの場合 true、テストデータの場合 false
///
/// Return:
/// * (Array2<f64>, Vec<usize>): 画像と正解ラベル
fn load_mnist(train: bool) -> std::io::Result<(Array2<f64>, Vec<usize>)> {
    let prefix = if train { "train" } else { "t10k" };
    let dir = mnist_dir();

    let (img_dims, img_body) = read_idx(&dir.join(format!("{}-images-idx3-ubyte", prefix)))?;
    let (_, label_body) = read_idx(&dir.join(format!("{}-labels-idx1-ubyte", prefix)))?;

    let n = img_dims[0];
    let pixels = img_dims[1] * img_dims[2];
    let images = Array2::from_shape_vec(
        (n, pixels),
        img_body.iter().map(|&p| p as f64 / 255.0).collect(),
    )
    .unwrap();
    let labels = label_body.iter().map(|&l| l as usize).collect();

    Ok((images, labels))
}

/// MNIST のファイルが揃っているかどうか。
///
/// Return:
/// * bool: 全てのファイルが存在する場合 true
fn mnist_exists() -> bool {
    [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    .iter()
    .all(|f| mnist_dir().join(f).exists())
}

/// MNIST 用のデータソース
#[derive(Debug)]
struct MnistDataSet;

impl DataSource for MnistDataSet {
    /// データを読み込む。
    ///
    /// Arguments:
    /// * train (bool): 訓練データの場合 true、テストデータの場合 false
    ///
    /// Return:
    /// * (Array<f64, IxDyn>, Array<usize, IxDyn>): 画像 (枚数, 784) と正解ラベル (枚数)
    fn prepare(&self, train: bool) -> (Array<f64, IxDyn>, Array<usize, IxDyn>) {
        let (images, labels) = load_mnist(train).expect("MNIST の読み込みに失敗しました。");
        (images.into_dyn(), Array::from_vec(labels).into_dyn())
    }
}

/// 推論結果(各クラスのスコア)から、最もスコアの高いクラスを求める。
///
/// Arguments:
/// * y (&Variable<f64>): 推論結果 (バッチサイズ, クラス数)
///
/// Return:
/// * Vec<usize>: 推論したクラス
fn argmax(y: &Variable<f64>) -> Vec<usize> {
    y.get_data()
        .axis_iter(Axis(0))
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        })
        .collect()
}

/// MNIST の手書き数字を MLP で学習し、テストデータで推論する。
/// 事前に data/mnist に MNIST の idx ファイル(解凍済み)を配置しておくこと。
pub fn run() {
    println!("===== MNIST =====");

    // 大きな配列をログ出力すると非常に遅くなるため、警告以上のみ出力する。
    log::set_max_level(log::LevelFilter::Warn);

    // ハイパーパラメータの設定
    let max_epoch = 5;
    let batch_size = 100;
    let hidden_size = 100;
    let lr = 1.0;

    // データの読み込み
    if !mnist_exists() {
        println!(
            "MNIST が見つかりません。{:?} に以下のファイルを配置してください。",
            mnist_dir()
        );
        println!("  train-images-idx3-ubyte, train-labels-idx1-ubyte,");
        println!("  t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte");
        println!("  (https://ossci-datasets.s3.amazonaws.com/mnist/ から取得して gunzip)");
        return;
    }
    let train_set = Dataset::init(true, Rc::new(RefCell::new(MnistDataSet)));
    let test_set = Dataset::init(false, Rc::new(RefCell::new(MnistDataSet)));
    let train_len = train_set.len();
    let test_len = test_set.len();
    println!("train: {}, test: {}", train_len, test_len);

    // データセットは大きいため clone せず、所有権ごとデータローダーに渡す。
    let mut train_loader = DataLoader::init(train_set, batch_size, true);
    let mut test_loader = DataLoader::init(test_set, batch_size, false);

    // モデル・オプティマイザの生成
    let sigmoid = Rc::new(RefCell::new(SigmoidFunction {}));
    let sgd = Sgd::new(lr);
    let mut mlp = Mlp::new(vec![hidden_size, 10], sigmoid, sgd);

    Setting::set_retain_grad_disabled();
    Setting::set_backprop_enabled();

    // 学習
    for epoch in 0..max_epoch {
        let start = Instant::now();
        train_loader.reset();

        let mut sum_loss = 0.0;
        let mut sum_acc = 0.0;

        for (batch_x, batch_t) in &mut train_loader {
            let y = mlp.forward(vec![batch_x]);
            let loss = softmax_cross_entropy(y[0].clone(), batch_t.clone());
            mlp.cleargrads();
            loss.backward();
            mlp.update_parameters();

            let n = batch_t.get_data().len() as f64;
            sum_loss += loss.get_data()[[]] * n;
            sum_acc += DataLoader::accuracy(&y[0], &batch_t) * n;
        }

        println!(
            "epoch: {}, train loss: {:.4}, accuracy: {:.4} ({:.1}s)",
            epoch + 1,
            sum_loss / train_len as f64,
            sum_acc / train_len as f64,
            start.elapsed().as_secs_f64()
        );
    }

    // 推論 (逆伝播しないため、計算グラフを作成しない)
    let _guard = NoGradGuard::new();
    test_loader.reset();

    let mut sum_acc = 0.0;
    // 推論結果の表示用に、最初のバッチ (test[0..batch_size]) の画像・正解・推論を保持する。
    let mut samples = None;

    for (batch_x, batch_t) in &mut test_loader {
        let y = mlp.forward(vec![batch_x.clone()]);
        let n = batch_t.get_data().len() as f64;
        sum_acc += DataLoader::accuracy(&y[0], &batch_t) * n;

        if samples.is_none() {
            samples = Some((batch_x.get_data(), batch_t.get_data(), argmax(&y[0])));
        }
    }

    let correct = sum_acc.round() as usize;
    println!(
        "test accuracy: {:.4} ({}/{})",
        sum_acc / test_len as f64,
        correct,
        test_len
    );

    // 推論結果の例を表示する。
    let (x, t, predicts) = samples.expect("テストデータがありません。");
    for i in 0..3 {
        println!("--- test[{}] 正解: {}, 推論: {}", i, t[[0, i]], predicts[i]);
        for row in x.index_axis(Axis(0), i).iter().copied().collect::<Vec<f64>>().chunks(28) {
            let line: String = row
                .iter()
                .map(|&p| match p {
                    p if p > 0.66 => '#',
                    p if p > 0.33 => '+',
                    _ => ' ',
                })
                .collect();
            println!("{}", line.trim_end());
        }
    }
}
