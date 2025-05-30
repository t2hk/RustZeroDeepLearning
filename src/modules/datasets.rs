// ライブラリを一括でインポート
use crate::modules::*;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{s, Array, IxDyn};
use ndarray_rand::rand_distr::Normal;
use std::{cell::RefCell, rc::Rc};

use plotters::{
    chart::ChartBuilder,
    prelude::{BitMapBackend, Circle, Cross, IntoDrawingArea, IntoDynElement, TriangleMarker},
    series::PointSeries,
    style::{Color, IntoFont, BLACK, BLUE, GREEN, RED, WHITE},
};
use rand::{prelude::Distribution, rngs::StdRng, seq::SliceRandom, SeedableRng};

/// データローダー用のトレイト
pub trait DataLoader: std::fmt::Debug {
    /// データを読み込む。
    ///
    /// Arguments:
    /// * train (bool): 学習データの場合 true、テストデータの場合 false
    ///
    /// Retrun
    /// * (Array<f64, IxDyn>, Array<usize, IxDyn>): スパイラルデータと正解ラベル
    fn prepare(&self, train: bool) -> (Array<f64, IxDyn>, Array<usize, IxDyn>);
}

/// データセット構造体
#[derive(Debug, Clone)]
pub struct Dataset {
    train: bool, // 訓練データの場合 true、テストデータの場合 false
    data_loader: Rc<RefCell<dyn DataLoader>>, // データローダのトレイトオブジェクト
    data: Option<Array<f64, IxDyn>>, // データ
    label: Option<Array<usize, IxDyn>>, // 正解ラベル
}

impl Dataset {
    /// データセットの初期化
    ///
    /// Arguments:
    /// * train (bool): 訓練データの場合 true、テストデータの場合 false
    /// * batch_size (usize): バッチサイズ
    /// * shuffle (bool): シャッフルするかどうか
    /// * data_loader (Rc<RefCell<dyn DataLoader>>): データローダ用のトレイトオブジェクト
    ///
    /// Return:
    /// * Dataset: データセット構造体
    pub fn init(train: bool, data_loader: Rc<RefCell<dyn DataLoader>>) -> Dataset {
        let mut ds = Dataset {
            train: train,
            data_loader: data_loader,
            data: None,
            label: None,
        };
        ds.prepare();
        ds
    }

    /// データを読み込む。
    ///
    /// Arguments:
    /// * train (bool): 学習データの場合 true、テストデータの場合 false
    ///
    /// Retrun
    /// * (Array<f64, IxDyn>, Array<usize, IxDyn>): スパイラルデータと正解ラベル
    pub fn prepare(&mut self) {
        let (x, t) = self.data_loader.borrow().prepare(self.train);
        self.data = Some(x);
        self.label = Some(t);
    }

    /// 対象データの長さを取得する。
    ///
    /// Retrun
    /// * usize: 長さ
    pub fn len(self) -> usize {
        if let Some(data) = self.data {
            // data.len()
            data.shape().to_vec()[0]
        } else {
            panic!("Data does not exist.");
        }
    }

    /// 指定したインデックスのデータと正解ラベルを取得する。
    ///
    /// Arguments:
    /// * index (usize): インデックス
    ///
    /// Retrun:
    /// * (Array<f64, IxDyn>, Option<usize>): 対象データと正解ラベル
    pub fn get(self, index: usize) -> (Array<f64, IxDyn>, Option<usize>) {
        if let Some(data) = self.data {
            let index_row = data.slice(s![index, ..]).into_owned().into_dyn();

            if let Some(label) = self.label {
                let index_label = label.flatten().to_vec()[index];
                (index_row, Some(index_label))
            } else {
                (index_row, None)
            }
        } else {
            panic!("Data does not exist.");
        }
    }

    /// バッチ取得
    /// 指定したインデックスのデータ群を取得する。
    ///
    /// Arguments:
    /// * batch_index (&[usize]): バッチ取得するインデックス
    ///
    /// Retrun:
    /// * (Variable<f64>, Variable<usize>): インデックスで指定したデータと正解ラベル
    pub fn get_batch(&self, batch_index: &[usize]) -> (Variable<f64>, Variable<usize>) {
        let mut batch_x_vec = vec![];
        let mut batch_t_vec = vec![];

        for idx in batch_index.iter() {
            let idx_train_data = self.clone().get(*idx);
            batch_x_vec.extend(idx_train_data.0.flatten().to_vec());
            batch_t_vec.push(idx_train_data.1.unwrap());
        }
        let batch_x = Variable::new(RawData::from_shape_vec(
            vec![batch_index.len(), 2],
            batch_x_vec,
        ));
        let batch_t = Variable::new(RawData::from_shape_vec(
            vec![1, batch_index.len()],
            batch_t_vec.clone(),
        ));
        (batch_x, batch_t)
    }
}

/// スパイラルデータセット用のデータローダー
#[derive(Debug, Clone)]
pub struct SpiralDataSet;
impl DataLoader for SpiralDataSet {
    /// データを読み込む。
    ///
    /// Arguments:
    /// * train (bool): 学習データの場合 true、テストデータの場合 false
    ///
    /// Retrun
    /// * (Array<f64, IxDyn>, Array<usize, IxDyn>): スパイラルデータと正解ラベル
    fn prepare(&self, train: bool) -> (Array<f64, IxDyn>, Array<usize, IxDyn>) {
        get_spiral(train)
    }
}

/// スパイラルデータセット
///
/// Arguments:
/// * train (bool): true の場合は学習用データ、false の場合はテスト用データ
///
/// Retrun:
/// (Array<f64, IxDyn>, Array<usize, IxDyn>): (入力データ、教師ラベル)
pub fn get_spiral(train: bool) -> (Array<f64, IxDyn>, Array<usize, IxDyn>) {
    let seed = if train { 1984 } else { 2020 };

    let mut rng = StdRng::seed_from_u64(seed);

    let num_data = 100;
    let num_class = 3;
    let input_dim = 2;
    let data_size = num_class * num_data;

    let mut x: Array<f64, _> = Array::zeros((data_size, input_dim));
    let mut t: Array<usize, _> = Array::zeros((data_size));

    for j in 0..num_class {
        for i in 0..num_data {
            //  rate = i / num_data
            //  radius = 1.0 * rate
            //  theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            //  ix = num_data * j + i
            //  x[ix] = np.array([radius * np.sin(theta),
            //                    radius * np.cos(theta)]).flatten()
            //  t[ix] = j
            let rate = i as f64 / num_data as f64;
            let radius = 1.0 * rate;
            let theta =
                j as f64 * 4.0 + 4.0 * rate + Normal::new(0.0, 0.2).unwrap().sample(&mut rng);
            let ix = num_data * j + i;

            x[[ix, 0]] = radius * theta.sin();
            x[[ix, 1]] = radius * theta.cos();
            t[ix] = j;
        }
    }
    let mut indices: Vec<usize> = (0..data_size).collect();
    indices.shuffle(&mut rng);

    let mut x_shuffled: Array<f64, _> = Array::zeros((data_size, input_dim));
    let mut t_shuffled: Array<usize, _> = Array::zeros(data_size);

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        x_shuffled.row_mut(new_idx).assign(&x.row(old_idx));
        t_shuffled[new_idx] = t[old_idx];
    }

    (x_shuffled.into_dyn(), t_shuffled.into_dyn())
}

/// スパイラルデータをグラフ描画する。
pub fn draw_spiral_graph(train: bool) {
    let (x, t) = get_spiral(train);

    let file_path = if train {
        "graph/spiral_data_train.png"
    } else {
        "graph/spiral_data_test.png"
    };
    let caption = "spiral";

    // グラフ描画
    // 描画先の Backend を初期化する。
    let root = BitMapBackend::new(&file_path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // グラフの軸の設定など
    let mut chart = ChartBuilder::on(&root)
        .caption(&caption, ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1.0..1.0, -1.0..1.0)
        .unwrap();
    chart.configure_mesh().draw().unwrap();

    // データ点を生成し、値に応じて異なるマーカーを使用
    chart
        .draw_series((0..300).map(|i| {
            let x_x = x[[i, 0]];
            let x_y = x[[i, 1]];

            let point = (x_x, x_y);

            let t_var = t[i];

            match t_var {
                0 => Circle::new(point, 4, GREEN.filled()).into_dyn(),
                1 => TriangleMarker::new(point, 4, BLUE.filled()).into_dyn(),
                _ => Cross::new(point, 4, RED.filled()).into_dyn(),
            }
        }))
        .unwrap();

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
}

#[cfg(test)]
mod test {
    use std::{cell::RefCell, rc::Rc};

    use rand_isaac::Isaac64Rng;

    use super::*;

    #[test]
    fn test_get_spiral() {
        let (x, t) = get_spiral(true);

        println!("x shape: {:?}", x.shape());
        println!("t shape: {:?}", t.shape());

        println!("x[10]: ({}, {}), t[10]: {}", x[[10, 0]], x[[10, 1]], t[10]);
        println!(
            "x[110]: ({}, {}), t[110]: {}",
            x[[110, 0]],
            x[[110, 1]],
            t[110]
        );

        draw_spiral_graph(true);
        draw_spiral_graph(false);
    }

    /// Dataset を使った学習
    #[test]
    fn test_step49_spiral_dataset() {
        // ハイパーパラメータの設定
        let max_epoch = 300;
        let batch_size = 30;
        let hidden_size = 10;
        let lr = 1.0;

        // データの読み込み、モデル・オプティマイザの生成
        // let (x, t) = datasets::get_spiral(true);
        //let spiral_train_set = SpiralDataSet::init(true);
        let mut spiral_train_set = Dataset::init(true, Rc::new(RefCell::new(SpiralDataSet {})));
        spiral_train_set.prepare();

        let data_size = spiral_train_set.clone().len();
        let sigmoid = Rc::new(RefCell::new(SigmoidFunction {}));
        let sgd = Sgd::new(lr);
        let mut mlp = Mlp::new(vec![hidden_size, 3], sigmoid, sgd);

        // let data_size = x.shape().to_vec()[0];
        let max_iter = data_size.div_ceil(batch_size);
        println!(
            "max_epoch:{}, bath_size:{}, hidden_size:{}, data_size:{}, max_iter:{}",
            max_epoch, batch_size, hidden_size, data_size, max_iter
        );

        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let mut loss_result = vec![];

        for epoch in 0..max_epoch {
            let mut index: Vec<usize> = (0..data_size).collect();
            index.shuffle(&mut rng);

            let mut sum_loss = 0.0;

            for i in 0..max_iter {
                // ミニバッチの作成
                let mini_batch_from = i * batch_size;
                let mini_batch_to = (i + 1) * batch_size;
                let batch_index = &index[mini_batch_from..mini_batch_to];

                let (batch_x, batch_t) = spiral_train_set.get_batch(batch_index);

                // 勾配の算出、パラメータの更新
                let y = mlp.forward(vec![batch_x.clone()]);
                let loss = softmax_cross_entropy(y[0].clone(), batch_t.clone());
                mlp.cleargrads();

                loss.backward();
                mlp.update_parameters();

                sum_loss += loss.get_data().flatten().to_vec()[0] as f64
                    * batch_t.get_data().flatten().to_vec().len() as f64;
            }
            let avg_loss = sum_loss / data_size as f64;
            println!("epoch {}, loss: {}", epoch + 1, avg_loss);
            loss_result.push(avg_loss);
        }
    }
}
