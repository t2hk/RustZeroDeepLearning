// ライブラリを一括でインポート
use crate::modules::*;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::Axis;
use rand::{seq::SliceRandom, thread_rng};

/// データローダー
#[derive(Debug, Clone)]
pub struct DataLoader {
    dataset: Dataset,
    batch_size: usize,
    shuffle: bool,
    data_size: usize,
    max_iter: usize,
    iteration: usize,
    index: Vec<usize>,
}

impl DataLoader {
    /// 初期化
    ///
    /// Arguments:
    /// * dataset (Dataset): データセット
    /// * batch_size (usize): バッチサイズ
    /// * shuffle (bool): データセットをシャッフルするかどうか
    pub fn init(dataset: Dataset, batch_size: usize, shuffle: bool) -> Self {
        let data_size = dataset.clone().len();

        let index = {
            let mut vec = (0..data_size).collect::<Vec<usize>>();
            if shuffle {
                vec.shuffle(&mut thread_rng());
            }
            vec
        };

        Self {
            dataset: dataset,
            batch_size: batch_size,
            shuffle: shuffle,
            data_size: data_size,
            max_iter: data_size.div_ceil(batch_size),
            iteration: 0,
            index: index,
        }
    }

    /// 正解率
    ///
    /// Arguments:
    /// * y (&Variable<f64>): 推測
    /// * t (&Variable<usize>): 正解ラベル
    ///
    /// Return:
    /// * f64: 正解率
    pub fn accuracy(y: &Variable<f64>, t: &Variable<usize>) -> f64 {
        let pred: Vec<usize> = y
            .get_data()
            .axis_iter(Axis(0))
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap()
                    .0
            })
            .collect();

        let mut result = 0.0;

        let t_var = t.get_data().flatten().to_vec();

        for i in 0..pred.len() {
            if pred.get(i) == t_var.get(i) {
                result += 1.0;
            }
        }

        let acc = result / pred.len() as f64;
        acc
    }
}

impl Iterator for DataLoader {
    type Item = (Variable<f64>, Variable<usize>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.iteration >= self.max_iter {
            return None;
        }

        let batch_start = self.iteration * self.batch_size;
        let batch_end = self.batch_size * (self.iteration + 1);
        let batch_index = &self.index[batch_start..batch_end];
        let batch = self.dataset.get_batch(batch_index);
        self.iteration += 1;

        Some(batch)
    }
}

#[cfg(test)]
mod test {
    use std::{cell::RefCell, rc::Rc};

    use ndarray::Array;
    use plotters::{
        chart::ChartBuilder,
        prelude::{BitMapBackend, IntoDrawingArea},
        series::LineSeries,
        style::{Color, IntoFont, BLACK, RED, WHITE},
    };
    use rand::seq::SliceRandom;

    use crate::modules::core::data_loader::DataLoader;

    use super::*;

    #[test]
    fn test_hgoe() {
        let sds = SpiralDataSet {};
        let ds = Dataset::init(false, Rc::new(RefCell::new(sds)));

        let dl = DataLoader::init(ds, 10, false);
        for data in dl {
            dbg!(data);
        }
    }

    #[test]
    fn test_array() {
        let x = Array::from_vec(vec![0, 1, 2, 3, 4, 5, 6]);
        let y = Array::from_vec(vec![0, 1, 2, 3, 4, 1, 2]);

        let mut result = 0.0;

        for i in 0..x.len() {
            if x.get(i) == y.get(i) {
                result += 1.0;
            }
        }

        println!("acc: {}", result / x.len() as f64);
    }

    /// DataLoader を使った学習
    #[test]
    fn test_step50_spiral_dataloader() {
        // ハイパーパラメータの設定
        let max_epoch = 300;
        let batch_size = 30;
        let hidden_size = 10;
        let lr = 1.0;

        let sigmoid = Rc::new(RefCell::new(SigmoidFunction {}));
        let sgd = Sgd::new(lr);
        let mut mlp = Mlp::new(vec![hidden_size, 3], sigmoid, sgd);

        // データの読み込み、モデル・オプティマイザの生成
        // let (x, t) = datasets::get_spiral(true);
        //let spiral_train_set = SpiralDataSet::init(true);
        let mut spiral_train_set = Dataset::init(true, Rc::new(RefCell::new(SpiralDataSet {})));
        let train_data_loader = DataLoader::init(spiral_train_set.clone(), batch_size, true);

        let mut spiral_test_set = Dataset::init(false, Rc::new(RefCell::new(SpiralDataSet {})));
        let test_data_loader = DataLoader::init(spiral_test_set.clone(), batch_size, true);

        let mut loss_result = vec![];
        let mut acc_result = vec![];

        for epoch in 0..max_epoch {
            let mut sum_loss = 0.0;
            let mut sum_acc = 0.0;

            // 逆伝播を実行する。微分値を保持する。
            Setting::set_retain_grad_enabled();
            // バックプロパゲーションを行う。
            Setting::set_backprop_enabled();

            for (batch_x, batch_t) in train_data_loader.clone() {
                // 勾配の算出、パラメータの更新
                let y = mlp.forward(vec![batch_x.clone()]);
                let loss = softmax_cross_entropy(y[0].clone(), batch_t.clone());
                let acc = DataLoader::accuracy(&y[0], &batch_t);
                mlp.cleargrads();

                loss.backward();
                mlp.update_parameters();

                let batch_t_len = batch_t.get_data().flatten().to_vec().len() as f64;
                sum_loss += loss.get_data().flatten().to_vec()[0] as f64 * batch_t_len;

                sum_acc += acc * batch_t_len;
            }

            let train_len = spiral_train_set.clone().len() as f64;
            println!("epoch: {}", epoch + 1);
            println!(
                "train loss: {}, accuracy: {}",
                sum_loss / train_len,
                sum_acc / train_len
            );

            // 逆伝播を実行しない。微分値を保持しない。
            Setting::set_retain_grad_disabled();
            // バックプロパゲーションを行わない。
            Setting::set_backprop_disabled();

            let mut sum_loss = 0.0;
            let mut sum_acc = 0.0;

            for (batch_x, batch_t) in test_data_loader.clone() {
                // 勾配の算出、パラメータの更新
                let y = mlp.forward(vec![batch_x.clone()]);
                let loss = softmax_cross_entropy(y[0].clone(), batch_t.clone());
                let acc = DataLoader::accuracy(&y[0], &batch_t);

                let batch_t_len = batch_t.get_data().flatten().to_vec().len() as f64;
                sum_loss += loss.get_data().flatten().to_vec()[0] as f64 * batch_t_len;

                sum_acc += acc * batch_t_len;
            }

            let test_len = spiral_test_set.clone().len() as f64;
            let avg_loss = sum_loss / test_len;
            let avg_acc = sum_acc / test_len;
            println!("test loss: {}, accuracy: {}", avg_loss, avg_acc);

            loss_result.push(avg_loss);
            acc_result.push(avg_acc);
        }

        // 損失結果グラフ描画
        // 描画先の Backend を初期化する。
        let root =
            BitMapBackend::new("graph/step50_spiral_test_loss.png", (640, 480)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        // グラフの軸の設定など
        let mut chart = ChartBuilder::on(&root)
            .caption("Spiral data test loss", ("sans-serif", 50).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0..300, 0.0..1.5)
            .unwrap();
        chart.configure_mesh().draw().unwrap();

        // 損失の描画
        chart
            .draw_series(LineSeries::new(
                (0..loss_result.len()).map(|x| (x as i32, loss_result[x] as f64)),
                &RED,
            ))
            .unwrap();
        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .unwrap();

        // 認識制度グラフ描画
        // 描画先の Backend を初期化する。
        let root =
            BitMapBackend::new("graph/step50_spiral_test_acc.png", (640, 480)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        // グラフの軸の設定など
        let mut chart = ChartBuilder::on(&root)
            .caption("Spiral data test accuracy", ("sans-serif", 50).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0..300, 0.0..1.5)
            .unwrap();
        chart.configure_mesh().draw().unwrap();

        // 損失の描画
        chart
            .draw_series(LineSeries::new(
                (0..acc_result.len()).map(|x| (x as i32, acc_result[x] as f64)),
                &RED,
            ))
            .unwrap();
        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .unwrap();
    }
}
