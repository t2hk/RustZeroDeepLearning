extern crate rust_zero_deeplearning;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use rust_zero_deeplearning::modules::core::data_loader::DataLoader;
use rust_zero_deeplearning::modules::*;
#[path = "common/mod.rs"]
mod common;

use std::cell::RefCell;
use std::rc::Rc;
use std::time;

use plotters::chart::ChartBuilder;
use plotters::prelude::BitMapBackend;
use plotters::series::LineSeries;
use plotters::style::{Color, IntoFont, BLACK, RED, WHITE};

use plotters::drawing::IntoDrawingArea;

// use approx::assert_abs_diff_eq;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_sample() {
        println!("This is a sample test.");
    }

    #[test]
    fn test_mnist() {
        // ハイパーパラメータの設定
        let max_epoch = 5;
        let batch_size = 100;
        let hidden_size = 1000;
        let lr = 1.0;

        let sigmoid = Rc::new(RefCell::new(SigmoidFunction {}));
        let sgd = Sgd::new(lr);
        let mut mlp = Mlp::new(vec![hidden_size, 10], sigmoid, sgd);

        // データの読み込み、モデル・オプティマイザの生成
        let mut mnist_train_set = Dataset::init(true, Rc::new(RefCell::new(MnistDataSet {})));
        // let train_data_loader = DataLoader::init(mnist_train_set.clone(), batch_size, true);
        let train_data_loader = DataLoader::init(&mnist_train_set, batch_size, true);

        let mut mnist_test_set = Dataset::init(false, Rc::new(RefCell::new(MnistDataSet {})));
        let test_data_loader = DataLoader::init(&mnist_test_set, batch_size, true);

        let mut loss_result = vec![];
        let mut acc_result = vec![];

        for epoch in 0..max_epoch {
            let mut sum_loss = 0.0;
            let mut sum_acc = 0.0;

            // 逆伝播を実行する。微分値を保持する。
            Setting::set_retain_grad_enabled();
            // バックプロパゲーションを行う。
            Setting::set_backprop_enabled();

            let mut batch_count = 0;
            for (batch_x, batch_t) in train_data_loader.clone() {
                batch_count += 1;
                println!(
                    "batch_x: {:?} epoch: {}  batch: {}",
                    batch_x.get_data().shape(),
                    epoch + 1,
                    batch_count
                );
                // 勾配の算出、パラメータの更新
                let y = mlp.forward(vec![batch_x.clone()]);

                let loss = softmax_cross_entropy(y[0].clone(), batch_t.clone());
                let acc = DataLoader::accuracy(&y[0].clone(), &batch_t);

                mlp.cleargrads();

                loss.backward();

                mlp.update_parameters();

                let batch_t_len = batch_t.get_data().flatten().to_vec().len() as f64;
                sum_loss += loss.get_data().flatten().to_vec()[0] as f64 * batch_t_len;

                sum_acc += acc * batch_t_len;

                let file_name = format!(
                    "test_step51_mnist_dataloader_epoch-{}_batch-{}.png",
                    epoch, batch_count
                );

                // let tmp = y[0].clone();
                // plot_dot_graph!(tmp, file_name, true);
            }

            let train_len = mnist_train_set.clone().len() as f64;
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

            let start_now = time::Instant::now();
            for (batch_x, batch_t) in test_data_loader.clone() {
                // 勾配の算出、パラメータの更新
                let y = mlp.forward(vec![batch_x.clone()]);
                let loss = softmax_cross_entropy(y[0].clone(), batch_t.clone());
                let acc = DataLoader::accuracy(&y[0], &batch_t);

                let batch_t_len = batch_t.get_data().flatten().to_vec().len() as f64;
                sum_loss += loss.get_data().flatten().to_vec()[0] as f64 * batch_t_len;

                sum_acc += acc * batch_t_len;
                println!("   test one data end : {:?}", start_now.elapsed());
            }
            println!("test all data end : {:?}", start_now.elapsed());

            let test_len = mnist_test_set.clone().len() as f64;
            let avg_loss = sum_loss / test_len;
            let avg_acc = sum_acc / test_len;
            println!("test loss: {}, accuracy: {}", avg_loss, avg_acc);

            loss_result.push(avg_loss);
            acc_result.push(avg_acc);
        }

        // 損失結果グラフ描画
        // 描画先の Backend を初期化する。
        let root =
            BitMapBackend::new("graph/step51_mnist_test_loss.png", (640, 480)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        // グラフの軸の設定など
        let mut chart = ChartBuilder::on(&root)
            .caption("MNIST data test loss", ("sans-serif", 50).into_font())
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
            BitMapBackend::new("graph/step51_mnist_test_acc.png", (640, 480)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        // グラフの軸の設定など
        let mut chart = ChartBuilder::on(&root)
            .caption("MNIST data test accuracy", ("sans-serif", 50).into_font())
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
