extern crate rust_zero_deeplearning;

#[path = "common/mod.rs"]
mod common;

use rust_zero_deeplearning::modules::utils::*;
use rust_zero_deeplearning::modules::*;
use rust_zero_deeplearning::*;
// use approx::assert_abs_diff_eq;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};

#[test]
fn test_step33_second_differential() {
    // common::setup();

    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_enabled();
    // バックプロパゲーションを行う。
    Setting::set_backprop_enabled();

    /// y = x^4 - 2x^2
    fn f<V: MathOps>(x: &Variable<V>) -> Variable<V> {
        let y = &(x ^ 4) - &(2 * &(x ^ 2));
        y
    }

    let mut x = Variable::new(RawData::new(2.0));
    x.set_name("x".to_string());

    debug!("===== フォワード ======");

    let y = f(&x);
    let expected_y = Array::from_elem(IxDyn(&[]), 8.0);
    assert_eq!(expected_y, y.get_data());

    debug!("===== 1回目バックプロパゲーション======");
    y.backward();

    let expected_x_grad = Array::from_elem(IxDyn(&[]), 24.0);
    assert_eq!(expected_x_grad, x.get_grad().unwrap().get_data());

    // バックプロパゲーションを行わないモードに切り替え。
    Setting::set_backprop_disabled();
    debug!("===== 2回目バックプロパゲーション======");
    let gx = &x.get_grad().unwrap();
    let expected_2nd_grad = Array::from_elem(IxDyn(&[]), 44.0);
    x.clear_grad();
    gx.backward();
    let x_2nd_grad = x.get_grad().unwrap().get_data().clone();
    debug!("x 2nd grad: {:?}", x_2nd_grad);
    assert_eq!(expected_2nd_grad, x_2nd_grad);
}

/// ステップ33 ニュートン法による最適化のテスト(自動化)
#[test]
fn test_step33_newton_method() {
    // common::setup();

    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_enabled();
    // バックプロパゲーションを行う。
    Setting::set_backprop_enabled();

    /// y = x^4 - 2x^2
    fn f<V: MathOps>(x: &Variable<V>) -> Variable<V> {
        let y = &(x ^ 4) - &(2 * &(x ^ 2));
        y
    }

    let mut x = Variable::new(RawData::new(2.0));
    let mut results: Vec<f64> = vec![];
    let iters = 10;
    for i in 0..iters {
        debug!("i:{}, x:{:?}", i, x.get_data()[[]]);
        results.push(x.get_data()[[]]);
        let y = f(&x);
        x.clear_grad();
        y.backward();

        let gx = &x.get_grad().unwrap();
        x.clear_grad();
        gx.backward();
        let gx2 = x.get_grad().unwrap();

        let new_x_data = x.get_data() - (gx.get_data() / gx2.get_data());
        x.set_data(new_x_data);
    }
    debug!("results: {:?}", results);
    // 書籍と同じ値になることを確認する。
    assert_eq!(2.0, results[0]);
    assert_eq!(1.4545454545454546, results[1]);
    assert_eq!(1.1510467893775467, results[2]);
    assert_eq!(1.0253259289766978, results[3]);
    assert_eq!(1.0009084519430513, results[4]);
    assert_eq!(1.0000012353089454, results[5]);
    assert_eq!(1.000000000002289, results[6]);
    assert_eq!(1.0, results[7]);
    assert_eq!(1.0, results[8]);
    assert_eq!(1.0, results[9]);
}

/// Sin 関数の高階微分のテスト
#[test]
fn test_high_diffeential_sin() {
    // common::setup();

    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_enabled();
    // バックプロパゲーションを行う。
    Setting::set_backprop_enabled();

    let x = Variable::new(RawData::new(1.0));
    let y = sin(x.clone());
    y.backward();

    let mut results: Vec<f64> = vec![];
    for _i in 0..3 {
        let gx = &x.get_grad().unwrap();
        x.clear_grad();
        gx.backward();
        results.push(x.get_grad().unwrap().get_data()[[]]);
    }
    assert_eq!(-0.8414709848078965, results[0]);
    assert_eq!(-0.5403023058681398, results[1]);
    assert_eq!(0.8414709848078965, results[2]);
}

/// Sin 高階微分グラフ
#[test]
fn test_step34_graph() {
    use plotters::prelude::*;
    // common::setup();

    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_enabled();
    // バックプロパゲーションを行う。
    Setting::set_backprop_enabled();

    let mut logs: Vec<Vec<f64>> = vec![];

    // y = sin(x) の x 範囲
    let start = -7.0;
    let end = 7.0;
    let x = Variable::new(RawData::linspace(start, end, 200));
    let y = sin(x.clone());
    y.backward();
    let y_data = y.get_data();
    logs.push(y_data.flatten().to_vec());

    // 3階微分まで実行
    for _i in 0..3 {
        logs.push(x.get_grad().unwrap().get_data().flatten().to_vec());
        let gx = x.get_grad().unwrap();
        x.clear_grad();
        gx.backward();
    }

    // 描画先の Backend を初期化する。
    let root = BitMapBackend::new("graph/step34_sin.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // グラフの軸の設定など
    let mut chart = ChartBuilder::on(&root)
        .caption("y=sin(x)", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(start as f32..end as f32, -1.5f32..1.5f32)
        .unwrap();
    chart.configure_mesh().draw().unwrap();

    let labels = vec!["y=sin(x)", "y'", "y''", "y'''"];
    let colors = vec![RED, BLUE, GREEN, MAGENTA];

    let mut plot_data_vec = vec![];

    for log in logs {
        let values = log.clone();
        let plot_data: Vec<(f32, f32)> = values
            .iter()
            .enumerate()
            .map(|(i, j)| (x.get_data().flatten().to_vec()[i] as f32, *j as f32))
            .collect();
        plot_data_vec.push(plot_data);
    }

    // データの描画。(x, y)のイテレータとしてデータ点を渡す
    for (idx, plot_data) in plot_data_vec.iter().enumerate() {
        let color = colors[idx];

        chart
            .draw_series(LineSeries::new(
                //(-50..50).map(|x| x as f32 / 50.0).map(|x| (x, x * x)),
                plot_data.clone(),
                &color,
            ))
            .unwrap()
            .label(labels[idx])
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
}

/// tanh 高階微分
#[test]
fn test_step34_tanh() {
    use plotters::prelude::*;
    // common::setup();

    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_enabled();
    // バックプロパゲーションを行う。
    Setting::set_backprop_enabled();

    let mut logs: Vec<Vec<f64>> = vec![];

    // y = tanh(x)
    let x = Variable::new(RawData::new(1.0));
    x.set_name("x".to_string());
    let y = tanh(x.clone());
    y.set_name("y".to_string());
    y.backward();

    let iters = 2;

    for _i in 0..iters {
        let gx = x.get_grad().unwrap();

        x.clear_grad();
        gx.backward();
    }

    let gx = x.get_grad().unwrap();
    let current = iters + 1;
    gx.set_name(format!("gx{}", current).to_string());
    let file_name = format!("test_step35_tanh_{}.png", current);
    plot_dot_graph!(gx, file_name, true);
}

#[test]
fn tests_stage36_double_backprop() {
    // common::setup();

    // 逆伝播を実行する。微分値を保持しない。
    // Setting::set_retain_grad_enabled();
    Setting::set_retain_grad_disabled();
    // バックプロパゲーションを行う。
    Setting::set_backprop_enabled();

    info!("===== y = x^2");
    // y = x ^2
    let mut x = Variable::new(RawData::new(2.0));
    x.set_name("x".to_string());
    let y = &x ^ 2;
    y.set_name("y".to_string());
    println!("y = x^2 -> y:{:?}", y.get_data()[[]]);

    let dummy = Variable::new(RawData::new(-999.0));

    info!("===== y backward");

    y.backward();
    let gx = x.get_grad().unwrap();
    gx.set_name("gx".to_string());
    println!(
        "y backward x grad:{:?}",
        x.get_grad().unwrap().get_data()[[]]
    );

    info!("===== x clear grad");
    x.clear_grad();

    // z = gx^3 + y
    let z = &(&gx ^ 3) + &y;
    z.set_name("z".to_string());
    info!(
        "z = gx^3 + y -> z:{:?}, gx:{:?}, y:{:?}",
        z.get_data()[[]],
        gx.get_data()[[]],
        y.get_data()[[]]
    );

    info!("===== z backward");
    z.backward();

    let x_grad = x.get_grad().unwrap().get_data()[[]];
    info!("x grad: {:?}", x_grad);

    assert_eq!(100.0, x_grad);

    let file_name = "test_step36_backprop.png";
    let x_grad = x.get_grad().unwrap();
    plot_dot_graph!(x_grad, file_name, true);
}
