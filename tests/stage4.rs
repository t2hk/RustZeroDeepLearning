extern crate rust_zero_deeplearning;

#[path = "common/mod.rs"]
mod common;

use std::cell::RefCell;
use std::f64::consts::PI;
use std::rc::Rc;

use ndarray_rand::RandomExt;
use num_traits::float::TotalOrder;
use plotters::chart::ChartBuilder;
use plotters::prelude::{
    BitMapBackend, Circle, Cross, EmptyElement, IntoDrawingArea, IntoDynElement, PathElement,
    TriangleMarker,
};
use plotters::series::{LineSeries, PointSeries};
use plotters::style::full_palette::{BLUE_300, GREEN_300, RED_300};
use plotters::style::{Color, IntoFont, BLACK, BLUE, GREEN, RED, WHITE};
use rand::distributions::Uniform;
use rand::seq::SliceRandom;
use rust_zero_deeplearning::modules::core::function_libs;
use rust_zero_deeplearning::modules::math::sin;
use rust_zero_deeplearning::modules::*;

// use approx::assert_abs_diff_eq;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{s, Array, Axis, IxDyn};
use rand::SeedableRng;
use rand_isaac::isaac64::Isaac64Rng;

#[test]
fn test_basic() {
    common::setup();

    // スカラ
    let x1 = Variable::new(RawData::new(1.0f64));
    let y1 = sin(x1);
    assert_eq!(0.8414709848078965, y1.get_data()[[]]);

    // 行列に対する Sin
    let x2 = Variable::new(RawData::from_shape_vec(
        vec![2, 3],
        vec![1., 2., 3., 4., 5., 6.],
    ));

    // dbg!(&x2);
    let y2 = sin(x2);
    // dbg!(&y2);
    assert_eq!(vec![2, 3], y2.get_data().shape().to_vec());
    let expect_y2 = vec![
        0.8414709848078965,
        0.9092974268256817,
        0.1411200080598672,
        -0.7568024953079282,
        -0.9589242746631385,
        -0.27941549819892586,
    ];

    let result = y2.get_data().flatten().to_vec();

    let epsilon = 1e-10;
    for (i, value) in result.iter().enumerate() {
        let abs_diff: f64 = value - expect_y2[i];
        assert!(abs_diff.abs() < epsilon);
    }

    // 行列同士の和
    let x3 = Variable::new(RawData::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]));
    let c = Variable::new(RawData::from_shape_vec(
        vec![2, 3],
        vec![10, 20, 30, 40, 50, 60],
    ));

    let x3_c = &x3 + &c;
    //dbg!(&x3_c);
    let expect_x3_c = vec![11, 22, 33, 44, 55, 66];
    assert_eq!(vec![2, 3], x3_c.get_data().shape().to_vec());
    assert_eq!(expect_x3_c, x3_c.get_data().flatten().to_vec());
}

#[test]
fn test_ndarray_reshape_transpose() {
    // ndarray のリシェイプに関する確認
    let shape1 = vec![2, 3];
    let values = vec![1, 2, 3, 4, 5, 6];
    let array1 = Array::from_shape_vec(shape1, values).unwrap();
    dbg!(&array1);

    let shape2 = vec![1, 6];

    let array2 = array1.clone().into_shape_clone(shape2).unwrap();
    dbg!(&array2);

    let shape3 = vec![3, 2];
    let array3 = array1.clone().into_shape_clone(shape3).unwrap();
    dbg!(&array3);

    let shape4 = vec![6, 1];
    let array4 = array1.clone().into_shape_clone(shape4).unwrap();
    dbg!(&array4);

    // ndarray の転置に関する確認
    let t_array1 = array1.t();
    dbg!(&t_array1);
    let rows = t_array1.rows();
    // 全ての行に対してイテレーション
    for (i, row) in rows.into_iter().enumerate() {
        println!("Row {}: {:?}", i, row);
    }

    let t_array2 = array2.t();
    dbg!(&t_array2);

    let axes_array1 = array4.permuted_axes(vec![1, 0]);
    dbg!(&axes_array1);

    let array_values = (0..=23).collect::<Vec<i32>>();
    let array5 = Array::from_shape_vec(vec![4, 3, 2], array_values).unwrap();
    dbg!(&array5);

    let array5_axes = array5.permuted_axes(vec![1, 0, 2]);
    dbg!(&array5_axes);

    let array5_axes_t = array5_axes.t();
    dbg!(&array5_axes_t);
}

/// Variable のリシェイプのテスト
#[test]
fn test_variable_reshape() {
    common::setup();

    let x = Variable::new(RawData::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]));

    let r1 = x.reshape(vec![6]);
    assert_eq!(vec![6], r1.get_data().shape().to_vec());
    assert_eq!(vec![1, 2, 3, 4, 5, 6], r1.get_data().flatten().to_vec());

    let r2 = r1.reshape(vec![3, 2]);
    assert_eq!(vec![3, 2], r2.get_data().shape().to_vec());
    assert_eq!(vec![1, 2, 3, 4, 5, 6], r2.get_data().flatten().to_vec());
}

#[test]
fn test_ndarray_broadcast() {
    let x = Array::from_shape_vec(vec![1, 3], vec![1, 2, 3]).unwrap();
    dbg!(&x);

    let y = x.broadcast(vec![2, 3]);
    dbg!(&y);
}

#[test]
fn test_ndarray_insert_axis() {
    let x = Array::from_shape_vec(vec![3, 2, 2], (1..13).collect()).unwrap();

    let x_sum_axis0 = x.sum_axis(Axis(1));
    dbg!(&x_sum_axis0);

    let x_keepdims_0 = x_sum_axis0.insert_axis(Axis(1));
    dbg!(&x_keepdims_0);

    let x2 = Array::from_shape_vec(vec![2, 3], (1..7).collect()).unwrap();

    let x2_sum_axis0 = x2.sum_axis(Axis(0)).sum_axis(Axis(0));
    let x2_keepdims_0 = x2_sum_axis0.insert_axis(Axis(0)).insert_axis(Axis(0));
    dbg!(&x2_keepdims_0);
    dbg!(&x2_keepdims_0.shape());

    let x3 = Array::from_shape_vec(vec![2, 3, 4], (0..24).collect()).unwrap();
    dbg!(&x3);

    // keepdims = false, axis=0
    let x3_sum_axis0 = x3.sum_axis(Axis(0));
    dbg!(&x3_sum_axis0);

    // keepdims = false, axis=1
    let x3_sum_axis1 = x3.sum_axis(Axis(1));
    dbg!(&x3_sum_axis1);

    // keepdims = true, axis=0
    let x3_sum_keepdims_axis0 = x3.sum_axis(Axis(0));
    let x3_sum_keepdims_axis0 = x3_sum_keepdims_axis0.insert_axis(Axis(0));
    dbg!(&x3_sum_keepdims_axis0);

    // keepdims = true, axis=1
    let x3_sum_keepdims_axis1 = x3.sum_axis(Axis(1));
    let x3_sum_keepdims_axis1 = x3_sum_keepdims_axis1.insert_axis(Axis(1));
    dbg!(&x3_sum_keepdims_axis1);

    // keepdims = true
    let x3_sum_keepdims_no_axis0 = x3.sum_axis(Axis(0)).sum_axis(Axis(0)).sum_axis(Axis(0));
    let x3_sum_keepdims_no_axis0 = x3_sum_keepdims_no_axis0
        .insert_axis(Axis(0))
        .insert_axis(Axis(0))
        .insert_axis(Axis(0));
    dbg!(&x3_sum_keepdims_no_axis0);
}

#[test]
fn test_ndarray_sum_to() {
    let x = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let y = x
        .clone()
        .sum_axis(Axis(0))
        .into_shape_with_order(vec![1, 3])
        .unwrap();
    dbg!(&y);

    let z = x
        .clone()
        .sum_axis(Axis(1))
        .into_shape_with_order(vec![2, 1])
        .unwrap();
    dbg!(&z);

    let orig_shape = x.shape();
    dbg!(&orig_shape);

    let mut result = x.to_owned();

    let target_shape = vec![3, 1];
    for (axis_idx, (&orig_size, &target_size)) in
        orig_shape.iter().zip(target_shape.iter()).enumerate()
    {
        println!(
            "axis_idx: {:?}, orig_size:{:?}, target_size:{:?}",
            axis_idx, orig_size, target_size
        );
        if orig_size > target_size {
            // この次元は集約が必要
            if target_size == 1 {
                // 完全に集約する場合
                result = result.sum_axis(Axis(axis_idx));
            } else {
                // 部分的な集約が必要な場合（より複雑なケース）
                // この例では単純化のため、1に集約するケースのみ対応
                panic!("Partial reduction not supported in this example");
            }
        } else if orig_size < target_size {
            // この次元はブロードキャストが必要（実装が複雑になるため省略）
            //panic!("Broadcasting to larger dimensions not supported in this example");
            println!(
                "result shape: {:?}, target_shape: {:?}",
                result.shape(),
                target_shape
            );
            //result = result.broadcast(target_shape.clone()).unwrap().to_owned();
            //result = result.permuted_axes(target_shape.clone());
            //result = result.
            //dbg!(&dummy);
        }
    }

    dbg!(&result);

    // let r = result
    //     .into_shape(target_shape.to_vec())
    //     .unwrap()
    //     .into_dimensionality::<Ix2>()
    //     .unwrap();
    // dbg!(&r);
}

/// 行列の和に関するテスト
#[test]
fn test_step40_sum1() {
    common::setup();
    let x0 = Variable::new(RawData::from_shape_vec(vec![1, 3], vec![1, 2, 3]));
    let x1 = Variable::new(RawData::from_shape_vec(vec![1], vec![10]));

    let y = &x0 + &x1;

    // 行列の和の形状と値が一致することを確認する。
    assert_eq!(vec![1, 3], y.get_data().shape().to_vec());
    assert_eq!(vec![11, 12, 13], y.get_data().flatten().to_vec());

    //dbg!(&y);

    y.backward();

    // 逆伝播による勾配の形状と値が一致することを確認する。
    let gx0 = x0.get_grad().unwrap().get_data();
    let gx1 = x1.get_grad().unwrap().get_data();

    assert_eq!(vec![1, 3], gx0.shape().to_vec());
    assert_eq!(vec![1, 1, 1], gx0.flatten().to_vec());
    assert_eq!(vec![3], gx1.flatten().to_vec());
}

#[test]
fn test_nd() {
    let x = Array::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).unwrap(); //        .into_diag();

    let w = Array::from_shape_vec((3, 4), vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        .unwrap(); //        .into_diag();
    let result = &x.dot(&w);
    dbg!(&result.shape());

    let gy = Array::from_shape_vec((2, 4), vec![1, 1, 1, 1, 1, 1, 1, 1]).unwrap();
    let gx = &gy.dot(&w.t());

    let gw = &x.t().dot(&gy);

    dbg!(&gx.shape());
    dbg!(&gw.shape());
}

/// 線形回帰
#[test]
fn test_linear_regression() {
    // 乱数による y = 2x + 5 の生成
    let seed = 0;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    let x_var = Array::random_using((100, 1), Uniform::new(0., 1.), &mut rng);
    let y_var: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> =
        5.0 + 2.0 * x_var.clone() + Array::random_using((100, 1), Uniform::new(0., 1.), &mut rng);

    let x = Variable::new(RawData::from_shape_vec(
        x_var.shape(),
        x_var.flatten().to_vec(),
    ));
    let y = Variable::new(RawData::from_shape_vec(
        y_var.shape(),
        y_var.flatten().to_vec(),
    ));

    // 線形回帰による予測
    let mut w = Variable::new(RawData::from_shape_vec(vec![1, 1], vec![0.0]));
    let mut b = Variable::new(RawData::from_shape_vec(vec![1], vec![0.0]));

    let lr = 0.1;
    let iters = 100;

    let mut loss_data = 0.0;
    for _i in 0..iters {
        let y_pred = &matmul(x.clone(), w.clone()) + &b.clone();
        let loss = mean_squared_error(y.clone(), y_pred.clone());

        w.clear_grad();
        b.clear_grad();
        loss.backward();

        let w_new_data = w.get_data() - w.get_grad().unwrap().get_data() * lr;
        w.set_data(w_new_data);

        let b_new_data = b.get_data() - b.get_grad().unwrap().get_data() * lr;
        b.set_data(b_new_data);

        println!(
            "w: {:?}, b: {:?}, loss: {:?}",
            w.get_data(),
            b.get_data(),
            loss.get_data()
        );
        loss_data = loss.get_data().flatten().to_vec()[0];
    }

    // グラフ描画
    // 描画先の Backend を初期化する。
    let root =
        BitMapBackend::new("graph/step42_linear_regression.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // グラフの軸の設定など
    let mut chart = ChartBuilder::on(&root)
        .caption("y=2x+5", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..1.0, 5.0..8.0)
        .unwrap();
    chart.configure_mesh().draw().unwrap();

    // 元データのプロット
    let mut plot_data_vec = vec![];
    let x_var_vec = x_var.flatten().to_vec();
    let y_var_vec = y_var.flatten().to_vec();

    for i in 0..100 {
        plot_data_vec.push(vec![x_var_vec[i], y_var_vec[i]]);
    }
    // 点グラフの定義＆描画
    let point_series = PointSeries::<_, _, Circle<_, _>, _>::new(
        plot_data_vec.iter().map(|xy| (xy[0], xy[1])),
        2,     // Circleのサイズ
        &BLUE, // 色を指定
    );
    chart.draw_series(point_series).unwrap();

    // 線形回帰による予測線の描画
    let pred_x = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let w_data = w.get_data().flatten().to_vec()[0];
    let b_data = b.get_data().flatten().to_vec()[0];

    chart
        .draw_series(LineSeries::new(
            pred_x.iter().map(|x| (*x, *x * w_data + b_data)),
            RED,
        ))
        .unwrap()
        .label(format!("w: {}", w_data).to_string())
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
    chart
        .draw_series(std::iter::once(EmptyElement::at((0.0, 0.0))))
        .unwrap() // 凡例に b の値を出力するためのダミー要素
        .label(format!("b: {}", b_data).to_string())
        .legend(|(x, y)| EmptyElement::at((x, y)));
    chart
        .draw_series(std::iter::once(EmptyElement::at((0.0, 0.0))))
        .unwrap() // 凡例に loss の値を出力するためのダミー要素
        .label(format!("loss: {}", loss_data).to_string())
        .legend(|(x, y)| EmptyElement::at((x, y)));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
}

fn predict(x: Variable<f64>) -> Variable<f64> {
    let w = Variable::new(RawData::from_shape_vec(vec![1, 1], vec![0.0]));
    let b = Variable::new(RawData::from_shape_vec(vec![1], vec![0.0]));

    let y_pred = &matmul(x.clone(), w.clone()) + &b.clone();

    w.clear_grad();
    b.clear_grad();

    return y_pred;
}

#[test]
fn test_non_linear_dataset() {
    let seed = 0;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    let x_var = Array::random_using((100, 1), Uniform::new(0., 1.), &mut rng);
    // let x = Variable::new(RawVariable::from_shape_vec(
    //     vec![100, 1],
    //     x_var.flatten().to_vec(),
    // ));

    let b_var = Array::random_using((100, 1), Uniform::new(0., 1.), &mut rng);
    // let b = Variable::new(RawVariable::from_shape_vec(
    //     vec![100, 1],
    //     b_var.flatten().to_vec(),
    // ));

    let y = (2.0 * PI * x_var.clone()).sin() + b_var;
    dbg!(&y);

    // グラフ描画
    // 描画先の Backend を初期化する。
    let root =
        BitMapBackend::new("graph/step43_non_linear_sin.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // グラフの軸の設定など
    let mut chart = ChartBuilder::on(&root)
        .caption("y=sin(2 * PI * x) + b", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..1.0, -1.0..2.0)
        .unwrap();
    chart.configure_mesh().draw().unwrap();

    // 元データのプロット
    let mut plot_data_vec = vec![];
    let x_var_vec = x_var.flatten().to_vec();
    let y_var_vec = y.flatten().to_vec();

    for i in 0..100 {
        plot_data_vec.push(vec![x_var_vec[i], y_var_vec[i]]);
    }
    // 点グラフの定義＆描画
    let point_series = PointSeries::<_, _, Circle<_, _>, _>::new(
        plot_data_vec.iter().map(|xy| (xy[0], xy[1])),
        2,     // Circleのサイズ
        &BLUE, // 色を指定
    );
    chart.draw_series(point_series).unwrap();

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
}

// #[test]
fn test_predict() {
    //env::set_var("RUST_LOG", "info");

    env_logger::init();

    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_disabled();
    // バックプロパゲーションを行う。
    Setting::set_backprop_enabled();

    let seed = 0;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    ////////////////////////////////////////////////////
    // データセット
    ////////////////////////////////////////////////////
    let x_var = Array::random_using((100, 1), Uniform::new(0., 1.), &mut rng);
    let x = Variable::new(RawData::from_shape_vec(
        vec![100, 1],
        x_var.flatten().to_vec(),
    ));

    let b_var = Array::random_using((100, 1), Uniform::new(0., 1.), &mut rng);
    let y_var = (2.0 * PI * x_var.clone()).sin() + b_var;
    let y = Variable::new(RawData::from_shape_vec(
        y_var.shape().to_vec(),
        y_var.flatten().to_vec(),
    ));

    ////////////////////////////////////////////////////
    // 重みの初期化
    ////////////////////////////////////////////////////
    let i = 1;
    let h = 10;
    let o = 1;

    let w1_var = Array::random_using((i, h), Uniform::new(0., 1.), &mut rng) * 0.01;
    let w1 = Variable::new(RawData::from_shape_vec(
        vec![i, h],
        w1_var.flatten().to_vec(),
    ));

    let b1 = Variable::new(RawData::from_vec(vec![0.0; h]));

    let w2_var = Array::random_using((h, o), Uniform::new(0., 1.), &mut rng) * 0.01;
    let w2 = Variable::new(RawData::from_shape_vec(
        vec![h, o],
        w2_var.flatten().to_vec(),
    ));

    let b2 = Variable::new(RawData::from_vec(vec![0.0; o]));

    ////////////////////////////////////////////////////
    // ニューラルネットワークの推論
    ////////////////////////////////////////////////////
    let lr = 0.2;
    let iters = 10000;

    ////////////////////////////////////////////////////
    // ニューラルネットワークの学習
    ////////////////////////////////////////////////////
    for idx in 0..iters {
        let y_pred = function_libs::predict_linear_with_sigmoid(
            x.clone(),
            vec![w1.clone(), w2.clone()],
            vec![b1.clone(), b2.clone()],
        );
        let loss = mean_squared_error(y.clone(), y_pred.clone());

        w1.clear_grad();
        w2.clear_grad();
        b1.clear_grad();
        b2.clear_grad();
        // y.clear_grad();

        loss.backward();

        let w1_udpate = w1.get_data() - lr * w1.get_grad().unwrap().get_data();
        let b1_udpate = b1.get_data() - lr * b1.get_grad().unwrap().get_data();
        let w2_udpate = w2.get_data() - lr * w2.get_grad().unwrap().get_data();
        let b2_udpate = b2.get_data() - lr * b2.get_grad().unwrap().get_data();

        w1.set_data(w1_udpate);
        w2.set_data(w2_udpate);
        b1.set_data(b1_udpate);
        b2.set_data(b2_udpate);

        ////////////////////////////////////////////////////
        // 学習途中の状況をグラフ出力する。
        ////////////////////////////////////////////////////
        if idx % 1000 == 0 || idx == iters - 1 {
            println!("[{}] loss: {:?}", idx, loss.get_data());
            let plot_x = x_var.flatten().to_vec();
            let plot_y = y_var.flatten().to_vec();

            let test_x: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
            let test_y_var = function_libs::predict_linear_with_sigmoid(
                Variable::new(RawData::from_shape_vec(vec![100, 1], test_x.clone())),
                vec![w1.clone(), w2.clone()],
                vec![b1.clone(), b2.clone()],
            );

            let test_y = test_y_var.get_data().flatten().to_vec();

            let mut test_xy = vec![];
            for (i, tmp_x) in test_x.iter().enumerate() {
                test_xy.push((*tmp_x, test_y[i]));
            }
            utils::draw_graph(
                "y=sin(2 * pi * x) + b",
                &format!("graph/step43_neural_network_pred_{}.png", idx),
                (0.0, 1.0),
                (-1.0, 2.0),
                plot_x,
                plot_y,
                test_xy,
                &format!("loss: {}", loss.get_data().flatten().to_vec()[0]),
            );
        }
    }
}

#[test]
fn test_step48_train_spiral() {
    // ハイパーパラメータの設定
    let max_epoch = 300;
    let batch_size = 30;
    let hidden_size = 10;
    let lr = 1.0;

    // データの読み込み、モデル・オプティマイザの生成
    let (x, t) = datasets::get_spiral(true);
    let sigmoid = Rc::new(RefCell::new(SigmoidFunction {}));
    let sgd = Sgd::new(lr);
    let mut mlp = Mlp::new(vec![hidden_size, 3], sigmoid, sgd);

    let data_size = x.shape().to_vec()[0];
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

            let mut batch_x_vec = vec![];
            let mut batch_t_vec = vec![];

            for idx in batch_index.iter() {
                let x_var = x.slice(s![*idx, ..]);
                batch_x_vec.extend(x_var.flatten().to_vec());
                batch_t_vec.push(t.flatten().to_vec()[*idx]);
            }
            let batch_x = Variable::new(RawData::from_shape_vec(vec![batch_size, 2], batch_x_vec));
            let batch_t = Variable::new(RawData::from_shape_vec(
                vec![1, batch_size],
                batch_t_vec.clone(),
            ));

            // 勾配の算出、パラメータの更新
            let y = mlp.forward(vec![batch_x.clone()]);
            let loss = softmax_cross_entropy(y[0].clone(), batch_t.clone());
            mlp.cleargrads();

            loss.backward();
            mlp.update_parameters();

            sum_loss += loss.get_data().flatten().to_vec()[0] as f64 * batch_t_vec.len() as f64;
        }
        let avg_loss = sum_loss / data_size as f64;
        println!("epoch {}, loss: {}", epoch + 1, avg_loss);
        loss_result.push(avg_loss);
    }

    // 損失結果グラフ描画
    // 描画先の Backend を初期化する。
    let root =
        BitMapBackend::new("graph/step48_spiral_train_loss.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // グラフの軸の設定など
    let mut chart = ChartBuilder::on(&root)
        .caption("Spiral data train loss", ("sans-serif", 50).into_font())
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

    // 学習後のニューラルネットワークの決定境界を描画する。
    let h = 0.001;
    let mut x_xs = vec![];
    let mut x_ys = vec![];
    for xs in x.axis_iter(Axis(0)) {
        x_xs.push(xs.flatten().to_vec()[0]);
        x_ys.push(xs.flatten().to_vec()[1]);
    }

    let mut x_min = 0f64;
    let mut x_max = 0f64;
    let mut y_min = 0f64;
    let mut y_max = 0f64;

    for x in x_xs.iter() {
        if *x > x_max {
            x_max = *x;
        }
        if *x < x_min {
            x_min = *x;
        }
    }

    for y in x_ys.iter() {
        if *y > y_max {
            y_max = *y;
        }
        if *y < y_min {
            y_min = *y;
        }
    }

    x_min -= 0.1;
    x_max += 0.1;
    y_min -= 0.1;
    y_max += 0.1;

    println!("x min: {} x max: {}", x_min, x_max);
    println!("y min: {} y max: {}", y_min, y_max);

    let xx = Array::range(x_min, x_max, h);
    let yy = Array::range(y_min, y_max, h);

    println!(
        "x shape:{:?}, xx shape: {:?}, yy shape:{:?}",
        x.shape(),
        xx.shape(),
        yy.shape()
    );

    let pred_batch_size = 100;
    let pred_max_iter = xx.shape().to_vec()[0] * yy.shape().to_vec()[0] / pred_batch_size;
    println!("pred max iter: {}", pred_max_iter);

    // 逆伝播を実行しない。微分値を保持しない。
    Setting::set_retain_grad_disabled();
    // バックプロパゲーションを行わない。
    Setting::set_backprop_disabled();

    let mut x_vec = vec![];
    for (i, xx_var) in xx.iter().enumerate() {
        for (j, yy_var) in yy.iter().enumerate() {
            //println!("xx: {:?} yy:{:?}", xx_var, yy_var);
            x_vec.push(*xx_var);
            x_vec.push(*yy_var);
        }
    }
    let x_var = Variable::new(RawData::from_shape_vec(vec![x_vec.len() / 2, 2], x_vec));

    dbg!(&x_var.get_data().shape());

    let score = mlp.forward(vec![x_var.clone()]);
    println!("score shape:{:?}", score[0].get_data().shape());

    let predict_cls = score[0]
        .get_data()
        .axis_iter(ndarray::Axis(0))
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();

    // for (i, val) in score[0].get_data().axis_iter(Axis(0)).enumerate() {
    //     println!("{:?}: {}", val, predict_cls[i]);
    // }
    let x_points: Vec<f64> = x_var.get_data().slice(s![.., 0]).to_vec();
    let y_points: Vec<f64> = x_var.get_data().slice(s![.., 1]).to_vec();

    draw_decision_boundary(x, t, x_points, y_points, predict_cls);
}

/// テスト用のスパイラルデータと学習後のモデルによる決定境界を描画する。
fn draw_decision_boundary(
    x: Array<f64, IxDyn>,
    t: Array<usize, IxDyn>,
    x_points: Vec<f64>,
    y_points: Vec<f64>,
    predict_cls: Vec<usize>,
) {
    let file_path = "graph/step48_spiral_decision_boundary.png";
    let caption = "step48_spiral_decision_boundary";

    // グラフ描画
    // 描画先の Backend を初期化する。
    let root = BitMapBackend::new(&file_path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // グラフの軸の設定など
    let mut chart = ChartBuilder::on(&root)
        .caption(&caption, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1.0..1.0, -1.0..1.0)
        .unwrap();
    chart.configure_mesh().draw().unwrap();

    // 決定境界の病害
    chart
        .draw_series((0..x_points.len()).map(|i| {
            let x_x = x_points[i];
            let x_y = y_points[i];

            let point = (x_x, x_y);

            let t_var = predict_cls[i];

            match t_var {
                0 => Circle::new(point, 1, GREEN_300.filled()).into_dyn(),
                1 => TriangleMarker::new(point, 1, BLUE_300.filled()).into_dyn(),
                _ => Cross::new(point, 1, RED_300.filled()).into_dyn(),
            }
        }))
        .unwrap();

    // テストデータの描画
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
