extern crate rust_zero_deeplearning;

#[path = "common/mod.rs"]
mod common;

use ndarray_rand::RandomExt;
use plotters::chart::ChartBuilder;
use plotters::prelude::{BitMapBackend, Circle, EmptyElement, IntoDrawingArea, PathElement};
use plotters::series::{LineSeries, PointSeries};
use plotters::style::{Color, IntoFont, BLACK, BLUE, GREEN, MAGENTA, RED, WHITE};
use rand::distributions::Uniform;
use rust_zero_deeplearning::modules::*;
use rust_zero_deeplearning::modules::{math::sin, utils::*};
use rust_zero_deeplearning::*;
// use approx::assert_abs_diff_eq;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, Axis};
use rand::SeedableRng;
use rand_isaac::isaac64::Isaac64Rng;

#[test]
fn test_basic() {
    common::setup();

    // スカラ
    let x1 = Variable::new(RawVariable::new(1.0f64));
    let y1 = sin(x1);
    assert_eq!(0.8414709848078965, y1.borrow().get_data()[[]]);

    // 行列に対する Sin
    let x2 = Variable::new(RawVariable::from_shape_vec(
        vec![2, 3],
        vec![1., 2., 3., 4., 5., 6.],
    ));

    // dbg!(&x2);
    let y2 = sin(x2);
    // dbg!(&y2);
    assert_eq!(vec![2, 3], y2.borrow().get_data().shape().to_vec());
    let expect_y2 = vec![
        0.8414709848078965,
        0.9092974268256817,
        0.1411200080598672,
        -0.7568024953079282,
        -0.9589242746631385,
        -0.27941549819892586,
    ];

    assert_eq!(expect_y2, y2.borrow().get_data().flatten().to_vec());

    // 行列同士の和
    let x3 = Variable::new(RawVariable::from_shape_vec(
        vec![2, 3],
        vec![1, 2, 3, 4, 5, 6],
    ));
    let c = Variable::new(RawVariable::from_shape_vec(
        vec![2, 3],
        vec![10, 20, 30, 40, 50, 60],
    ));

    let x3_c = &x3 + &c;
    //dbg!(&x3_c);
    let expect_x3_c = vec![11, 22, 33, 44, 55, 66];
    assert_eq!(vec![2, 3], x3_c.borrow().get_data().shape().to_vec());
    assert_eq!(expect_x3_c, x3_c.borrow().get_data().flatten().to_vec());
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

    let x = Variable::new(RawVariable::from_shape_vec(
        vec![2, 3],
        vec![1, 2, 3, 4, 5, 6],
    ));

    let r1 = x.reshape(vec![6]);
    assert_eq!(vec![6], r1.borrow().get_data().shape().to_vec());
    assert_eq!(
        vec![1, 2, 3, 4, 5, 6],
        r1.borrow().get_data().flatten().to_vec()
    );

    let r2 = r1.reshape(vec![3, 2]);
    assert_eq!(vec![3, 2], r2.borrow().get_data().shape().to_vec());
    assert_eq!(
        vec![1, 2, 3, 4, 5, 6],
        r2.borrow().get_data().flatten().to_vec()
    );
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
    let x0 = Variable::new(RawVariable::from_shape_vec(vec![1, 3], vec![1, 2, 3]));
    let x1 = Variable::new(RawVariable::from_shape_vec(vec![1], vec![10]));

    let y = &x0 + &x1;

    // 行列の和の形状と値が一致することを確認する。
    assert_eq!(vec![1, 3], y.borrow().get_data().shape().to_vec());
    assert_eq!(vec![11, 12, 13], y.borrow().get_data().flatten().to_vec());

    //dbg!(&y);

    y.backward();

    // 逆伝播による勾配の形状と値が一致することを確認する。
    let gx0 = x0.borrow().get_grad().unwrap().borrow().get_data();
    let gx1 = x1.borrow().get_grad().unwrap().borrow().get_data();

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

    let x = Variable::new(RawVariable::from_shape_vec(
        x_var.shape(),
        x_var.flatten().to_vec(),
    ));
    let y = Variable::new(RawVariable::from_shape_vec(
        y_var.shape(),
        y_var.flatten().to_vec(),
    ));

    // 線形回帰による予測
    let mut w = Variable::new(RawVariable::from_shape_vec(vec![1, 1], vec![0.0]));
    let mut b = Variable::new(RawVariable::from_shape_vec(vec![1], vec![0.0]));

    let lr = 0.1;
    let iters = 100;

    let mut loss_data = 0.0;
    for _i in 0..iters {
        let y_pred = &matmul(x.clone(), w.clone()) + &b.clone();
        let loss = mean_squared_error(y.clone(), y_pred.clone());

        w.borrow_mut().clear_grad();
        b.borrow_mut().clear_grad();
        loss.backward();

        let w_new_data =
            w.borrow().get_data() - w.borrow().get_grad().unwrap().borrow().get_data() * lr;
        w.set_data(w_new_data);

        let b_new_data =
            b.borrow().get_data() - b.borrow().get_grad().unwrap().borrow().get_data() * lr;
        b.set_data(b_new_data);

        println!(
            "w: {:?}, b: {:?}, loss: {:?}",
            w.borrow().get_data(),
            b.borrow().get_data(),
            loss.borrow().get_data()
        );
        loss_data = loss.borrow().get_data().flatten().to_vec()[0];
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
        plot_data_vec.iter().map(|(xy)| (xy[0], xy[1])),
        2,     // Circleのサイズ
        &BLUE, // 色を指定
    );
    chart.draw_series(point_series).unwrap();

    // 線形回帰による予測線の描画
    let pred_x = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let w_data = w.borrow().get_data().flatten().to_vec()[0];
    let b_data = b.borrow().get_data().flatten().to_vec()[0];

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
    let w = Variable::new(RawVariable::from_shape_vec(vec![1, 1], vec![0.0]));
    let b = Variable::new(RawVariable::from_shape_vec(vec![1], vec![0.0]));

    let y_pred = &matmul(x.clone(), w.clone()) + &b.clone();

    w.borrow_mut().clear_grad();
    b.borrow_mut().clear_grad();

    return y_pred;
}
