extern crate rust_zero_deeplearning;

#[path = "common/mod.rs"]
mod common;

use rust_zero_deeplearning::modules::*;
use rust_zero_deeplearning::modules::{math::sin, utils::*};
use rust_zero_deeplearning::*;
// use approx::assert_abs_diff_eq;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, Axis, IxDyn};

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
