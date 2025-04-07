extern crate rust_zero_deeplearning;

#[path = "common/mod.rs"]
mod common;

use rust_zero_deeplearning::modules::*;
use rust_zero_deeplearning::modules::{math::sin, utils::*};
use rust_zero_deeplearning::*;
// use approx::assert_abs_diff_eq;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};

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
        -0.7568024953079283,
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
