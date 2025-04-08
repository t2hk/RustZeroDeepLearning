extern crate rust_zero_deeplearning;

#[path = "common/mod.rs"]
mod common;

use rust_zero_deeplearning::modules::*;
use rust_zero_deeplearning::modules::{math::sin, utils::*};
use rust_zero_deeplearning::*;
// use approx::assert_abs_diff_eq;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, ArrayBase, Axis, Ix2, IxDyn, OwnedArcRepr};

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
    let y = x.clone().sum_axis(Axis(0)).into_shape(vec![1, 3]).unwrap();
    dbg!(&y);

    let z = x.clone().sum_axis(Axis(1)).into_shape(vec![2, 1]).unwrap();
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

    let r = result
        .into_shape(target_shape.to_vec())
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    dbg!(&r);
}

// // 汎用的なsum_to_shape関数
// fn sum_to_shape<D>(array: &ArrayBase<OwnedRepr<i32>, D>, target_shape: &[usize]) -> Array<i32, Ix2>
// where
//     D: Dimension,
// {
//     let orig_shape = array.shape();

//     // 入力と目標の形状のランクが一致することを確認
//     assert_eq!(
//         orig_shape.len(),
//         target_shape.len(),
//         "Input shape and target shape must have the same rank"
//     );

//     // 各次元ごとに処理するためにクローンを作成
//     let mut result = array.to_owned();

//     // 各次元について、集約が必要かどうかをチェック
//     for (axis_idx, (&orig_size, &target_size)) in
//         orig_shape.iter().zip(target_shape.iter()).enumerate()
//     {
//         if orig_size > target_size {
//             // この次元は集約が必要
//             if target_size == 1 {
//                 // 完全に集約する場合
//                 result = result.sum_axis(Axis(axis_idx));
//             } else {
//                 // 部分的な集約が必要な場合（より複雑なケース）
//                 // この例では単純化のため、1に集約するケースのみ対応
//                 panic!("Partial reduction not supported in this example");
//             }
//         } else if orig_size < target_size {
//             // この次元はブロードキャストが必要（実装が複雑になるため省略）
//             panic!("Broadcasting to larger dimensions not supported in this example");
//         }
//         // orig_size == target_size の場合はそのまま
//     }

//     // 最終的な形状に変換
//     result
//         .into_shape(target_shape.to_vec())
//         .unwrap()
//         .into_dimensionality::<Ix2>()
//         .unwrap()
// }
