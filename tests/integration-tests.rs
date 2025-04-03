extern crate rust_zero_deeplearning;

#[path = "common/mod.rs"]
mod common;

use num_bigint::{BigInt, ToBigInt};
use rust_zero_deeplearning::modules::utils::*;
use rust_zero_deeplearning::modules::*;
use rust_zero_deeplearning::*;
// use approx::assert_abs_diff_eq;
use core::fmt::Debug;
use env_logger;
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use rand::prelude::*;
use std::env;

use std::rc::Rc;

/// BigInt 型を Variable に設定するテスト
#[test]
fn test_bigint_variable() {
    common::setup();

    let big_int_variable =
        Variable::new(RawVariable::new(BigIntWrapper(Rc::new(BigInt::from(10)))));

    let expect = Array::from_elem(IxDyn(&[]), BigIntWrapper(Rc::new(BigInt::from(10))));

    assert_eq!(expect, big_int_variable.borrow().get_data());
}

#[test]
fn test_add_mul() {
    common::setup();

    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_enabled();

    // バックプロパゲーションを行う。
    Setting::set_backprop_enabled();

    // 順伝播
    let a = Variable::new(RawVariable::new(3.0f32));
    let b = Variable::new(RawVariable::new(2.0f32));
    let c = Variable::new(RawVariable::new(1.0f32));
    let expected = RawVariable::new(7f32);

    let result = add(mul(a.clone(), b.clone()), c.clone());
    assert_eq!(expected.get_data(), result.borrow().get_data());

    // 逆伝播
    //FunctionExecutor::backward_all(vec![Rc::clone(&result)]);
    //result.as_ref().clone().borrow().backward();
    result.backward();

    println!(
        "result grad: {:?}, a grad: {:?}, b grad: {:?}, c grad: {:?}",
        &result.borrow().get_grad(),
        &a.borrow().get_grad(),
        &b.borrow().get_grad(),
        &c.borrow().get_grad(),
    );

    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        result
            .borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 2.0),
        a.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 3.0),
        b.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
}

#[test]
/// 乗算のテスト(f32)
fn test_mul_2() {
    common::setup();

    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_enabled();

    // バックプロパゲーションを行う。
    Setting::set_backprop_enabled();

    // 順伝播
    let x1 = Variable::new(RawVariable::new(5.0f32));
    let x2 = Variable::new(RawVariable::new(10.0f32));
    let expected = RawVariable::new(50.0f32);

    let result = mul(x1.clone(), x2.clone());
    assert_eq!(expected.get_data(), result.borrow().get_data());

    // 逆伝播
    //FunctionExecutor::backward_all(vec![Rc::clone(&result)]);
    // result.as_ref().clone().borrow().backward();
    result.backward();

    println!(
        "result grad: {:?}, x1 grad: {:?}, x2 grad: {:?}",
        &result.borrow().get_grad(),
        &x1.borrow().get_grad(),
        &x2.borrow().get_grad()
    );

    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        result
            .borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 10.0),
        x1.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 5.0),
        x2.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
}

#[test]
/// 乗算のテスト(i32)
fn test_mul_1() {
    common::setup();

    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_enabled();

    // バックプロパゲーションを行う。
    Setting::set_backprop_enabled();

    // 順伝播
    let x1 = Variable::new(RawVariable::new(5i32));
    let x2 = Variable::new(RawVariable::new(10i32));
    let expected = RawVariable::new(50);

    let result = mul(x1.clone(), x2.clone());
    assert_eq!(expected.get_data(), result.borrow().get_data());

    // 逆伝播
    result.backward();

    println!(
        "result grad: {:?}, x1 grad: {:?}, x2 grad: {:?}",
        &result.borrow().get_grad().expect("No grad exist."),
        &x1.borrow().get_grad().expect("No grad exist."),
        &x2.borrow().get_grad().expect("No grad exist.")
    );

    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1),
        result
            .borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 10),
        x1.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 5),
        x2.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
}

/// 中間変数の微分結果を保持し無い場合のテスト
#[test]
fn test_retain_grad_disabled_u32() {
    common::setup();

    let x1 = Variable::new(RawVariable::new(2u32));
    let x2 = Variable::new(RawVariable::new(3u32));
    let a = square(x1.clone());
    let b = square(a.clone());
    let c = square(a.clone());
    let d = add(b.clone(), c.clone());
    let y = add(d.clone(), x2.clone());

    // 順伝播の結果
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 35),
        y.borrow().get_data().clone()
    );
    // 各変数の世代のテスト
    assert_eq!(0, x1.borrow().get_generation());
    assert_eq!(0, x2.borrow().get_generation());
    assert_eq!(1, a.borrow().get_generation());
    assert_eq!(2, b.borrow().get_generation());
    assert_eq!(2, c.borrow().get_generation());
    assert_eq!(3, d.borrow().get_generation());
    assert_eq!(4, y.borrow().get_generation());

    // 各関数の世代のテスト
    assert_eq!(0, a.borrow().get_creator_generation());
    assert_eq!(1, b.borrow().get_creator_generation());
    assert_eq!(1, c.borrow().get_creator_generation());
    assert_eq!(2, d.borrow().get_creator_generation());
    assert_eq!(3, y.borrow().get_creator_generation());

    // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
    let creators = FunctionExecutor::extract_creators(vec![y.clone()]);

    // dbg!(&d);
    // dbg!(&creators);

    // 実行した関数の数をチェックする。
    assert_eq!(5, creators.len());

    // 逆伝播を実行する。微分値を保持しない。
    Setting::set_retain_grad_disabled();
    for (_gen, creator) in creators.iter() {
        creator.borrow_mut().backward();
    }

    // 逆伝播の結果の確認
    // 途中結果の変数には微分値が設定されていないことを確認する。
    assert_eq!(Array::from_elem(IxDyn(&[]), 2), x1.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 64),
        x1.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 3), x2.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1),
        x2.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 4), a.borrow().get_data());
    assert!(a.borrow().get_grad().is_none());
    assert_eq!(Array::from_elem(IxDyn(&[]), 16), b.borrow().get_data());
    assert!(b.borrow().get_grad().is_none());
    assert_eq!(Array::from_elem(IxDyn(&[]), 16), c.borrow().get_data());
    assert!(c.borrow().get_grad().is_none());
    assert_eq!(Array::from_elem(IxDyn(&[]), 32), d.borrow().get_data());
    assert!(d.borrow().get_grad().is_none());
    assert_eq!(Array::from_elem(IxDyn(&[]), 35), y.borrow().get_data());
    assert!(y.borrow().get_grad().is_none());
}

/// 2乗と加算のテスト
/// (x1 + x2)^2 の順伝播と逆伝播をテストする。
#[test]
fn test_multidim_add_square_1() {
    common::setup();

    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_enabled();

    // テスト用の入力値
    let sh1 = vec![2, 2];
    let val1 = vec![1., 2., 3., 4.];
    let var1 = RawVariable::from_shape_vec(sh1, val1);
    dbg!(&var1);
    let sh2 = vec![2, 2];
    let val2 = vec![11., 12., 13., 14.];
    let var2 = RawVariable::from_shape_vec(sh2, val2);
    dbg!(&var2);

    let x1 = Variable::new(var1);
    let x2 = Variable::new(var2);

    // 順伝播の結果 [[12., 14.],[16., 18.]]^2 = [[144., 196.], [256., 324.]]
    let expected = RawVariable::from_shape_vec(vec![2, 2], vec![144., 196., 256., 324.]);
    // 逆伝播の結果 2 * [[12., 14.], [16., 18.]]
    let expected_grad = RawVariable::from_shape_vec(vec![2, 2], vec![24., 28., 32., 36.]);

    let result = square(add(x1.clone(), x2.clone()));

    // 順伝播の結果を確認する。
    // 逆伝播の微分結果 grad が入力値に設定されていないことも確認する。
    assert_eq!(true, x1.borrow().get_grad().is_none());
    assert_eq!(true, x2.borrow().get_grad().is_none());
    assert_eq!(expected.get_data(), result.borrow().get_data().clone());

    // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
    result.backward();

    // 逆伝播の結果を確認する。
    // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
    assert_eq!(
        expected_grad.get_data(),
        x1.borrow().get_grad().unwrap().borrow().get_data()
    );
    assert_eq!(
        expected_grad.get_data(),
        x2.borrow().get_grad().unwrap().borrow().get_data()
    );
}

#[test]
fn test_multidim_square() {
    common::setup();

    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_enabled();

    // 入力値の準備
    let sh1 = vec![2, 2];
    let arr1 = vec![1., 2., 3., 4.];
    let var1 = RawVariable::from_shape_vec(sh1, arr1);
    let x1 = Variable::new(var1.clone());

    let expected_var = RawVariable::from_shape_vec(vec![2, 2], vec![1., 4., 9., 16.]);
    let expected_grad = RawVariable::from_shape_vec(vec![2, 2], vec![2., 4., 6., 8.]);

    // 順伝播、逆伝播を実行する。
    let result = square(x1.clone());
    assert_eq!(&expected_var.get_data(), &result.borrow().get_data());

    result.backward();
    assert_eq!(
        &expected_grad.get_data(),
        &x1.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
}

#[test]
fn test_multidim_add() {
    common::setup();

    let sh1 = vec![2, 2];
    let val1 = vec![1., 2., 3., 4.];
    let var1 = RawVariable::from_shape_vec(sh1, val1);
    dbg!(&var1);
    let sh2 = vec![2, 2];
    let val2 = vec![11., 12., 13., 14.];
    let var2 = RawVariable::from_shape_vec(sh2, val2);
    // dbg!(&var2);

    // 加算値をランダムに生成する。
    let x1 = Variable::new(var1);
    let x2 = Variable::new(var2);

    let expected_var = RawVariable::from_shape_vec(vec![2, 2], vec![12., 14., 16., 18.]);

    // 加算した結果の期待値を計算する。
    // let expected_output_data = Array::from_elem(IxDyn(&[]), 2.0);

    // 順伝播を実行する。
    let result = add(x1.clone(), x2.clone());
    assert_eq!(&expected_var.get_data(), &result.borrow().get_data());

    // dbg!(&result.borrow().get_data());
}

/// バックプロパゲーションの有効・無効のテスト。
#[test]
fn test_disable_backprop() {
    common::setup();

    // バックプロパゲーションを行わない場合
    Setting::set_backprop_disabled();
    let x = Variable::new(RawVariable::new(Array::from_elem(
        IxDyn(&[100, 100, 100]),
        1.0,
    )));

    let result = square(square(square(x.clone())));

    // dbg!(&result.borrow().generation);
    assert_eq!(1, result.borrow().get_generation());

    // バックプロパゲーションを行う場合
    Setting::set_backprop_enabled();
    let x = Variable::new(RawVariable::new(Array::from_elem(
        IxDyn(&[100, 100, 100]),
        1.0,
    )));

    let result = square(square(square(x.clone())));

    //dbg!(&result.borrow().generation);
    assert_eq!(3, result.borrow().get_generation());
}

/// 中間変数の微分結果を保持し無い場合のテスト
#[test]
fn test_retain_grad_disabled() {
    common::setup();

    let x1 = Variable::new(RawVariable::new(2.0));
    let x2 = Variable::new(RawVariable::new(3.0));
    let a = square(x1.clone());
    let b = square(a.clone());
    let c = square(a.clone());
    let d = add(b.clone(), c.clone());
    let y = add(d.clone(), x2.clone());

    // 順伝播の結果
    assert_eq!(Array::from_elem(IxDyn(&[]), 35.0), y.borrow().get_data());
    // 各変数の世代のテスト
    assert_eq!(0, x1.borrow().get_generation());
    assert_eq!(0, x2.borrow().get_generation());
    assert_eq!(1, a.borrow().get_generation());
    assert_eq!(2, b.borrow().get_generation());
    assert_eq!(2, c.borrow().get_generation());
    assert_eq!(3, d.borrow().get_generation());
    assert_eq!(4, y.borrow().get_generation());

    // 各関数の世代のテスト
    assert_eq!(0, a.borrow().get_creator_generation());
    assert_eq!(1, b.borrow().get_creator_generation());
    assert_eq!(1, c.borrow().get_creator_generation());
    assert_eq!(2, d.borrow().get_creator_generation());
    assert_eq!(3, y.borrow().get_creator_generation());

    // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
    let creators = FunctionExecutor::extract_creators(vec![y.clone()]);

    // dbg!(&d);
    // dbg!(&creators);

    // 実行した関数の数をチェックする。
    assert_eq!(5, creators.len());

    // 逆伝播を実行する。微分値を保持しない。
    Setting::set_retain_grad_disabled();

    y.backward();

    // 逆伝播の結果の確認
    // 途中結果の変数には微分値が設定されていないことを確認する。
    assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), x1.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 64.0),
        x1.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 3.0), x2.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        x2.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 4.0), a.borrow().get_data());
    assert!(a.borrow().get_grad().is_none());
    assert_eq!(Array::from_elem(IxDyn(&[]), 16.0), b.borrow().get_data());
    assert!(b.borrow().get_grad().is_none());
    assert_eq!(Array::from_elem(IxDyn(&[]), 16.0), c.borrow().get_data());
    assert!(c.borrow().get_grad().is_none());
    assert_eq!(Array::from_elem(IxDyn(&[]), 32.0), d.borrow().get_data());
    assert!(d.borrow().get_grad().is_none());
    assert_eq!(Array::from_elem(IxDyn(&[]), 35.0), y.borrow().get_data());
    assert!(y.borrow().get_grad().is_none());
}

/// 中間変数の微分結果を保持する場合のテスト。
#[test]
fn test_retain_grad_enabled() {
    common::setup();

    let x1 = Variable::new(RawVariable::new(2.0));
    let x2 = Variable::new(RawVariable::new(3.0));
    let a = square(x1.clone());
    let b = square(a.clone());
    let c = square(a.clone());
    let d = add(b.clone(), c.clone());
    let y = add(d.clone(), x2.clone());

    // 順伝播の結果
    assert_eq!(Array::from_elem(IxDyn(&[]), 35.0), y.borrow().get_data());
    // 各変数の世代のテスト
    assert_eq!(0, x1.borrow().get_generation());
    assert_eq!(0, x2.borrow().get_generation());
    assert_eq!(1, a.borrow().get_generation());
    assert_eq!(2, b.borrow().get_generation());
    assert_eq!(2, c.borrow().get_generation());
    assert_eq!(3, d.borrow().get_generation());
    assert_eq!(4, y.borrow().get_generation());

    // 各関数の世代のテスト
    assert_eq!(0, a.borrow().get_creator_generation());
    assert_eq!(1, b.borrow().get_creator_generation());
    assert_eq!(1, c.borrow().get_creator_generation());
    assert_eq!(2, d.borrow().get_creator_generation());
    assert_eq!(3, y.borrow().get_creator_generation());

    // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
    let creators = FunctionExecutor::extract_creators(vec![y.clone()]);

    // dbg!(&d);
    // dbg!(&creators);

    // 実行した関数の数をチェックする。
    assert_eq!(5, creators.len());

    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_enabled();

    y.backward();

    // 逆伝播の結果の確認
    // 途中結果の変数には微分値が設定されていないことを確認する。
    assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), x1.borrow().get_data());

    // 逆伝播の結果の確認
    // 途中結果の変数には微分値が設定されていないことを確認する。
    assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), x1.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 64.0),
        x1.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 3.0), x2.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        x2.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 4.0), a.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 16.0),
        a.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 16.0), b.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        b.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 16.0), c.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        c.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 32.0), d.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        d.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 35.0), y.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        y.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
}

// 世代に関するテスト。
// x1 -> x1^2 -> a -> a^2 -> b -> b+c -> d -> d+x2 -> y
//               -> a^2 -> c /          x2
#[test]
fn test_generations() {
    common::setup();

    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_enabled();

    let x1 = Variable::new(RawVariable::new(2.0));
    let x2 = Variable::new(RawVariable::new(3.0));
    let a = square(x1.clone());
    let b = square(a.clone());
    let c = square(a.clone());
    let d = add(b.clone(), c.clone());
    let y = add(d.clone(), x2.clone());

    // 順伝播の結果
    assert_eq!(Array::from_elem(IxDyn(&[]), 35.0), y.borrow().get_data());
    // 各変数の世代のテスト
    assert_eq!(0, x1.borrow().get_generation());
    assert_eq!(0, x2.borrow().get_generation());
    assert_eq!(1, a.borrow().get_generation());
    assert_eq!(2, b.borrow().get_generation());
    assert_eq!(2, c.borrow().get_generation());
    assert_eq!(3, d.borrow().get_generation());
    assert_eq!(4, y.borrow().get_generation());

    // 各関数の世代のテスト
    assert_eq!(0, a.borrow().get_creator_generation());
    assert_eq!(1, b.borrow().get_creator_generation());
    assert_eq!(1, c.borrow().get_creator_generation());
    assert_eq!(2, d.borrow().get_creator_generation());
    assert_eq!(3, y.borrow().get_creator_generation());

    // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
    let creators = FunctionExecutor::extract_creators(vec![y.clone()]);

    // dbg!(&d);
    // dbg!(&creators);

    // 実行した関数の数をチェックする。
    assert_eq!(5, creators.len());

    // 逆伝播を実行する。
    y.backward();

    // 逆伝播の結果の確認
    assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), x1.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 64.0),
        x1.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 3.0), x2.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        x2.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 4.0), a.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 16.0),
        a.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 16.0), b.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        b.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 16.0), c.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        c.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 32.0), d.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        d.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 35.0), y.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        y.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
}

/// ステップ14 同一の値を３回加算した場合のテスト。
#[test]
fn test_add_same_input_3times() {
    common::setup();

    // 加算値をランダムに生成する。
    let x = Variable::new(RawVariable::new(2.0));

    // 加算した結果の期待値を計算する。
    let expected_output_data = Array::from_elem(IxDyn(&[]), 6.0);

    let result = add(add(x.clone(), x.clone()), x.clone());

    // 順伝播の結果を確認する。
    // 逆伝播の微分結果 grad が入力値に設定されていないことも確認する。
    // dbg!(x.clone());
    assert_eq!(
        expected_output_data.clone(),
        //results.clone().get(0).unwrap().borrow().clone().data
        // results.get(0).unwrap().borrow().data.clone()
        result.borrow().get_data()
    );

    // 逆伝播を実行する。
    result.backward();

    // 逆伝播の結果を確認する。
    // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
    // dbg!(x.clone());

    let expected_grad = Array::from_elem(IxDyn(&[]), 3.0);
    assert_eq!(
        expected_grad,
        x.borrow().get_grad().unwrap().borrow().get_data()
    );
    assert_eq!(expected_output_data.clone(), result.borrow().get_data());
}

/// ステップ14 微分のクリアに関するテスト
#[test]
fn test_clear_grad() {
    common::setup();

    // 加算値を生成する。
    let x = Variable::new(RawVariable::new(2.0));

    // 加算した結果の期待値を計算する。
    let expected_output_data = Array::from_elem(IxDyn(&[]), 4.0);

    let result = add(x.clone(), x.clone());

    // 逆伝播を実行する。
    result.backward();

    // 逆伝播の結果を確認する。
    // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
    // dbg!(x.clone());

    let expected_grad = Array::from_elem(IxDyn(&[]), 2.0);
    assert_eq!(
        expected_grad,
        x.borrow().get_grad().unwrap().borrow().get_data()
    );
    assert_eq!(expected_output_data.clone(), result.borrow().get_data());

    ////////////////////////////////
    // 微分をクリアせずにもう一度計算する。
    ////////////////////////////////
    let result2 = add(x.clone(), x.clone());

    // 逆伝播を実行する。
    result2.backward();

    // 逆伝播の結果を確認する。
    // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
    // dbg!(x.clone());

    // 1回目の微分と２回目の微分を加算した ４ になってしまうことを確認する。
    let expected_grad2 = Array::from_elem(IxDyn(&[]), 4.0);
    assert_eq!(
        expected_grad2,
        x.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(expected_output_data.clone(), result2.borrow().get_data());

    ////////////////////////////////
    // 微分をクリアしてもう一度計算する。
    ////////////////////////////////
    x.borrow_mut().clear_grad();
    let result3 = add(x.clone(), x.clone());

    // 逆伝播を実行する。
    result3.backward();

    // 逆伝播の結果を確認する。
    // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
    // dbg!(x.clone());
    // 微分をクリアしたことで正しい結果となることを確認する。
    let expected_grad3 = Array::from_elem(IxDyn(&[]), 2.0);
    assert_eq!(
        expected_grad3,
        x.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(expected_output_data.clone(), result2.borrow().get_data());
}

/// 二乗のテスト
#[test]
fn test_square() {
    common::setup();

    // 2乗する値をランダムに生成する。
    let mut rng = rand::rng();
    let rand_x = rng.random::<f64>();
    let x = Variable::new(RawVariable::new(rand_x));

    // 2乗した結果の期待値を計算する。
    let expected_output_data = Array::from_elem(IxDyn(&[]), rand_x * rand_x);
    let expected_grad_val = rand_x * 2.0 * 1.0;
    let expected_output_grad = Array::from_elem(IxDyn(&[]), expected_grad_val);

    // 順伝播、逆伝播を実行する。
    // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
    let result = square(x.clone());
    let creators = FunctionExecutor::extract_creators(vec![result.clone()]);

    // dbg!(&creators);

    // 実行した関数の数をチェックする。
    assert_eq!(1, creators.len());

    // 逆伝播を実行する。
    result.backward();

    // 二乗の結果
    assert_eq!(expected_output_data, result.borrow().get_data());
    // 逆伝播の結果
    assert_eq!(
        expected_output_grad,
        x.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
}

/// 加算のテスト
#[test]
fn test_add() {
    common::setup();

    // 加算値をランダムに生成する。
    let mut rng = rand::rng();
    let rand_x1 = rng.random::<f64>();
    let rand_x2 = rng.random::<f64>();
    let x1 = Variable::new(RawVariable::new(rand_x1));
    let x2 = Variable::new(RawVariable::new(rand_x2));

    // 加算した結果の期待値を計算する。
    let expected_output_data = Array::from_elem(IxDyn(&[]), rand_x1 + rand_x2);

    // 順伝播、逆伝播を実行する。
    // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
    let result = add(x1.clone(), x2.clone());
    let creators = FunctionExecutor::extract_creators(vec![result.clone()]);

    // dbg!(&creators);

    // 実行した関数の数をチェックする。
    assert_eq!(1, creators.len());

    // 逆伝播を実行する。
    result.backward();

    // 足し算の結果
    assert_eq!(expected_output_data, result.borrow().get_data());
    // 逆伝播の結果
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        x1.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        x2.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
}

/// Exp 関数のテスト。
#[test]
fn test_exp() {
    common::setup();

    let x = Variable::new(RawVariable::new(2.0));

    let e = std::f64::consts::E;
    let expected_output_data = Array::from_elem(IxDyn(&[]), e.powf(2.0));
    dbg!(expected_output_data.clone());

    // 順伝播、逆伝播を実行する。
    let result = exp(x.clone());
    let creators = FunctionExecutor::extract_creators(vec![result.clone()]);

    dbg!(&creators);

    // 実行した関数の数をチェックする。
    assert_eq!(1, creators.len());

    // 逆伝播を実行する。
    result.backward();

    // exp 結果
    assert_eq!(expected_output_data, result.borrow().get_data());
    // 逆伝播の結果 exp^x の微分は exp^x
    assert_eq!(
        expected_output_data,
        x.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );

    assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), x.borrow().get_data());
    // assert_eq!(
    //     Array::from_elem(IxDyn(&[]), 1.0),
    //     result.borrow().get_grad()
    // );
}

/// 2乗と加算のテスト
/// (x1 + x2)^2 の順伝播と逆伝播をテストする。
#[test]
fn test_add_square_1() {
    common::setup();

    // テスト用の入力値
    let x1_arr = Array::from_elem(IxDyn(&[]), 2.0);
    let x2_arr = Array::from_elem(IxDyn(&[]), 3.0);
    let x1 = Variable::new(RawVariable::new(x1_arr.clone()));
    let x2 = Variable::new(RawVariable::new(x2_arr.clone()));

    let expected = Array::from_elem(IxDyn(&[]), 25.0);

    // 関数を用意する。
    // let mut sq_exe = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
    // let mut add_exe = FunctionExecutor::new(Rc::new(RefCell::new(Add)));

    // 順伝播を実行する。
    // let results = sq_exe.forward(add_exe.forward(vec![x1.clone(), x2.clone()]));
    let result = square(add(x1.clone(), x2.clone()));

    // 順伝播の結果を確認する。
    // 逆伝播の微分結果 grad が入力値に設定されていないことも確認する。
    // dbg!(x1.clone());
    // dbg!(x2.clone());
    assert_eq!(x1_arr.clone(), x1.borrow().get_data());
    assert_eq!(x2_arr.clone(), x2.borrow().get_data());
    assert_eq!(true, x1.borrow().get_grad().is_none());
    assert_eq!(true, x2.borrow().get_grad().is_none());
    assert_eq!(
        expected.clone(),
        //results.clone().get(0).unwrap().borrow().clone().data
        // results.get(0).unwrap().borrow().data.clone()
        result.borrow().get_data()
    );

    // 逆伝播を実行する。
    result.backward();

    // 逆伝播の結果を確認する。
    // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
    // dbg!(x1.clone());
    // dbg!(x2.clone());

    let expected_grad = Array::from_elem(IxDyn(&[]), 10.0);
    assert_eq!(x1_arr.clone(), x1.borrow().get_data());
    assert_eq!(x2_arr.clone(), x2.borrow().get_data());
    assert_eq!(
        expected_grad,
        x1.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(
        expected_grad,
        x2.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(expected.clone(), result.borrow().get_data());
}

/// 2乗と加算のテスト
/// x1^2 + x2^2 の順伝播と逆伝播をテストする。
#[test]
fn test_add_square_2() {
    common::setup();

    // テスト用の入力値
    let x1_arr = Array::from_elem(IxDyn(&[]), 2.0);
    let x2_arr = Array::from_elem(IxDyn(&[]), 3.0);
    let x1 = Variable::new(RawVariable::new(x1_arr.clone()));
    let x2 = Variable::new(RawVariable::new(x2_arr.clone()));

    let expected = Array::from_elem(IxDyn(&[]), 13.0);

    // 関数を用意する。
    // let mut sq_exe_1 = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
    // let mut sq_exe_2 = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
    // let mut add_exe = FunctionExecutor::new(Rc::new(RefCell::new(Add)));

    // 順伝播の実行
    // let results = add_exe.forward(vec![
    //     sq_exe_1.forward(vec![x1.clone()]).get(0).unwrap().clone(),
    //     sq_exe_2.forward(vec![x2.clone()]).get(0).unwrap().clone(),
    // ]);
    let result = add(square(x1.clone()), square(x2.clone()));

    // 順伝播の結果を確認する。
    // 逆伝播の微分結果 grad が入力値に設定されていないことも確認する。
    //  dbg!(x1.clone());
    // dbg!(x2.clone());
    assert_eq!(x1_arr.clone(), x1.borrow().get_data());
    assert_eq!(x2_arr.clone(), x2.borrow().get_data());
    assert_eq!(true, x1.borrow().get_grad().is_none());
    assert_eq!(true, x2.borrow().get_grad().is_none());
    assert_eq!(
        expected.clone(),
        //results.clone().get(0).unwrap().borrow().clone().data
        result.borrow().get_data()
    );

    // 逆伝播を実行する。
    result.backward();

    // 逆伝播の結果を確認する。
    // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
    // dbg!(x1.clone());
    // dbg!(x2.clone());
    // dbg!(result.clone().borrow().generation);

    let expected_x1_grad = Array::from_elem(IxDyn(&[]), 4.0);
    let expected_x2_grad = Array::from_elem(IxDyn(&[]), 6.0);
    assert_eq!(x1_arr.clone(), x1.borrow().get_data());
    assert_eq!(x2_arr.clone(), x2.borrow().get_data());
    assert_eq!(
        expected_x1_grad,
        x1.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(
        expected_x2_grad,
        x2.borrow()
            .get_grad()
            .expect("No grad exist.")
            .borrow()
            .get_data()
    );
    assert_eq!(expected.clone(), result.borrow().get_data());
}

/// 2乗と加算のテスト
/// x1^2 + x2^2 の順伝播と逆伝播をテストする。
#[test]
fn test_square_exp_square() {
    common::setup();

    info!("info log");
    warn!("warn log");

    // テスト用の入力値
    let x_arr = Array::from_elem(IxDyn(&[]), 0.5);
    let x = Variable::new(RawVariable::new(x_arr.clone()));

    let e = std::f64::consts::E;
    let expected = Array::from_elem(IxDyn(&[]), e.powf(0.5 * 0.5) * e.powf(0.5 * 0.5));
    // dbg!(expected.clone());

    // 関数を用意する。
    // let mut sq_exe_1 = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
    // let mut exp_exe = FunctionExecutor::new(Rc::new(RefCell::new(Exp)));
    // let mut sq_exe_2 = FunctionExecutor::new(Rc::new(RefCell::new(Square)));

    // 順伝播の実行
    // let results = sq_exe_2.forward(exp_exe.forward(sq_exe_1.forward(vec![x.clone()])));
    let result = square(exp(square(x.clone())));

    // 順伝播の結果を確認する。
    // 逆伝播の微分結果 grad が入力値に設定されていないことも確認する。
    // dbg!(x.clone());
    assert_eq!(x_arr.clone(), x.borrow().get_data());
    assert_eq!(true, x.borrow().get_grad().is_none());
    assert_eq!(expected.clone(), result.borrow().get_data());

    // 逆伝播を実行する。
    result.backward();

    // 逆伝播の結果を確認する。
    // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
    // dbg!(x.clone());

    // 逆伝播の正解は書籍の値を使用。
    let expected_x_grad = Array::from_elem(IxDyn(&[]), 3.297442541400256);

    assert_eq!(x_arr.clone(), x.borrow().get_data());
    assert_eq!(
        expected_x_grad,
        x.borrow().get_grad().unwrap().borrow().get_data()
    );
    assert_eq!(expected.clone(), result.borrow().get_data());
}

#[test]
fn test_step33_second_differential() {
    common::setup();

    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_enabled();
    // バックプロパゲーションを行う。
    Setting::set_backprop_enabled();

    /// y = x^4 - 2x^2
    fn f<V: MathOps>(x: &Variable<V>) -> Variable<V> {
        let y = &(x ^ 4) - &(2 * &(x ^ 2));
        y
    }

    let mut x = Variable::new(RawVariable::new(2.0));
    x.borrow_mut().set_name("x".to_string());

    debug!("===== フォワード ======");

    let mut y = f(&x);
    y.borrow_mut().set_name("y".to_string());

    debug!("===== 1回目バックプロパゲーション======");
    y.backward();

    // let gx_creators = FunctionExecutor::extract_creators(vec![gx.clone()]);
    // gx_creators.iter().for_each(|c| {
    //     dbg!(&c);
    // });

    fn debug_variable<V: MathOps>(x: Variable<V>, indent_num: usize) {
        let indent = format!("{}", "  ".repeat(indent_num));

        println!("{}variable", indent);
        println!("{}  name: {:?}", indent, x.borrow().get_name());
        println!("{}  data: {:?}", indent, x.borrow().get_data());
        println!("{}  generation: {:?}", indent, x.borrow().get_generation());

        match x.borrow().get_grad() {
            Some(grad) => {
                println!("{}  grad", indent);
                debug_variable(grad.clone(), indent_num + 2usize);
            }
            _ => println!("{}  grad is None", indent),
        }
        let creator = x.borrow().get_creator();
        match creator {
            Some(creator) => {
                println!(
                    "{}  creator: {:?} gen: {:?}",
                    indent,
                    creator.borrow().get_creator().borrow().get_name(),
                    creator.borrow().get_generation()
                );
                println!("{}  inputs", indent);
                let inputs = creator.borrow().get_inputs();
                for input in inputs {
                    println!("{}    {:?}", indent, input.borrow().get_data());
                    debug_variable(input.clone(), indent_num + 2usize);
                }
                println!("{}  outputs", indent);
                let outputs = creator.borrow().get_outputs();
                for output in outputs {
                    let tmp_output = output.upgrade().unwrap();
                    println!("{}    {:?}", indent, tmp_output.borrow().get_data());
                    // debug_variable(
                    //     Variable::new(tmp_output.borrow().clone()),
                    //     format!("{}{}", indent, indent),
                    // );
                }
            }
            _ => println!("{}  creator is None.", indent),
        }
    }

    // debug_variable(gx.clone(), " ".to_string());
    // debug_variable(x.clone(), 0usize);

    // x.borrow_mut().clear_grad();
    // バックプロパゲーションを行わない。
    // Setting::set_backprop_disabled();
    // gx.borrow_mut().set_name("gx".to_string());

    debug!("===== 2回目バックプロパゲーション======");
    let gx = &x.borrow().get_grad().unwrap(); //.get_data(); //.clone();
    gx.backward();

    // let gx = x.borrow_mut().get_grad().unwrap();
    // gx.backward();
    // x.borrow_mut().get_grad().unwrap().backward();
    debug!(
        "gx data: {:?}  grad: {:?}",
        // x.borrow().get_grad().unwrap().borrow().get_data()
        // gx.borrow().get_grad().unwrap().borrow().get_data()
        gx.borrow().get_data(),
        gx.borrow().get_grad().unwrap().borrow().get_data() // x.borrow_mut()
                                                            //     .get_grad()
                                                            //     .unwrap()
                                                            //     .borrow()
                                                            //     .get_grad()
                                                            //     .unwrap()
                                                            //     .borrow()
                                                            //     .get_data()
    );

    debug_variable(gx.clone(), 0usize);
    //debug_variable(x, 0usize);

    //  dbg!(&gx);

    // let gx_creators2 = FunctionExecutor::extract_creators(vec![gx.clone()]);
    // gx_creators2.iter().for_each(|c| {
    //     dbg!(&c);
    // });

    // let creators = FunctionExecutor::extract_creators(vec![y.clone()]);

    // creators.iter().map(|c| dbg!(c));

    // let file_name = "test_step33_second_differential.png";

    // plot_dot_graph!(gx, file_name, true);
    // dbg!(&gx);
}

#[test]
fn test_step33_second_differential_2() {
    common::setup();

    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_enabled();
    // バックプロパゲーションを行う。
    Setting::set_backprop_enabled();

    /// y = x^4 - 2x^2
    fn f<V: MathOps>(x: Variable<V>) -> Variable<V> {
        // let y = &(&x ^ 4) - &(2 * &(&x ^ 2));
        let y = &(4 * &(&x ^ 3)) - &(4 * &x);
        y
    }

    let mut x = Variable::new(RawVariable::new(2.0));
    x.borrow_mut().set_name("x".to_string());

    debug!("===== フォワード ======");

    let mut y = f(x.clone());
    y.borrow_mut().set_name("y".to_string());

    debug!("===== 1回目バックプロパゲーション======");
    y.backward();
    let x_grad = x.borrow().get_grad().unwrap().borrow().get_data().clone();
    debug!("x grad: {:?}", x_grad);
}
