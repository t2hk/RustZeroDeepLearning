//! ステップ20 演算子のオーバーロード(1)
mod modules;

use crate::modules::functions::*;
use crate::modules::math::*;
use crate::modules::settings::*;
use crate::modules::variable::*;

use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    println!("Rust Zero Deep Learning");

    ///////////// 実行に関する設定 /////////////
    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_enabled();

    // バックプロパゲーションを行う設定
    Setting::set_backprop_enabled();

    // 入力値を用意する。
    let x1 = Rc::new(RefCell::new(RawVariable::new(2.0)));
    let x2 = Rc::new(RefCell::new(RawVariable::new(3.0)));

    // 計算する。
    let a = square(Rc::clone(&x1));
    let b = square(Rc::clone(&a));
    let c = square(Rc::clone(&a));
    let d = add(Rc::clone(&b), Rc::clone(&c));
    let y = add(Rc::clone(&d), Rc::clone(&x2));

    // 順伝播の結果を確認する。
    assert_eq!(Array::from_elem(IxDyn(&[]), 35.0), y.borrow().get_data());

    // 各変数の世代を確認する。
    assert_eq!(0, x1.borrow().get_generation());
    assert_eq!(0, x2.borrow().get_generation());
    assert_eq!(1, a.borrow().get_generation());
    assert_eq!(2, b.borrow().get_generation());
    assert_eq!(2, c.borrow().get_generation());
    assert_eq!(3, d.borrow().get_generation());
    assert_eq!(4, y.borrow().get_generation());

    // 各関数の世代を確認する。
    assert_eq!(0, a.borrow().get_creator_generation());
    assert_eq!(1, b.borrow().get_creator_generation());
    assert_eq!(1, c.borrow().get_creator_generation());
    assert_eq!(2, d.borrow().get_creator_generation());
    assert_eq!(3, y.borrow().get_creator_generation());

    // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
    let creators = FunctionExecutor::extract_creators(vec![y.clone()]);
    // 実行した関数の数をチェックする。
    assert_eq!(5, creators.len());

    // 逆伝播を実行する。
    y.as_ref().clone().borrow().backward();

    // 逆伝播の結果の確認
    // 途中結果の変数には微分値が設定されていないことを確認する。
    assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), x1.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 64.0),
        x1.borrow().get_grad().expect("No grad exist.")
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 3.0), x2.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        x2.borrow().get_grad().expect("No grad exist.")
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 4.0), a.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 16.0),
        a.borrow().get_grad().expect("No grad exist.")
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 16.0), b.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        b.borrow().get_grad().expect("No grad exist.")
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 16.0), c.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        c.borrow().get_grad().expect("No grad exist.")
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 32.0), d.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        d.borrow().get_grad().expect("No grad exist.")
    );
    assert_eq!(Array::from_elem(IxDyn(&[]), 35.0), y.borrow().get_data());
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        y.borrow().get_grad().expect("No grad exist.")
    );
}
