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
    let x1 = Variable::new(Rc::new(RefCell::new(RawVariable::new(2.0))));
    let x2 = Variable::new(Rc::new(RefCell::new(RawVariable::new(3.0))));

    // 計算する。
    // let a = square(x1.clone());
    let a = square(x1.clone());
    let b = square(a.clone());
    let c = square(a.clone());
    let d = add(b.clone(), c.clone());
    let y = add(d.clone(), x2.clone());

    // 順伝播の結果を確認する。
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 35.0),
        y.raw().borrow().get_data()
    );

    // 各変数の世代を確認する。
    assert_eq!(0, x1.raw().borrow().get_generation());
    assert_eq!(0, x2.raw().borrow().get_generation());
    assert_eq!(1, a.raw().borrow().get_generation());
    assert_eq!(2, b.raw().borrow().get_generation());
    assert_eq!(2, c.raw().borrow().get_generation());
    assert_eq!(3, d.raw().borrow().get_generation());
    assert_eq!(4, y.raw().borrow().get_generation());

    // 各関数の世代を確認する。
    assert_eq!(0, a.raw().borrow().get_creator_generation());
    assert_eq!(1, b.raw().borrow().get_creator_generation());
    assert_eq!(1, c.raw().borrow().get_creator_generation());
    assert_eq!(2, d.raw().borrow().get_creator_generation());
    assert_eq!(3, y.raw().borrow().get_creator_generation());

    // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
    let creators = FunctionExecutor::extract_creators(vec![y.clone()]);
    // 実行した関数の数をチェックする。
    assert_eq!(5, creators.len());

    // 逆伝播を実行する。
    y.raw().as_ref().clone().borrow().backward();

    // 逆伝播の結果の確認
    // 途中結果の変数には微分値が設定されていないことを確認する。
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 2.0),
        x1.raw().borrow().get_data()
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 64.0),
        x1.raw().borrow().get_grad().expect("No grad exist.")
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 3.0),
        x2.raw().borrow().get_data()
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        x2.raw().borrow().get_grad().expect("No grad exist.")
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 4.0),
        a.raw().borrow().get_data()
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 16.0),
        a.raw().borrow().get_grad().expect("No grad exist.")
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 16.0),
        b.raw().borrow().get_data()
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        b.raw().borrow().get_grad().expect("No grad exist.")
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 16.0),
        c.raw().borrow().get_data()
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        c.raw().borrow().get_grad().expect("No grad exist.")
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 32.0),
        d.raw().borrow().get_data()
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        d.raw().borrow().get_grad().expect("No grad exist.")
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 35.0),
        y.raw().borrow().get_data()
    );
    assert_eq!(
        Array::from_elem(IxDyn(&[]), 1.0),
        y.raw().borrow().get_grad().expect("No grad exist.")
    );
}
