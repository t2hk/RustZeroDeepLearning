mod modules;

// ライブラリを一括でインポート
use rust_zero_deeplearning::*;

#[allow(unused_imports)]
use ::core::fmt::Debug;
use env_logger;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::env;

fn main() {
    env::set_var("RUST_LOG", "debug");
    env_logger::init();
    println!("Rust Zero Deep Learning");

    // 各レベルのログ出力
    trace!("トレースメッセージ");
    debug!("デバッグメッセージ");
    info!("情報メッセージ");
    warn!("警告メッセージ");
    error!("エラーメッセージ");

    ///////////// 実行に関する設定 /////////////
    // 逆伝播を実行する。微分値を保持する。
    Setting::set_retain_grad_enabled();

    // バックプロパゲーションを行う設定
    Setting::set_backprop_enabled();

    ///////////// 入力値を用意する。 /////////////
    let x1 = Variable::new(RawData::new(2.0));
    let x2 = Variable::new(RawData::new(3.0));

    ///////////// 計算する(順伝播)。 /////////////
    // 入力値を clone して計算用のメソッドに渡す。
    // 加算と乗算についてはオーバーロードに対応しており、
    // &a + &b のように記述できる。
    let a = square(x1.clone());
    let b = square(a.clone());
    let c = square(a.clone());
    let d = add(b.clone(), c.clone());
    // オーバーロードで加算する。
    let y = &d + &x2;

    dbg!(&y);

    // 順伝播の結果を確認する。
    // x1^2^2 + x1^2^2 + x2 = 35
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
    // y.as_ref().clone().borrow().backward();
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

    // 階乗
    let num = 1;
    let result = factorial(num);
    let result_naive = factorial(num);
    assert_eq!(result_naive, result);
}
