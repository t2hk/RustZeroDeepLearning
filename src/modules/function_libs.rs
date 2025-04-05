// ライブラリを一括でインポート
#[allow(unused_imports)]
use crate::modules::math::*;
use crate::modules::*;
#[allow(unused_imports)]
use core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

/// Sphere 関数
/// z = x^2 + y^2 を計算する。
///
/// Arguments:
/// * x (Variable<V>)
/// * y (Variable<V>)
/// Returns:
/// * Variable<V>: Sphere 関数の計算結果
pub fn sphere<V: MathOps>(x: Variable<V>, y: Variable<V>) -> Variable<V> {
    let z = &(&x ^ 2) + &(&y ^ 2);
    z
}

/// matyas 関数
/// z = 0.26 * (x^2 + y^2) - 0.48 * x * y を計算する。
///
/// Arguments:
/// * x (Variable<V>)
/// * y (Variable<V>)
/// Returns:
/// * Variable<V>: matyas 関数の計算結果
pub fn matyas<V: MathOps>(x: Variable<V>, y: Variable<V>) -> Variable<V> {
    let z = &(0.26 * &(&(&x ^ 2) + &(&y ^ 2))) - &(0.48 * &(&x * &y));
    z
}

/// Goldstein-Price 関数
/// z = [1 + (x * y * 1)^2 (19 - 14x + 3x^2 -14y + 6xy + 3y^2)][30 + (2x - 3y)^2 (18 - 32x + 12x^2 + 48y -36xy + 27y^2)] を計算する。
///
/// Arguments:
/// * x (Variable<V>)
/// * y (Variable<V>)
/// Returns:
/// * Variable<V>: matyas 関数の計算結果
pub fn goldstein<V: MathOps>(x: Variable<V>, y: Variable<V>) -> Variable<V> {
    // let a = &(&(&x + &y) + 1) ^ 2usize;
    // let b = 14 * &x;
    // let c = 3 * &(&x ^ 2);
    // let d = 14 * &y;
    // let e = &(6 * &x) * &y;
    // let f = 3 * &(&y ^ 2);
    // let a_f = &(1 + &(&a * &(&(&(&(&(19 - &b) + &c) - &d) + &e) + &f)));

    // let g = &(&(2 * &x) - &(3 * &y)) ^ 2usize;
    // let h = 18 - &(32 * &x);
    // let i = 12 * &(&x ^ 2usize);
    // let j = &(48 * &y) - &(36 * &(&x * &y));
    // let k = 27 * &(&y ^ 2usize);

    // let g_k = &(30 + &(&g * (&(&(&(&h + &i) + &j) + &k))));

    // let z = a_f * g_k;

    // let a_f = &(1 + &(&(&(&(&x + &y) + 1) ^ 2usize) * &(&(&(&(&(19 - &(14 * &x)) + &(3 * &(&x ^ 2))) - &(14 * &y)) + &(&(6 * &x) * &y)) + &(3 * &(&y ^ 2)))));
    // let g_k = &(30 + &(&(&(&(2 * &x) - &(3 * &y)) ^ 2usize) * (&(&(&(&(18 - &(32 * &x)) + &(12 * &(&x ^ 2usize))) + &(&(48 * &y) - &(36 * &(&x * &y)))) + &(27 * &(&y ^ 2usize))))));

    let z = &(1 + &(&(&(&(&x + &y) + 1) ^ 2)
        * &(&(&(&(&(19 - &(14 * &x)) + &(3 * &(&x ^ 2))) - &(14 * &y)) + &(&(6 * &x) * &y))
            + &(3 * &(&y ^ 2)))))
        * &(30
            + &(&(&(&(2 * &x) - &(3 * &y)) ^ 2)
                * (&(&(&(&(18 - &(32 * &x)) + &(12 * &(&x ^ 2)))
                    + &(&(48 * &y) - &(36 * &(&x * &y))))
                    + &(27 * &(&y ^ 2))))));

    z
}

/// ローゼンブロック関数
pub fn rosenblock<V: MathOps>(x0: Variable<V>, x1: Variable<V>) -> Variable<V> {
    let lhs = 100 * &(&(&x1 - &(&x0 ^ 2)) ^ 2);
    let rhs = &(&x0 - 1) ^ 2;

    &lhs + &rhs
}

fn type_of<T>(_: T) -> String {
    let a = std::any::type_name::<T>();
    return a.to_string();
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, IxDyn};
    use rand::prelude::*;

    /// ステップ29 ニュートン法による最適化の実装
    #[test]
    fn test_step29_newton_method() {
        /// y = f(x)
        fn f<V: MathOps>(x: Variable<V>) -> Variable<V> {
            let y = &(&x ^ 4) - &(2 * &(&x ^ 2));
            return y;
        }

        /// y = f(x) の２階微分
        fn gx2<V: MathOps>(x: Variable<V>) -> Variable<V> {
            let y = &(12 * &(&x ^ 2)) - 4;
            return y;
        }

        let mut x = Variable::new(RawVariable::new(2.0));
        let iters = 10;

        for i in 0..iters {
            println!("i: {} x: {:?}", i, x.borrow().get_data());
            // 書籍と同じ値になるかテストする。
            match i {
                0 => assert_eq!(2.0, x.borrow().get_data()[[]]),
                1 => assert_eq!(1.4545454545454546, x.borrow().get_data()[[]]),
                2 => assert_eq!(1.1510467893775467, x.borrow().get_data()[[]]),
                3 => assert_eq!(1.0253259289766978, x.borrow().get_data()[[]]),
                4 => assert_eq!(1.0009084519430513, x.borrow().get_data()[[]]),
                5 => assert_eq!(1.0000012353089454, x.borrow().get_data()[[]]),
                6 => assert_eq!(1.000000000002289, x.borrow().get_data()[[]]),
                7 => assert_eq!(1.0, x.borrow().get_data()[[]]),
                8 => assert_eq!(1.0, x.borrow().get_data()[[]]),
                9 => assert_eq!(1.0, x.borrow().get_data()[[]]),
                _ => {}
            }

            let y = f(x.clone());
            x.borrow_mut().clear_grad();

            y.backward();
            let x_data = Variable::new(RawVariable::new(x.borrow().get_data()));
            let x_grad = x.borrow().get_grad().unwrap().borrow().get_data()[[]];

            let new_data: Variable<f64> = &x_data - &(x_grad / &gx2(x_data.clone()));

            x.set_data(new_data.borrow().get_data());
        }
    }

    #[test]
    /// ローゼンブロック関数のテスト
    fn test_rosenblock() {
        let x0 = Variable::new(RawVariable::new(0.0));
        let x1 = Variable::new(RawVariable::new(2.0));

        let y = rosenblock(x0.clone(), x1.clone());
        y.backward();

        let expected_x0_grad = Array::from_elem(IxDyn(&[]), -2.0);
        let expected_x1_grad = Array::from_elem(IxDyn(&[]), 400.0);

        assert_eq!(
            expected_x0_grad,
            x0.borrow().get_grad().unwrap().borrow().get_data()
        );
        assert_eq!(
            expected_x1_grad,
            x1.borrow().get_grad().unwrap().borrow().get_data()
        );
    }

    /// ローゼンブロック関数の勾配降下法
    #[test]
    fn test_step28() {
        let mut x0 = Variable::new(RawVariable::new(0.0));
        let mut x1 = Variable::new(RawVariable::new(2.0));
        let lr = 0.001;
        let iters = 10000;

        for _i in 0..iters {
            println!(
                "x0: {:?} x1: {:?}",
                x0.borrow().get_data(),
                x1.borrow().get_data()
            );

            let y = rosenblock(x0.clone(), x1.clone());
            x0.borrow_mut().clear_grad();
            x1.borrow_mut().clear_grad();
            y.backward();

            let x0_data = x0.borrow().get_data();
            let x1_data = x1.borrow().get_data();
            let x0_grad = x0.borrow().get_grad().unwrap().borrow().get_data()[[]];
            let x1_grad = x1.borrow().get_grad().unwrap().borrow().get_data()[[]];

            x0.set_data(x0_data - lr * x0_grad);
            x1.set_data(x1_data - lr * x1_grad);
        }
    }

    /// Sphere 関数のテスト
    #[test]
    fn test_sphere_1() {
        let x = Variable::new(RawVariable::new(1));
        let y = Variable::new(RawVariable::new(1));
        let z = sphere(x.clone(), y.clone());

        z.backward();

        let expected = Array::from_elem(IxDyn(&[]), 2);
        let expect_x_grad = Array::from_elem(IxDyn(&[]), 2);
        let expect_y_grad = Array::from_elem(IxDyn(&[]), 2);
        assert_eq!(expected, z.borrow().get_data());
        assert_eq!(
            expect_x_grad,
            x.borrow()
                .get_grad()
                .expect("No grad exist.")
                .borrow()
                .get_data()
        );
        assert_eq!(
            expect_y_grad,
            y.borrow()
                .get_grad()
                .expect("No grad exist.")
                .borrow()
                .get_data()
        );
    }

    /// matyas 関数のテスト
    #[test]
    fn test_matyas_1() {
        let x = Variable::new(RawVariable::new(1.0));
        let y = Variable::new(RawVariable::new(1.0));
        let z = matyas(x.clone(), y.clone());

        z.backward();

        let expected = Array::from_elem(IxDyn(&[]), 0.040000000000000036);
        let expect_x_grad = Array::from_elem(IxDyn(&[]), 0.040000000000000036);
        let expect_y_grad = Array::from_elem(IxDyn(&[]), 0.040000000000000036);
        assert_eq!(expected, z.borrow().get_data());
        assert_eq!(
            expect_x_grad,
            x.borrow()
                .get_grad()
                .expect("No grad exist.")
                .borrow()
                .get_data()
        );
        assert_eq!(
            expect_y_grad,
            y.borrow()
                .get_grad()
                .expect("No grad exist.")
                .borrow()
                .get_data()
        );
    }

    /// goldstein 関数のテスト
    #[test]
    fn test_goldstein_1() {
        let x = Variable::new(RawVariable::new(1));
        let y = Variable::new(RawVariable::new(1));
        let z = goldstein(x.clone(), y.clone());

        z.backward();

        let expected = Array::from_elem(IxDyn(&[]), 1876);
        let expect_x_grad = Array::from_elem(IxDyn(&[]), -5376);
        let expect_y_grad = Array::from_elem(IxDyn(&[]), 8064);

        assert_eq!(expected, z.borrow().get_data());
        assert_eq!(
            expect_x_grad,
            x.borrow()
                .get_grad()
                .expect("No grad exist.")
                .borrow()
                .get_data()
        );
        assert_eq!(
            expect_y_grad,
            y.borrow()
                .get_grad()
                .expect("No grad exist.")
                .borrow()
                .get_data()
        );
    }
}
