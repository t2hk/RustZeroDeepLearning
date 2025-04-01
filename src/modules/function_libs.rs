// ライブラリを一括でインポート
use crate::modules::math::*;
use crate::modules::*;

use core::fmt::Debug;

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

    let z = &(1 + &(&(&(&(&x + &y) + 1) ^ 2usize)
        * &(&(&(&(&(19 - &(14 * &x)) + &(3 * &(&x ^ 2))) - &(14 * &y)) + &(&(6 * &x) * &y))
            + &(3 * &(&y ^ 2)))))
        * &(30
            + &(&(&(&(2 * &x) - &(3 * &y)) ^ 2usize)
                * (&(&(&(&(18 - &(32 * &x)) + &(12 * &(&x ^ 2usize)))
                    + &(&(48 * &y) - &(36 * &(&x * &y))))
                    + &(27 * &(&y ^ 2usize))))));

    z
}

/// ローゼンブロック関数
pub fn rosenblock<V: MathOps>(x0: Variable<V>, x1: Variable<V>) -> Variable<V> {
    let lhs = 100 * &(&(&x1 - &(&x0 ^ 2)) ^ 2);
    let rhs = &(&x0 - 1) ^ 2usize;

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

    #[test]
    /// ローゼンブロック関数のテスト
    fn test_rosenblock() {
        let x0 = Variable::new(RawVariable::new(0.0));
        let x1 = Variable::new(RawVariable::new(2.0));

        let y = rosenblock(x0.clone(), x1.clone());
        y.backward();

        let expected_x0_grad = Array::from_elem(IxDyn(&[]), -2.0);
        let expected_x1_grad = Array::from_elem(IxDyn(&[]), 400.0);

        assert_eq!(expected_x0_grad, x0.borrow().get_grad().unwrap());
        assert_eq!(expected_x1_grad, x1.borrow().get_grad().unwrap());
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
            let x0_grad = x0.borrow().get_grad().unwrap()[[]];
            let x1_grad = x1.borrow().get_grad().unwrap()[[]];

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
            x.borrow().get_grad().expect("No grad exist.")
        );
        assert_eq!(
            expect_y_grad,
            y.borrow().get_grad().expect("No grad exist.")
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
            x.borrow().get_grad().expect("No grad exist.")
        );
        assert_eq!(
            expect_y_grad,
            y.borrow().get_grad().expect("No grad exist.")
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
            x.borrow().get_grad().expect("No grad exist.")
        );
        assert_eq!(
            expect_y_grad,
            y.borrow().get_grad().expect("No grad exist.")
        );
    }
}
