// ライブラリを一括でインポート
use crate::modules::functions::*;
use crate::modules::math::*;
use crate::modules::settings::*;
use crate::modules::variable::*;

use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// Sphere 関数
/// x^2 + y^2 を計算する。
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    /// Sphere 関数のテスト
    #[test]
    fn test_sphere_1() {
        let x = Variable::new(RawVariable::new(1));
        let y = Variable::new(RawVariable::new(1));
        let z = sphere(x.clone(), y.clone());

        z.backward();

        let expect_x_grad = Array::from_elem(IxDyn(&[]), 2);
        let expect_y_grad = Array::from_elem(IxDyn(&[]), 2);
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
