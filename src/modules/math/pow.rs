// ライブラリを一括でインポート
use crate::modules::math::*;

use core::fmt::Debug;
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::ops::BitXor;
use std::rc::Rc;

/// 累乗関数
#[derive(Debug, Clone)]
pub struct PowFunction {
    exp: usize, // 指数
}

impl<V: MathOps> Function<V> for PowFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Pow".to_string()
    }

    /// 順伝播
    /// 関数のインスタンス作成時に指数 exp: usize を設定しておくこと。
    ///
    /// Arguments
    /// * xs (Vec<Array<V,IxDyn>>): 基数
    ///
    /// Returns
    /// * Vec<Array<V, IxDyn>>: 累乗の結果
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        let x0 = &xs[0];
        debug!("pow: {:?} ^ {:?}", &x0[[]], &self.exp);

        let result = x0.mapv(|x| num_traits::pow(V::from(x).unwrap(), self.exp));
        vec![result]
    }

    /// 逆伝播
    /// y=x^exp の微分であるため、dy/dx = exp * x^(exp-1) である。
    fn backward(&self, inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        let x = inputs[0].borrow().get_data();

        let x_diff_exp = x.mapv(|x| {
            num_traits::pow(V::from(x).unwrap(), self.exp - 1) * V::from(self.exp).unwrap()
        });

        let gxs = &gys[0].clone() * &x_diff_exp;
        vec![gxs]
    }
}

/// 累乗関数
///
/// Arguments
/// * input (Variable<V>): 基数
/// * exp (usize): 指数
///
/// Return
/// * Variable<V>: 累乗の結果
pub fn pow<V: MathOps>(input: Variable<V>, exp: usize) -> Variable<V> {
    let mut pow = FunctionExecutor::new(Rc::new(RefCell::new(PowFunction { exp: exp })));

    // 順伝播
    pow.forward(vec![input]).get(0).unwrap().clone()
}

impl<V: MathOps> BitXor<usize> for &Variable<V> {
    type Output = Variable<V>;
    fn bitxor(self, exp: usize) -> Variable<V> {
        // 順伝播
        let mut pow = FunctionExecutor::new(Rc::new(RefCell::new(PowFunction { exp: exp })));
        let result = pow.forward(vec![self.clone()]).get(0).unwrap().clone();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    /// 累乗のテスト(f64)
    /// [[1.0,2.0],[3.0,4.0]] の3乗
    #[test]
    fn test_pow_f64() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 2],
            vec![1.0f64, 2.0f64, 3.0f64, 4.0f64],
        ));
        let expect = Array::from_shape_vec(vec![2, 2], vec![1.0, 8.0, 27.0, 64.0]).unwrap();
        let result = pow(x.clone(), 3);
        assert_eq!(expect, result.borrow().get_data());

        // 微分
        // [[3., 12.], [27., 48.]]
        result.backward();
        dbg!(&result);
        dbg!(&x);
        let expect_grad = Array::from_shape_vec(vec![2, 2], vec![3.0, 12.0, 27.0, 48.0]).unwrap();
        assert_eq!(
            expect_grad,
            x.borrow().get_grad().unwrap().borrow().get_data()
        );
    }

    /// 累乗のテスト(i32)
    /// [[1.0,2.0],[3.0,4.0]] の3乗
    #[test]
    fn test_pow_i32() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 2],
            vec![1i32, 2i32, 3i32, 4i32],
        ));
        let expect = Array::from_shape_vec(vec![2, 2], vec![1, 8, 27, 64]).unwrap();
        let result = pow(x.clone(), 3);
        assert_eq!(expect, result.borrow().get_data());

        // 微分
        // [[3, 12], [27, 48]]
        result.backward();
        dbg!(&result);
        dbg!(&x);
        let expect_grad =
            Array::from_shape_vec(vec![2, 2], vec![3i32, 12i32, 27i32, 48i32]).unwrap();
        assert_eq!(
            expect_grad,
            x.borrow().get_grad().unwrap().borrow().get_data()
        );
    }

    /// オーバーロードした累乗のテスト(i32)
    /// [[1.0,2.0],[3.0,4.0]] の3乗
    #[test]
    fn test_pow_i32_overload() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 2],
            vec![1i32, 2i32, 3i32, 4i32],
        ));
        let expect = Array::from_shape_vec(vec![2, 2], vec![1, 8, 27, 64]).unwrap();

        let result = &x ^ 3;
        assert_eq!(expect, result.borrow().get_data());

        // 微分
        // [[3, 12], [27, 48]]
        result.backward();
        dbg!(&result);
        dbg!(&x);
        let expect_grad =
            Array::from_shape_vec(vec![2, 2], vec![3i32, 12i32, 27i32, 48i32]).unwrap();
        assert_eq!(
            expect_grad,
            x.borrow().get_grad().unwrap().borrow().get_data()
        );
    }
}
