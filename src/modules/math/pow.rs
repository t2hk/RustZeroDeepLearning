// ライブラリを一括でインポート
use crate::modules::math::*;

use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use num_traits::Pow;
use std::cell::RefCell;
use std::rc::Rc;

/// 累乗関数
#[derive(Debug, Clone)]
pub struct PowFunction;
impl<V: MathOps> Function<V> for PowFunction {
    /// 順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        let result = vec![xs[0].mapv(|x| x * x)];

        result
    }

    /// 逆伝播
    /// y=x^2 の微分であるため、dy/dx=2x である。
    fn backward(
        &self,
        inputs: Vec<Variable<V>>,
        gys: Vec<Array<V, IxDyn>>,
    ) -> Vec<Array<V, IxDyn>> {
        let x = inputs[0].borrow().get_data();
        let x_gys = &gys[0].clone() * &x;
        let gxs = vec![x_gys.mapv(|x| x * V::from(2).unwrap())];
        gxs
    }
}

/// 累乗関数
///
/// Arguments
/// * input (Variable<V>): 加算する変数
///
/// Return
/// * Variable<V>: 累乗の結果
pub fn pow<V: MathOps>(input: Variable<V>) -> Variable<V> {
    let mut pow = FunctionExecutor::new(Rc::new(RefCell::new(PowFunction)));
    // 二乗の順伝播
    pow.forward(vec![input]).get(0).unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    #[test]
    /// 累乗のテスト(f32)
    fn test_pow_1() {}
}
