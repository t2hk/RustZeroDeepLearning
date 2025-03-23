// ライブラリを一括でインポート
use crate::modules::math::*;

use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::ops::{Mul, Neg};
use std::rc::Rc;

/// 二乗関数
#[derive(Debug, Clone)]
pub struct SquareFunction;
impl<V: MathOps> Function<V> for SquareFunction {
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

/// 二乗関数
///
/// Arguments
/// * input (Variable<V>): 加算する変数
///
/// Return
/// * Variable<V>: 二乗の結果
pub fn square<V: MathOps>(input: Variable<V>) -> Variable<V> {
    let mut square = FunctionExecutor::new(Rc::new(RefCell::new(SquareFunction)));
    // 二乗の順伝播
    square.forward(vec![input]).get(0).unwrap().clone()
}
#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    /// 二乗のテスト
    #[test]
    fn test_square() {
        // 2乗する値をランダムに生成する。
        let mut rng = rand::rng();
        let rand_x = rng.random::<f64>();
        let x = Variable::new(RawVariable::new(rand_x));

        // 2乗した結果の期待値を計算する。
        let expected_output_data = Array::from_elem(IxDyn(&[]), rand_x * rand_x);

        // 順伝播実行する。
        let result = square(x);

        // 二乗の結果
        assert_eq!(expected_output_data, result.borrow().get_data());
    }
}
