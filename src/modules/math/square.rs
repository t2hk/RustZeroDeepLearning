// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// 二乗関数
#[derive(Debug, Clone)]
pub struct SquareFunction;
impl<V: MathOps> Function<V> for SquareFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Square".to_string()
    }

    /// 順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("square(forward)");
        debug!("square(forward): {:?} ^2", xs[0].flatten().to_vec());
        let result = vec![xs[0].mapv(|x| x.clone() * x.clone())];

        result
    }

    /// 逆伝播
    /// y=x^2 の微分であるため、dy/dx=2x である。
    fn backward(&self, inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("square(backward)");
        debug!(
            "square(backward): 2 * {:?} * {:?}",
            &inputs[0].borrow().get_data().flatten().to_vec(),
            &gys[0].borrow().get_data().flatten().to_vec()
        );
        // let x = inputs[0].borrow().get_data();
        //let x_gys = &gys[0].clone() * &x;
        let x_gys = &gys[0] * &inputs[0];
        let gxs = vec![&x_gys * &Variable::new(RawVariable::new(V::from(2).unwrap()))];
        // let gxs = vec![Variable::new(RawVariable::new(
        //     x_gys.borrow().get_data().mapv(|x| x * V::from(2).unwrap()),
        // ))];
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
