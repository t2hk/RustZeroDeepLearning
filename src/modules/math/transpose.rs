// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn, Shape};
use std::cell::RefCell;
use std::rc::Rc;

/// 転置関数
#[derive(Debug, Clone)]
pub struct TransposeFunction {}

impl<V: MathOps> Function<V> for TransposeFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Transpose".to_string()
    }

    /// 順伝播
    ///
    /// Arguments
    /// * xs (Vec<Array<V,IxDyn>>): 変換対象のテンソル
    ///
    /// Returns
    /// * Vec<Array<V, IxDyn>>: 転置の結果
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("transpose(forward)");
        let y = xs[0].clone().t().to_owned();

        debug!("transpose(forwad) {:?} -> {:?}", &xs[0].shape(), y.shape());

        return vec![y];
    }

    /// 逆伝播
    fn backward(&self, _inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("transpose(backward)");

        let t_gy = transpose(gys[0].clone());

        debug!(
            "transpose(backward) {:?} -> {:?}",
            gys[0].borrow().get_data().shape(),
            t_gy.borrow().get_data().shape(),
        );

        vec![t_gy]
    }
}

/// 転置関数
///
/// Arguments
/// * input (Variable<V>): 転置対象
///
/// Return
/// * Variable<V>: 転置の結果
pub fn transpose<V: MathOps>(input: Variable<V>) -> Variable<V> {
    let mut transpose = FunctionExecutor::new(Rc::new(RefCell::new(TransposeFunction {})));

    // 順伝播
    transpose.forward(vec![input]).get(0).unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 行列の転置のテスト
    #[test]
    fn test_transpose() {
        let input_shape = vec![2, 3];
        let x = Variable::new(RawVariable::from_shape_vec(
            input_shape.clone(),
            vec![1, 2, 3, 4, 5, 6],
        ));
        dbg!(&x);

        let y = transpose(x.clone());

        // 転置後の確認
        assert_eq!(vec![3, 2], y.borrow().get_data().shape().to_vec());
        assert_eq!(
            vec![1, 4, 2, 5, 3, 6],
            y.borrow().get_data().flatten().to_vec()
        );

        y.backward();

        let x_grad = x.borrow().get_grad().unwrap();
        dbg!(&x_grad);

        // 微分値が入力値と同じ形状で、全て1であることを確認する。
        assert_eq!(input_shape, x_grad.borrow().get_data().shape().to_vec());
        assert_eq!(
            vec![1, 1, 1, 1, 1, 1],
            x_grad.borrow().get_data().flatten().to_vec()
        );
    }
}
