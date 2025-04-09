// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// ブロードキャスト関数
#[derive(Debug, Clone)]
pub struct BroadcastToFunction {
    x_shape: Vec<usize>,
    shape: Vec<usize>,
}
impl<V: MathOps> Function<V> for BroadcastToFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "BroadcastTo".to_string()
    }

    // BroadcastTo の順伝播
    fn forward(&self, x: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("broadcast_to(forward)");
        debug!(
            "broadcast_to(backward) {:?} -> {:?}",
            self.x_shape.clone(),
            self.shape.clone()
        );

        let y = x[0].broadcast(self.shape.clone()).unwrap().to_owned();
        vec![y]
    }

    /// 逆伝播
    fn backward(&self, _inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("broadcast_to(backward)");

        let gx = sum_to(gys[0].clone(), self.x_shape.clone());
        debug!(
            "broadcast_to(backward) {:?} -> {:?}",
            gys[0].borrow().get_data().shape(),
            gx.borrow().get_data().shape()
        );

        vec![gx]
    }
}

/// ブロードキャスト関数
///
/// Arguments
/// * x (Variable<V>): 対象の変数
/// * shape (Vec<usize>): ブロードキャストする形状
///
/// Return
/// * : 結果
pub fn broadcast_to<V: MathOps>(x: Variable<V>, shape: Vec<usize>) -> Variable<V> {
    let x_shape = x.borrow().get_data().shape().to_vec();
    let mut broadcasto_to = FunctionExecutor::new(Rc::new(RefCell::new(BroadcastToFunction {
        x_shape: x_shape,
        shape: shape,
    })));
    // 順伝播
    broadcasto_to
        .forward(vec![x.clone()])
        .get(0)
        .unwrap()
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// broadcast_to 関数のテスト。
    #[test]
    fn test_broadcast_to_1() {
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![1, 6],
            vec![1, 2, 3, 4, 5, 6],
        ));

        let y = broadcast_to(x, vec![6, 6]);
        dbg!(&y);
    }
}
