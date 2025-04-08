// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn, Shape};
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
    /// y=x0 * x1 の微分であるため、dy/dx0=x1 * gy, dy/dx1= x0 * gy である。
    fn backward(&self, inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("broadcast_to(backward)");
        let x0 = &inputs[0];
        let x1 = &inputs[1];
        let gx_x0 = x1 * &gys[0];
        let gx_x1 = x0 * &gys[0];
        debug!(
            "broadcast_to(backward) dy/dx0 = {:?} * {:?}",
            &x1.borrow().get_data().flatten().to_vec(),
            &gys[0].borrow().get_data().flatten().to_vec(),
        );
        debug!(
            "broadcast_to(backward) dy/dx1 = {:?} * {:?}",
            &x0.borrow().get_data().flatten().to_vec(),
            &gys[0].borrow().get_data().flatten().to_vec(),
        );
        let gxs = vec![gx_x0, gx_x1];

        debug!(
            "broadcast_to(backward) result: {:?} {:?}",
            gxs[0].borrow().get_data().flatten().to_vec(),
            gxs[1].borrow().get_data().flatten().to_vec()
        );

        gxs
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
    use std::f32::consts::PI as PIf32;

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
