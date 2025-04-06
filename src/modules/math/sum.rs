// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, Axis, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// 加算関数
#[derive(Debug, Clone)]
pub struct SumFunction {
    x_shape: Option<Vec<usize>>,
    axis: Option<Axis>,
    keepdims: bool,
}

impl<V: MathOps> Function<V> for SumFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Sum".to_string()
    }

    // Sum (加算) の順伝播
    fn forward(&self, x: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("sum(forward)");
        debug!("sum(forward) {:?}", &x[0].flatten().to_vec());

        if let Some(axis) = self.axis {
            vec![x[0].sum_axis(axis)]
        } else {
            vec![Array::from_elem(IxDyn(&[]), x[0].sum())]
        }
    }

    /// 逆伝播
    /// y=x0+x1 の微分であるため、dy/dx0=1, dy/dx1=1 である。
    fn backward(&self, _inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("sum(backward)");

        let mut ndim = None;

        if let Some(x_shape) = self.x_shape.clone() {
            ndim = Some(x_shape.len());
        }

        /*
        let mut tupled_axis = None;
        if let Some(axis) = self.axis {
            tupled_axis = self.axis.clone();
        }
        */
        let mut actual_axis;

        // 次元を保持する場合、何もしない。
        if ndim == None || ndim == Some(0) || self.axis.is_none() || self.keepdims {
            // No reshaping needed
            gys
        } else {
            // 順伝播時の次元に変換する場合
            if let Some(axis) = self.axis {
                // axis が正数の場合はそのまま使用し、負の場合は実際の次元に置き換える。
                if axis.index() >= 0 {
                    actual_axis = axis;
                } else {
                    if let Some(tmp_ndim) = ndim {
                        let axis_index = tmp_ndim + axis.index();
                        actual_axis = Axis(axis_index);
                    }
                }
            }

            // 返還後の Shape を算出する。
            let mut shape = gys[0].borrow().get_data().shape().to_vec().clone();
            shape.insert(actual_axis.index(), 1);
            let reshape_gy = gys[0].borrow().get_data().to_shape(shape).unwrap();
            let reshape_gy = gys[0].set_data(reshape_gy);

            // gx = broadcast_to(gy, self.x_shape)
        }
    }
}

/// 加算関数
///
/// Arguments
/// * x0 (Variable<V>): 加算する変数
/// * x1 (Variable<V>): 加算する変数
///
/// Return
/// * Rc<RefCell<RawVariable>>: 加算結果
pub fn sum<V: MathOps>(input: Variable<V>, axis: Option<Axis>, keepdims: bool) -> Variable<V> {
    debug!("SumFunction::sum");
    let shape = input.borrow().get_data().shape().to_vec();

    let mut sum = FunctionExecutor::new(Rc::new(RefCell::new(SumFunction {
        x_shape: Some(shape),
        axis: axis,
        keepdims,
    })));
    // 加算の順伝播
    sum.forward(vec![input]).get(0).unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    /// 行列の要素の和のテスト
    #[test]
    fn test_sum_matrix() {
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 2],
            vec![1.0, 2.0, 3.0, 4.0],
        ));

        let sum_axis = sum(x.clone(), None, false);
        let sum_axis0 = sum(x.clone(), Some(Axis(0)), false);
        let sum_axis1 = sum(x.clone(), Some(Axis(1)), false);

        // dbg!(&sum_axis); // 10.0
        // dbg!(&sum_axis0); // [4.0, 6.0]
        // dbg!(&sum_axis1); // [3.0, 7.0]

        assert_eq!(vec![10.0], sum_axis.borrow().get_data().flatten().to_vec());
        assert_eq!(
            vec![4.0, 6.0],
            sum_axis0.borrow().get_data().flatten().to_vec()
        );
        assert_eq!(
            vec![3.0, 7.0],
            sum_axis1.borrow().get_data().flatten().to_vec()
        );
    }
}
