// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// SumTo 関数
#[derive(Debug, Clone)]
pub struct SumToFunction {
    x_shape: Vec<usize>,
    shape: Vec<usize>,
}
impl<V: MathOps> Function<V> for SumToFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "SumTo".to_string()
    }

    // SumTo の順伝播
    fn forward(&self, x: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("sum_to(forward)");

        debug!(
            "sum_to(forward) {:?} -> {:?}",
            x[0].flatten().to_vec(),
            self.shape.to_vec()
        );

        let ndim = self.shape.len() as i32;
        let x_ndim = x[0].ndim() as i32;
        let lead = x_ndim - ndim;

        // Create vectors for the axes to sum over
        let mut axes_to_sum = Vec::new();

        // Add leading axes
        for i in 0..lead {
            axes_to_sum.push(i);
        }

        // Add axes where target shape is 1
        for (i, &sx) in self.shape.iter().enumerate() {
            if sx == 1 {
                axes_to_sum.push(i as i32 + lead);
            }
        }

        // Sort axes in descending order to avoid changing indices during reduction
        axes_to_sum.sort_unstable_by(|a, b| b.cmp(a));

        // Start with a clone of the input array
        let mut y = x[0].to_owned();

        // Sum over specified axes, keeping dimensions
        for &axis in &axes_to_sum {
            y = y
                .sum_axis(ndarray::Axis(axis as usize))
                .insert_axis(ndarray::Axis(axis as usize));
        }

        // If there were leading dimensions, squeeze them out
        if lead > 0 {
            let squeeze_y = utils::squeeze(&y);
            return vec![squeeze_y.into_dyn()];
        } else {
            return vec![y.into_dyn()];
        }
    }

    /// SumTo の逆伝播
    fn backward(&self, _inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("sum_to(backward)");
        let result = broadcast_to(gys[0].clone(), self.x_shape.clone());

        debug!(
            "sum_to(backward) {:?} {:?} -> {:?} {:?}",
            gys[0].borrow().get_data().shape().to_vec(),
            gys[0].borrow().get_data().flatten().to_vec(),
            result.borrow().get_data().shape().to_vec(),
            result.borrow().get_data().flatten().to_vec()
        );
        vec![result]
    }
}

/// SumTo 関数
///
/// Arguments
/// * x (Variable<V>): 対象の変数
///
/// Return
/// * Variable<V>: 結果
pub fn sum_to<V: MathOps>(x: Variable<V>, shape: Vec<usize>) -> Variable<V> {
    let x_shape = x.borrow().get_data().shape().to_vec();
    let mut sum_to = FunctionExecutor::new(Rc::new(RefCell::new(SumToFunction {
        x_shape: x_shape,
        shape: shape,
    })));
    // 順伝播
    sum_to.forward(vec![x.clone()]).get(0).unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_sum_to_forward1() {
        let x = Variable::new(RawVariable::from_shape_vec(vec![1, 10], (1..11).collect()));
        let y = sum_to(x.clone(), vec![1]);

        let expected = sum(x, None, false);
        assert_eq!(1, y.borrow().get_data().shape().to_vec()[0]);
        assert_eq!(
            expected.borrow().get_data().flatten().to_vec(),
            y.borrow().get_data().flatten().to_vec()
        );
    }

    #[test]
    fn test_sum_to_forward2() {
        let x = Variable::new(RawVariable::from_shape_vec(vec![2, 3], (1..7).collect()));
        let y = sum_to(x.clone(), vec![1, 3]);
        dbg!(&y);
        let expected = sum(x.clone(), Some(vec![0]), true);
        dbg!(&expected);

        assert_eq!(
            expected.borrow().get_data().shape(),
            y.borrow().get_data().shape()
        );
        assert_eq!(expected.borrow().get_data(), y.borrow().get_data());
    }

    #[test]
    fn test_sum_to_forward3() {
        let x = Variable::new(RawVariable::from_shape_vec(vec![10], (1..11).collect()));
        let y = sum_to(x.clone(), vec![10]);

        assert_eq!(x.borrow().get_data().shape(), y.borrow().get_data().shape());
        assert_eq!(x.borrow().get_data(), y.borrow().get_data());
    }

    #[test]
    fn test_sum_to_backward1() {
        let x = Variable::new(RawVariable::from_shape_vec(vec![10], (1..11).collect()));

        let y = sum_to(x.clone(), vec![1]);
        y.backward();

        assert_eq!(
            vec![10],
            x.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .shape()
                .to_vec()
        );
        let expect: Vec<i32> = std::iter::repeat(1).take(10).collect();
        assert_eq!(
            expect,
            x.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .flatten()
                .to_vec()
        );
    }

    #[test]
    fn test_sum_to_backward2() {
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![10, 10],
            (1..101).collect(),
        ));

        let y = sum_to(x.clone(), vec![1]);
        y.backward();

        assert_eq!(
            vec![10, 10],
            x.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .shape()
                .to_vec()
        );

        let expect: Vec<i32> = std::iter::repeat(1).take(100).collect();
        assert_eq!(
            expect,
            x.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .flatten()
                .to_vec()
        );
    }

    #[test]
    fn test_sum_to_backward3() {
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![10, 20, 20],
            (1..4001).collect(),
        ));

        let y = sum_to(x.clone(), vec![1]);
        y.backward();

        assert_eq!(
            vec![10, 20, 20],
            x.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .shape()
                .to_vec()
        );

        let expect: Vec<i32> = std::iter::repeat(1).take(4000).collect();
        assert_eq!(
            expect,
            x.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .flatten()
                .to_vec()
        );
    }
}
