// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, Axis, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// SumTo 関数
#[derive(Debug, Clone)]
pub struct SumToFunction {
    x_shape: Vec<usize>,
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
    // ToDo 未実装
    fn forward(&self, x: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("sum_to(forward)");

        // TODO utils.sum_to 相当の処理の実装に変更すること。
        let y = x[0].sum();

        debug!("sum_to(backward) {:?} -> {:?}", x[0].flatten().to_vec(), y);

        vec![Array::from_elem(IxDyn(&[]), y)]
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
pub fn sum_to<V: MathOps>(x: Variable<V>) -> Variable<V> {
    let x_shape = x.borrow().get_data().shape().to_vec();
    let mut sum_to =
        FunctionExecutor::new(Rc::new(RefCell::new(SumToFunction { x_shape: x_shape })));
    // 順伝播
    sum_to.forward(vec![x.clone()]).get(0).unwrap().clone()
}

/// ToDo 実装するか削除する必要あり
pub fn util_sum_to<V: MathOps>(x: Variable<V>, shape: Vec<usize>) {
    // ndim = len(shape)
    // lead = x.ndim - ndim
    // lead_axis = tuple(range(lead))

    // axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    // y = x.sum(lead_axis + axis, keepdims=True)
    // if lead > 0:
    //     y = y.squeeze(lead_axis)
    // return y

    let ndim = shape.len() as i32;
    // let lead = x.borrow().get_data().shape().len() - ndim;
    let x_len = x.borrow().get_data().shape().len() as i32;

    let lead = x_len - ndim;
    let lead_axis: Vec<i32> = (0..lead).map(|x| x).collect();

    println!(
        "ndim: {:?}, x_len:{:?}, lead: {:?}, lead_axis:{:?}",
        ndim, x_len, lead, lead_axis
    );

    dbg!(&lead_axis);

    let mut axis = vec![];
    for (index, sx) in shape.iter().enumerate() {
        if *sx == 1 {
            axis.push(index as i32 + lead);
        }
    }
    dbg!(&axis);
    // let y = x.borrow().get_data().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    /// シンプルな全要素の和
    #[test]
    fn test_simple_sum() {
        let x = Variable::new(RawVariable::from_shape_vec(vec![1, 6], (1..7).collect()));
        let y = sum(x.clone(), None, false);
        y.backward();

        assert_eq!(vec![21], y.borrow().get_data().flatten().to_vec());

        assert_eq!(
            vec![1, 6],
            x.borrow().get_grad().unwrap().borrow().get_data().shape()
        );
        assert_eq!(
            vec![1, 1, 1, 1, 1, 1],
            x.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .flatten()
                .to_vec()
        );

        // 逆伝播結果
        // dbg!(&x.borrow().get_grad().unwrap());
        assert_eq!(
            vec![1, 6],
            x.borrow().get_grad().unwrap().borrow().get_data().shape()
        );
        assert_eq!(
            vec![1, 1, 1, 1, 1, 1],
            x.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .flatten()
                .to_vec()
        );
    }

    /// シンプルな全要素の和
    #[test]
    fn test_simple_sum2() {
        let x = Variable::new(RawVariable::from_shape_vec(vec![2, 3], (1..7).collect()));
        let y = sum(x.clone(), Some(vec![0]), false);
        y.backward();

        // 順伝播結果
        assert_eq!(vec![5, 7, 9], y.borrow().get_data().flatten().to_vec());
        assert_eq!(vec![3], y.borrow().get_data().shape().to_vec());

        // 逆伝播結果
        // dbg!(&x.borrow().get_grad().unwrap());
        assert_eq!(
            vec![2, 3],
            x.borrow().get_grad().unwrap().borrow().get_data().shape()
        );
        assert_eq!(
            vec![1, 1, 1, 1, 1, 1],
            x.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .flatten()
                .to_vec()
        );
    }

    /// keepdims を指定した全要素の和
    #[test]
    fn test_sum_keepdims() {
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 3, 4],
            (0..24).collect(),
        ));
        let y = sum(x.clone(), None, true);

        let tmp = y.borrow().get_data();

        assert_eq!(vec![276], tmp.flatten().to_vec());
        assert_eq!(vec![1, 1, 1], tmp.shape().to_vec());
    }

    /// keepdims を指定しない Axis(0) の和
    #[test]
    fn test_sum_axis0() {
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 3, 4],
            (0..24).collect(),
        ));
        let y = sum(x.clone(), Some(vec![0]), false);

        let tmp = y.borrow().get_data();
        dbg!(&tmp);

        assert_eq!(
            vec![12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
            tmp.flatten().to_vec()
        );
        assert_eq!(vec![3, 4], tmp.shape().to_vec());
    }

    /// keepdims を指定した Axis(0) の和
    #[test]
    fn test_sum_keepdims_axis0() {
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 3, 4],
            (0..24).collect(),
        ));
        let y = sum(x.clone(), Some(vec![0]), true);

        let tmp = y.borrow().get_data();
        dbg!(&tmp);

        assert_eq!(
            vec![12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
            tmp.flatten().to_vec()
        );
        assert_eq!(vec![1, 3, 4], tmp.shape().to_vec());

        y.backward();

        // 逆伝播結果
        // dbg!(&x.borrow().get_grad().unwrap());
        assert_eq!(
            vec![2, 3, 4],
            x.borrow().get_grad().unwrap().borrow().get_data().shape()
        );

        let grad: Vec<i32> = std::iter::repeat(1).take(24).collect();
        assert_eq!(
            grad,
            x.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .flatten()
                .to_vec()
        );
    }

    /// keepdims を指定しない Axis(1) の和
    #[test]
    fn test_sum_axis1() {
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 3, 4],
            (0..24).collect(),
        ));
        let y = sum(x.clone(), Some(vec![1]), false);

        let tmp = y.borrow().get_data();
        dbg!(&tmp);

        assert_eq!(vec![12, 15, 18, 21, 48, 51, 54, 57], tmp.flatten().to_vec());
        assert_eq!(vec![2, 4], tmp.shape().to_vec());

        y.backward();

        // 逆伝播結果
        // dbg!(&x.borrow().get_grad().unwrap());
        assert_eq!(
            vec![2, 3, 4],
            x.borrow().get_grad().unwrap().borrow().get_data().shape()
        );

        let grad: Vec<i32> = std::iter::repeat(1).take(24).collect();
        assert_eq!(
            grad,
            x.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .flatten()
                .to_vec()
        );
    }

    /// keepdims を指定した Axis(1) の和
    #[test]
    fn test_sum_keepdims_axis1() {
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 3, 4],
            (0..24).collect(),
        ));
        let y = sum(x.clone(), Some(vec![1]), true);

        let tmp = y.borrow().get_data();
        dbg!(&tmp);

        assert_eq!(vec![12, 15, 18, 21, 48, 51, 54, 57], tmp.flatten().to_vec());
        assert_eq!(vec![2, 1, 4], tmp.shape().to_vec());

        y.backward();

        // 逆伝播結果
        // dbg!(&x.borrow().get_grad().unwrap());
        assert_eq!(
            vec![2, 3, 4],
            x.borrow().get_grad().unwrap().borrow().get_data().shape()
        );

        let grad: Vec<i32> = std::iter::repeat(1).take(24).collect();
        assert_eq!(
            grad,
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
