// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{linalg::Dot, Array1, Array2, ArrayD, Ix1, Ix2};
use ndarray::{Array, IxDyn, LinalgScalar};
use std::cell::RefCell;
use std::rc::Rc;

/// Sum 関数
#[derive(Debug, Clone)]
pub struct MatmulFunction {}
impl<V: MathOps> Function<V> for MatmulFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Matmul".to_string()
    }

    // matmul の順伝播
    fn forward(&self, inputs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("matmul(forward)");

        let x_data = inputs[0].clone();
        let w_data = inputs[1].clone();

        debug!(
            "matmul(forward) x ndim: {:?}, w ndim: {:?}",
            x_data.ndim(),
            w_data.ndim()
        );

        // ベクトルの場合
        if x_data.dim()[0] == 1 && x_data.dim() == w_data.dim() {
            let result_value = x_data
                .iter()
                .zip(w_data.iter())
                .fold(V::zero(), |acc, (a, b)| acc + (a.clone() * b.clone()));

            return vec![Array::from_elem(IxDyn(&[]), result_value)];
        }

        match (x_data.ndim(), w_data.ndim()) {
            (1, 1) => {
                let x_tmp = x_data.into_dimensionality::<Ix1>().unwrap();
                let w_tmp = w_data.into_dimensionality::<Ix1>().unwrap();

                let result_value = x_tmp
                    .iter()
                    .zip(w_tmp.iter())
                    .fold(V::zero(), |acc, (a, b)| acc + (a.clone() * b.clone()));

                vec![Array::from_elem(IxDyn(&[]), result_value)]
            }
            (1, 2) => {
                let x_tmp = x_data.into_dimensionality::<Ix1>().unwrap();
                let w_tmp = w_data.into_dimensionality::<Ix2>().unwrap();

                let x_len = x_tmp.len();
                let w_cols = w_tmp.shape()[1];

                let mut result = Array::zeros((w_cols,));

                for j in 0..w_cols {
                    let mut sum = V::zero();
                    for i in 0..x_len {
                        sum = sum + x_tmp[i].clone() * w_tmp[[i, j]].clone();
                    }
                    result[j] = sum;
                }

                vec![result.into_dimensionality::<IxDyn>().unwrap()]
            }
            (2, 1) => {
                let x_tmp = x_data.into_dimensionality::<Ix2>().unwrap();
                let w_tmp = w_data.into_dimensionality::<Ix1>().unwrap();

                let x_rows = x_tmp.shape()[0];
                let x_cols = x_tmp.shape()[1];

                // 結果は x_rows 長のベクトルになる
                let mut result = Array::zeros((x_rows,));

                for i in 0..x_rows {
                    let mut sum = V::zero();
                    for j in 0..x_cols {
                        sum = sum + x_tmp[[i, j]].clone() * w_tmp[j].clone();
                    }
                    result[i] = sum;
                }

                vec![result.into_dimensionality::<IxDyn>().unwrap()]
            }
            (2, 2) => {
                let x_tmp = x_data.into_dimensionality::<Ix2>().unwrap();
                let w_tmp = w_data.into_dimensionality::<Ix2>().unwrap();

                let x_rows = x_tmp.shape()[0];
                let x_cols = x_tmp.shape()[1];
                let w_cols = w_tmp.shape()[1];

                // x_cols と w_rows（= x_tmp.shape()[1]とw_tmp.shape()[0]）は同じサイズであることを確認
                assert_eq!(x_cols, w_tmp.shape()[0], "行列の次元が不一致です");

                // 結果は x_rows x w_cols の行列になる
                let mut result = Array::zeros((x_rows, w_cols));

                for i in 0..x_rows {
                    for j in 0..w_cols {
                        let mut sum = V::zero();
                        for k in 0..x_cols {
                            sum = sum + x_tmp[[i, k]].clone() * w_tmp[[k, j]].clone();
                        }
                        result[[i, j]] = sum;
                    }
                }

                vec![result.into_dimensionality::<IxDyn>().unwrap()]
            }
            _ => {
                panic!("error: invalid dimension. x: {:?}, w: {:?}", x_data, w_data);
            }
        }
    }

    /// 逆伝播
    fn backward(&self, _inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("matmul(backward)");

        //let reshape_gy = Variable::new(RawVariable::new(gy));

        gys
    }
}

/// matmul 関数
///
/// Arguments
/// * x (Variable<V>): 変数
/// * w (Variable<V>): 変数
///
/// Return
/// * Variable<V>: 結果
pub fn matmul<V: MathOps>(x: Variable<V>, w: Variable<V>) -> Variable<V> {
    let x_shape = x.borrow().get_data().shape().to_vec();
    let mut matmul = FunctionExecutor::new(Rc::new(RefCell::new(MatmulFunction {})));
    // 順伝播
    matmul
        .forward(vec![x.clone(), w.clone()])
        .get(0)
        .unwrap()
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Ix1, Ix2, Ix3};
    use rand::prelude::*;

    #[test]
    fn test_forward1() {
        let x = Variable::new(RawVariable::from_shape_vec(vec![1, 3], vec![1, 2, 3]));
        let w = Variable::new(RawVariable::from_shape_vec(vec![1, 3], vec![4, 5, 6]));

        let y = matmul(x, w);
        assert_eq!(32, y.borrow().get_data().flatten().to_vec()[0]);
    }

    #[test]
    fn test_forward2() {
        let x = Variable::new(RawVariable::from_shape_vec(vec![2, 2], (1..=4).collect()));
        // let y = matmul(x.clone(), x.clone());

        let w = Variable::new(RawVariable::from_shape_vec(vec![2, 2], (5..=8).collect()));

        let y = matmul(x, w);
        assert_eq!(vec![2, 2], y.borrow().get_data().shape().to_vec());
        assert_eq!(
            vec![19, 22, 43, 50],
            y.borrow().get_data().flatten().to_vec()
        );
    }

    /// シンプルな行列の積
    #[test]
    fn test_simple_matmul() {
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
}
