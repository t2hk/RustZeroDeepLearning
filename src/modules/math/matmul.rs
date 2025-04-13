// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, Array1, Array2, Ix0, IxDyn};
use ndarray::{Ix1, Ix2};
use std::cell::RefCell;
use std::rc::Rc;

/// matmul 関数
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

        debug!(
            "matmul(forward) x_data ndim: {:?}, dim: {:?}, shape: {:?}",
            x_data.ndim(),
            x_data.dim(),
            x_data.shape()
        );
        debug!(
            "matmul(forward) w_data ndim: {:?}, dim: {:?}, shape: {:?}",
            w_data.ndim(),
            w_data.dim(),
            w_data.shape()
        );

        match (x_data.ndim(), w_data.ndim()) {
            (0, 0) => {
                info!("matmul(forward) for shape (0, 0)");

                let x_tmp = x_data.into_dimensionality::<Ix0>().unwrap();
                let w_tmp = w_data.into_dimensionality::<Ix0>().unwrap();

                let result_value = x_tmp
                    .iter()
                    .zip(w_tmp.iter())
                    .fold(V::zero(), |acc, (a, b)| acc + (a.clone() * b.clone()));

                vec![Array::from_elem(IxDyn(&[]), result_value)]
            }
            (0, 1) => {
                info!("matmul(forward) for shape (0, 1)");

                let x_tmp = x_data.into_dimensionality::<Ix0>().unwrap();
                let w_tmp = w_data.into_dimensionality::<Ix1>().unwrap();

                let result_value = w_tmp
                    .iter()
                    .zip(x_tmp.iter())
                    .fold(V::zero(), |acc, (a, b)| acc + (a.clone() * b.clone()));

                vec![Array::from_shape_vec(vec![1], vec![result_value]).unwrap()]
            }
            (1, 0) => {
                info!("matmul(forward) for shape (1, 0)");

                let x_tmp = x_data.into_dimensionality::<Ix1>().unwrap();
                let w_tmp = w_data.into_dimensionality::<Ix0>().unwrap();

                let result_value = x_tmp
                    .iter()
                    .zip(w_tmp.iter())
                    .fold(V::zero(), |acc, (a, b)| acc + (a.clone() * b.clone()));

                vec![Array::from_shape_vec(vec![1], vec![result_value]).unwrap()]
            }
            (1, 1) => {
                info!("matmul(forward) for shape (1, 1)");

                let x_tmp = x_data.into_dimensionality::<Ix1>().unwrap();
                let w_tmp = w_data.into_dimensionality::<Ix1>().unwrap();

                let result_value = x_tmp
                    .iter()
                    .zip(w_tmp.iter())
                    .fold(V::zero(), |acc, (a, b)| acc + (a.clone() * b.clone()));

                vec![Array::from_elem(IxDyn(&[]), result_value)]
            }
            (1, 2) => {
                info!("matmul(forward) for shape (1, 2)");

                if x_data.shape()[0] != w_data.shape()[0] {
                    panic!(
                        "shapes {:?} and {:?} not aligned. {} != {}",
                        x_data.shape(),
                        w_data.shape(),
                        x_data.shape()[0],
                        w_data.shape()[0]
                    );
                }

                let x_tmp: Array1<V> = x_data.into_dimensionality::<Ix1>().unwrap();
                let w_tmp: Array2<V> = w_data.into_dimensionality::<Ix2>().unwrap();

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

                dbg!(&result);

                vec![result.into_dimensionality::<IxDyn>().unwrap()]
            }
            (2, 1) => {
                info!("matmul(forward) for shape (2, 1)");

                if x_data.shape()[0] != w_data.shape()[0] {
                    panic!(
                        "shapes {:?} and {:?} not aligned. {} != {}",
                        x_data.shape(),
                        w_data.shape(),
                        x_data.shape()[0],
                        w_data.shape()[0]
                    );
                }
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
                info!("matmul(forward) for shape (2, 2)");
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
                error!("matmul(forward) for invalid shape");
                debug!("x ndim: {} w ndim: {}", x_data.ndim(), w_data.ndim());
                debug!(
                    "x : {:?} w : {:?}",
                    x_data.flatten().to_vec(),
                    w_data.flatten().to_vec()
                );
                panic!("error: invalid dimension. x: {:?}, w: {:?}", x_data, w_data);
            }
        }
    }

    /// 逆伝播
    fn backward(&self, inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("matmul(backward)");

        let x = inputs[0].clone();
        let w = inputs[1].clone();

        dbg!(&gys[0].borrow().get_data());
        dbg!(&w.borrow().get_data());

        let gx = matmul(gys[0].clone(), w.transpose());
        let gw = matmul(x.transpose(), gys[0].clone());

        vec![gx, gw]
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
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use rand::prelude::*;
    use rand::{distributions::Uniform, Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;

    /// 数値微分による近似チェック
    #[test]
    fn test_num_grad_check_0_0() {
        let rand_x0 = rand::random::<f64>();
        let rand_x1 = rand::random::<f64>();

        let x0 = Variable::new(RawVariable::new(rand_x0));
        let x1 = Variable::new(RawVariable::new(rand_x1));

        let mut matmul: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(MatmulFunction {})));

        utils::gradient_check(&mut matmul, vec![x0.clone(), x1.clone()]);
    }

    /// 数値微分による近似チェック
    #[test]
    fn test_num_grad_check_0_1() {
        let rand_x0 = rand::random::<f64>();
        let x0 = Variable::new(RawVariable::new(rand_x0));

        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x1_var = Array::random_using((1), Uniform::new(0., 10.), &mut rng);

        let x1 = Variable::new(RawVariable::from_shape_vec(
            vec![1],
            x1_var.flatten().to_vec(),
        ));

        let mut matmul: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(MatmulFunction {})));

        utils::gradient_check(&mut matmul, vec![x0.clone(), x1.clone()]);
    }

    /// 数値微分による近似チェック
    #[test]
    fn test_num_grad_check_1_0() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x0_var = Array::random_using((1), Uniform::new(0., 10.), &mut rng);

        let x0 = Variable::new(RawVariable::from_shape_vec(
            vec![1],
            x0_var.flatten().to_vec(),
        ));

        let rand_x1 = rand::random::<f64>();
        let x1 = Variable::new(RawVariable::new(rand_x1));

        let mut matmul: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(MatmulFunction {})));

        utils::gradient_check(&mut matmul, vec![x0.clone(), x1.clone()]);
    }

    /// 数値微分による近似チェック
    #[test]
    fn test_num_grad_check_1_1() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x0_var = Array::random_using((1), Uniform::new(0., 10.), &mut rng);

        let x0 = Variable::new(RawVariable::from_shape_vec(
            vec![1],
            x0_var.flatten().to_vec(),
        ));

        let x1_var = Array::random_using((1), Uniform::new(0., 10.), &mut rng);

        let x1 = Variable::new(RawVariable::from_shape_vec(
            vec![1],
            x1_var.flatten().to_vec(),
        ));

        let mut matmul: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(MatmulFunction {})));

        utils::gradient_check(&mut matmul, vec![x0.clone(), x1.clone()]);
    }

    #[test]
    fn test_forward1() {
        let x = Variable::new(RawVariable::from_vec(vec![1., 2., 3.]));
        let w = Variable::new(RawVariable::from_vec(vec![4., 5., 6.]));

        let y = matmul(x, w);
        assert_eq!(32., y.borrow().get_data().flatten().to_vec()[0]);
    }

    #[test]
    fn test_num_grad_check2() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x0_var = Array::random_using((1, 100), Uniform::new(0., 10.), &mut rng);

        let x0 = Variable::new(RawVariable::from_shape_vec(
            vec![1, 100],
            x0_var.flatten().to_vec(),
        ));

        let x1_var = Array::random_using((100, 1), Uniform::new(0., 10.), &mut rng);

        let x1 = Variable::new(RawVariable::from_shape_vec(
            vec![100, 1],
            x1_var.flatten().to_vec(),
        ));

        let mut matmul: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(MatmulFunction {})));

        utils::gradient_check(&mut matmul, vec![x0.clone(), x1.clone()]);
    }

    #[test]
    fn test_num_grad_check3() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x0_var = Array::random_using((2, 100), Uniform::new(0., 10.), &mut rng);

        let x0 = Variable::new(RawVariable::from_shape_vec(
            vec![2, 100],
            x0_var.flatten().to_vec(),
        ));

        let x1_var = Array::random_using((100, 2), Uniform::new(0., 10.), &mut rng);

        let x1 = Variable::new(RawVariable::from_shape_vec(
            vec![100, 2],
            x1_var.flatten().to_vec(),
        ));

        let mut matmul: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(MatmulFunction {})));

        utils::gradient_check(&mut matmul, vec![x0.clone(), x1.clone()]);
    }

    #[test]
    fn test_forward10() {
        let x = Variable::new(RawVariable::from_shape_vec(vec![1, 1], vec![10.]));
        let w = Variable::new(RawVariable::from_shape_vec(vec![1, 1], vec![20.]));

        let y = matmul(x, w);
        assert_eq!(200., y.borrow().get_data().flatten().to_vec()[0]);

        y.backward();
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

    #[test]
    fn test_backward1() {
        let x = Variable::new(RawVariable::from_shape_vec(vec![2, 3], (1..=6).collect()));
        let w = Variable::new(RawVariable::from_shape_vec(vec![3, 4], (1..=12).collect()));

        let y = matmul(x.clone(), w.clone());
        dbg!(&y);
        y.backward();

        assert_eq!(
            vec![2, 3],
            x.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .shape()
                .to_vec()
        );
        assert_eq!(
            vec![3, 4],
            w.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .shape()
                .to_vec()
        );
    }

    #[test]
    fn test_backward_num_grad_1() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let x_var = Array::random_using((2, 3), Uniform::new(0., 10.), &mut rng);
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 3],
            x_var.flatten().to_vec(),
        ));

        let w_var = Array::random_using((3, 4), Uniform::new(0., 10.), &mut rng);
        let w = Variable::new(RawVariable::from_shape_vec(
            vec![3, 4],
            w_var.flatten().to_vec(),
        ));

        let mut matmul: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(MatmulFunction {})));

        utils::gradient_check(&mut matmul, vec![x.clone(), w.clone()]);
    }

    #[test]
    fn test_backward_num_grad_2() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let x_var = Array::random_using((10, 1), Uniform::new(0., 10.), &mut rng);
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![10, 1],
            x_var.flatten().to_vec(),
        ));

        let w_var = Array::random_using((1, 5), Uniform::new(0., 10.), &mut rng);
        let w = Variable::new(RawVariable::from_shape_vec(
            vec![1, 5],
            w_var.flatten().to_vec(),
        ));

        let mut matmul: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(MatmulFunction {})));

        utils::gradient_check(&mut matmul, vec![x.clone(), w.clone()]);
    }
}
