// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// mean_squared_error 関数
#[derive(Debug, Clone)]
pub struct MeanSquaredErrorFunction {}
impl<V: MathOps> Function<V> for MeanSquaredErrorFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "MeanSquaredError".to_string()
    }

    // 順伝播
    fn forward(&self, inputs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("mean_squared_error(forward)");
        let x0 = inputs[0].clone();
        let x1 = inputs[1].clone();

        let diff = x0 - x1;
        let y = (diff.clone() * diff.clone()).sum() / V::from(diff.len()).unwrap();
        vec![Array::from_elem(IxDyn(&[]), y)]
    }

    /// 逆伝播
    fn backward(&self, inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("mean_squared_error(backward)");

        let x0 = inputs[0].clone();
        let x1 = inputs[1].clone();

        let diff = &x0 - &x1;
        let gy = broadcast_to(gys[0].clone(), diff.borrow().get_data().shape().to_vec());
        let gx0 = &(&gy * &diff) * (2.0 / diff.borrow().get_data().len() as f64);
        let gx1 = -gx0.clone();

        vec![gx0, gx1]
    }
}

/// 平均二乗誤差
///
/// Arguments:
/// * x0 (Variable<V>):
/// * x1 (Variable<V>):
///
/// Return
/// * Variable<V>
pub fn mean_squared_error<V: MathOps>(x0: Variable<V>, x1: Variable<V>) -> Variable<V> {
    let mut mean_squared_error =
        FunctionExecutor::new(Rc::new(RefCell::new(MeanSquaredErrorFunction {})));
    // 順伝播
    mean_squared_error
        .forward(vec![x0.clone(), x1.clone()])
        .get(0)
        .unwrap()
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{cell::RefCell, rc::Rc};

    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use rand::{distributions::Uniform, SeedableRng};
    use rand_isaac::Isaac64Rng;

    #[test]
    fn test_mse_forward1() {
        let x0 = Variable::new(RawVariable::from_shape_vec(vec![1, 3], vec![0.0, 1.0, 2.0]));
        let x1 = Variable::new(RawVariable::from_shape_vec(vec![1, 3], vec![0.0, 1.0, 2.0]));

        let result = mean_squared_error(x0.clone(), x1.clone());
        // 書籍と同じ結果であることを確認する。
        assert_eq!(0.0, result.borrow().get_data()[[]]);
    }

    /// 数値微分による近似テスト
    #[test]
    fn test_mse_backward1() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x0_var = Array::random_using((10, 1), Uniform::new(0., 1.), &mut rng);
        let x1_var = Array::random_using((10, 1), Uniform::new(0., 1.), &mut rng);

        let x0 = Variable::new(RawVariable::from_shape_vec(
            vec![10, 1],
            x0_var.flatten().to_vec(),
        ));
        let x1 = Variable::new(RawVariable::from_shape_vec(
            vec![10, 1],
            x1_var.flatten().to_vec(),
        ));

        let mut mean_squared_error =
            FunctionExecutor::new(Rc::new(RefCell::new(MeanSquaredErrorFunction {})));

        utils::gradient_check(&mut mean_squared_error, vec![x0.clone(), x1.clone()]);
    }

    #[test]
    fn test_mse_backward2() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x0_var = Array::random_using((100, 1), Uniform::new(0., 1.), &mut rng);
        let x1_var = Array::random_using((100, 1), Uniform::new(0., 1.), &mut rng);

        let x0 = Variable::new(RawVariable::from_shape_vec(
            vec![100, 1],
            x0_var.flatten().to_vec(),
        ));
        let x1 = Variable::new(RawVariable::from_shape_vec(
            vec![100, 1],
            x1_var.flatten().to_vec(),
        ));

        let mut mean_squared_error =
            FunctionExecutor::new(Rc::new(RefCell::new(MeanSquaredErrorFunction {})));

        utils::gradient_check(&mut mean_squared_error, vec![x0.clone(), x1.clone()]);
    }
}
