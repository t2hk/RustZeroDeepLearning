// ライブラリを一括でインポート
use crate::modules::core::*;
use crate::modules::math::*;
#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};

use std::cell::RefCell;
use std::rc::Rc;

/// linear 関数
#[derive(Debug, Clone)]
pub struct LinearFunction {}
impl<V: MathOps> Function<V> for LinearFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Linear".to_string()
    }

    // linear の順伝播
    fn forward(&self, inputs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("linear(forward)");

        let x = inputs[0].clone();
        let w = inputs[1].clone();

        debug!(
            "linear(forward) x: {:?}, w: {:?}, b: {}",
            x.flatten().to_vec(),
            w.flatten().to_vec(),
            if inputs.len() > 2 {
                format!("{:?}", inputs[2].flatten().to_vec())
            } else {
                "None".to_string()
            }
        );

        let mut y = function_libs::dot(x.clone(), w.clone());

        if inputs.len() > 2 {
            let b = inputs[2].clone();
            y = y + b;
        }

        vec![y]
    }

    /// 逆伝播
    fn backward(&self, inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("linear(backward)");

        let x = inputs[0].clone();
        let w = inputs[1].clone();

        let gx = matmul(gys[0].clone(), w.transpose().clone());
        let gw = matmul(x.transpose().clone(), gys[0].clone());

        if inputs.len() > 2 {
            let b = inputs[2].clone();
            let gb = sum_to(gys[0].clone(), b.borrow().get_data().shape().to_vec());
            return vec![gx, gw, gb];
        }

        vec![gx, gw]
    }
}

/// linear 関数
///
/// Arguments
/// * x (Variable<V>): 変数
/// * w (Variable<V>): 重み
/// * b (Option<Variable<V>>): バイアス
///
/// Return
/// * Variable<V>: 結果
pub fn linear<V: MathOps>(x: Variable<V>, w: Variable<V>, b: Option<Variable<V>>) -> Variable<V> {
    let mut linear = FunctionExecutor::new(Rc::new(RefCell::new(LinearFunction {})));

    let mut inputs = vec![x, w];

    if let Some(b_tmp) = b {
        inputs.push(b_tmp);
    }

    // 順伝播
    linear
        //.forward(vec![x.clone(), w.clone()])
        .forward(inputs)
        .get(0)
        .unwrap()
        .clone()
}

#[cfg(test)]
mod tests {
    use std::env;

    use super::*;
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    // use rand::prelude::*;
    use rand::{distributions::Uniform, SeedableRng};
    use rand_isaac::Isaac64Rng;

    /// 数値微分による近似チェック
    #[test]
    fn test_num_grad_check_0_0() {
        let rand_x0 = rand::random::<f64>();
        let rand_x1 = rand::random::<f64>();

        let x0 = Variable::new(RawVariable::new(rand_x0));
        let x1 = Variable::new(RawVariable::new(rand_x1));

        let mut linear: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(LinearFunction {})));

        utils::gradient_check(&mut linear, vec![x0.clone(), x1.clone()]);
    }

    /// 数値微分による近似チェック
    #[test]
    fn test_forward1() {
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 3],
            vec![1., 2., 3., 4., 5., 6.],
        ));
        let w = x.transpose().clone();
        let b = None;
        let y = linear(x.clone(), w.clone(), b);

        assert_eq!(vec![2, 2], y.borrow().get_data().shape().to_vec());
        assert_eq!(
            vec![14., 32., 32., 77.],
            y.borrow().get_data().flatten().to_vec()
        );
    }

    #[test]
    fn test_backward1() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let x_var = Array::random_using((3, 2), Uniform::new(0., 10.), &mut rng);
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![3, 2],
            x_var.flatten().to_vec(),
        ));

        let w_var = Array::random_using((2, 3), Uniform::new(0., 10.), &mut rng);
        let w = Variable::new(RawVariable::from_shape_vec(
            vec![2, 3],
            w_var.flatten().to_vec(),
        ));

        let b_var = Array::random_using(3, Uniform::new(0., 10.), &mut rng);
        let b = Variable::new(RawVariable::from_vec(b_var.flatten().to_vec()));

        let mut linear: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(LinearFunction {})));

        utils::gradient_check(&mut linear, vec![x.clone(), w.clone(), b.clone()]);
    }

    #[test]
    fn test_backward2() {
        // env::set_var("RUST_LOG", "info");
        // env_logger::init();

        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let x_var = Array::random_using((10, 100), Uniform::new(0., 10.), &mut rng);
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![10, 100],
            x_var.flatten().to_vec(),
        ));

        let w_var = Array::random_using((100, 30), Uniform::new(0., 10.), &mut rng);
        let w = Variable::new(RawVariable::from_shape_vec(
            vec![100, 30],
            w_var.flatten().to_vec(),
        ));

        // let y = linear(x.clone(), w.clone(), None);
        // y.backward();

        let mut linear: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(LinearFunction {})));

        utils::gradient_check(&mut linear, vec![x.clone(), w.clone()]);

        // x = np.random.randn(100, 200)
        // W = np.random.randn(200, 300)
        // b = None
        // f = lambda x: F.linear(x, W, b)
        // self.assertTrue(gradient_check(f, x))
    }
}
