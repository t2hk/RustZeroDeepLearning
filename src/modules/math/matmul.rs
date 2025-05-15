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
use std::rc::Weak;

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

        vec![function_libs::dot(x_data, w_data)]
    }

    /// 逆伝播
    fn backward(
        &self,
        inputs: Vec<Variable<V>>,
        _outputs: Vec<Weak<RefCell<RawData<V>>>>,
        gys: Vec<Variable<V>>,
    ) -> Vec<Variable<V>> {
        info!("matmul(backward)");

        let x = inputs[0].clone();
        let w = inputs[1].clone();

        dbg!(&gys[0].get_data());
        dbg!(&w.get_data());

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
    // use rand::prelude::*;
    use rand::{distributions::Uniform, SeedableRng};
    use rand_isaac::Isaac64Rng;

    /// 数値微分による近似チェック
    #[test]
    fn test_num_grad_check_0_0() {
        let rand_x0 = rand::random::<f64>();
        let rand_x1 = rand::random::<f64>();

        let x0 = Variable::new(RawData::new(rand_x0));
        let x1 = Variable::new(RawData::new(rand_x1));

        let mut matmul: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(MatmulFunction {})));

        utils::gradient_check(&mut matmul, vec![x0.clone(), x1.clone()]);
    }

    /// 数値微分による近似チェック
    #[test]
    fn test_num_grad_check_0_1() {
        let rand_x0 = rand::random::<f64>();
        let x0 = Variable::new(RawData::new(rand_x0));

        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x1_var = Array::random_using(1, Uniform::new(0., 10.), &mut rng);

        let x1 = Variable::new(RawData::from_shape_vec(vec![1], x1_var.flatten().to_vec()));

        let mut matmul: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(MatmulFunction {})));

        utils::gradient_check(&mut matmul, vec![x0.clone(), x1.clone()]);
    }

    /// 数値微分による近似チェック
    #[test]
    fn test_num_grad_check_1_0() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x0_var = Array::random_using(1, Uniform::new(0., 10.), &mut rng);

        let x0 = Variable::new(RawData::from_shape_vec(vec![1], x0_var.flatten().to_vec()));

        let rand_x1 = rand::random::<f64>();
        let x1 = Variable::new(RawData::new(rand_x1));

        let mut matmul: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(MatmulFunction {})));

        utils::gradient_check(&mut matmul, vec![x0.clone(), x1.clone()]);
    }

    /// 数値微分による近似チェック
    #[test]
    fn test_num_grad_check_1_1() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x0_var = Array::random_using(1, Uniform::new(0., 10.), &mut rng);

        let x0 = Variable::new(RawData::from_shape_vec(vec![1], x0_var.flatten().to_vec()));

        let x1_var = Array::random_using(1, Uniform::new(0., 10.), &mut rng);

        let x1 = Variable::new(RawData::from_shape_vec(vec![1], x1_var.flatten().to_vec()));

        let mut matmul: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(MatmulFunction {})));

        utils::gradient_check(&mut matmul, vec![x0.clone(), x1.clone()]);
    }

    #[test]
    fn test_forward1() {
        let x = Variable::new(RawData::from_vec(vec![1., 2., 3.]));
        let w = Variable::new(RawData::from_vec(vec![4., 5., 6.]));

        let y = matmul(x, w);
        assert_eq!(32., y.get_data().flatten().to_vec()[0]);
    }

    #[test]
    fn test_num_grad_check2() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x0_var = Array::random_using((1, 100), Uniform::new(0., 10.), &mut rng);

        let x0 = Variable::new(RawData::from_shape_vec(
            vec![1, 100],
            x0_var.flatten().to_vec(),
        ));

        let x1_var = Array::random_using((100, 1), Uniform::new(0., 10.), &mut rng);

        let x1 = Variable::new(RawData::from_shape_vec(
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

        let x0 = Variable::new(RawData::from_shape_vec(
            vec![2, 100],
            x0_var.flatten().to_vec(),
        ));

        let x1_var = Array::random_using((100, 2), Uniform::new(0., 10.), &mut rng);

        let x1 = Variable::new(RawData::from_shape_vec(
            vec![100, 2],
            x1_var.flatten().to_vec(),
        ));

        let mut matmul: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(MatmulFunction {})));

        utils::gradient_check(&mut matmul, vec![x0.clone(), x1.clone()]);
    }

    #[test]
    fn test_forward10() {
        let x = Variable::new(RawData::from_shape_vec(vec![1, 1], vec![10.]));
        let w = Variable::new(RawData::from_shape_vec(vec![1, 1], vec![20.]));

        let y = matmul(x, w);
        assert_eq!(200., y.get_data().flatten().to_vec()[0]);

        y.backward();
    }

    #[test]
    fn test_forward2() {
        let x = Variable::new(RawData::from_shape_vec(vec![2, 2], (1..=4).collect()));
        // let y = matmul(x.clone(), x.clone());

        let w = Variable::new(RawData::from_shape_vec(vec![2, 2], (5..=8).collect()));

        let y = matmul(x, w);
        assert_eq!(vec![2, 2], y.get_data().shape().to_vec());
        assert_eq!(vec![19, 22, 43, 50], y.get_data().flatten().to_vec());
    }

    /// シンプルな行列の積
    #[test]
    fn test_simple_matmul() {
        let x = Variable::new(RawData::from_shape_vec(vec![1, 6], (1..7).collect()));
        let y = sum(x.clone(), None, false);
        y.backward();

        assert_eq!(vec![21], y.get_data().flatten().to_vec());

        assert_eq!(vec![1, 6], x.get_grad().unwrap().get_data().shape());
        assert_eq!(
            vec![1, 1, 1, 1, 1, 1],
            x.get_grad().unwrap().get_data().flatten().to_vec()
        );

        // 逆伝播結果
        // dbg!(&x.get_grad().unwrap());
        assert_eq!(vec![1, 6], x.get_grad().unwrap().get_data().shape());
        assert_eq!(
            vec![1, 1, 1, 1, 1, 1],
            x.get_grad().unwrap().get_data().flatten().to_vec()
        );
    }

    #[test]
    fn test_backward1() {
        let x = Variable::new(RawData::from_shape_vec(vec![2, 3], (1..=6).collect()));
        let w = Variable::new(RawData::from_shape_vec(vec![3, 4], (1..=12).collect()));

        let y = matmul(x.clone(), w.clone());
        dbg!(&y);
        y.backward();

        assert_eq!(
            vec![2, 3],
            x.get_grad().unwrap().get_data().shape().to_vec()
        );
        assert_eq!(
            vec![3, 4],
            w.get_grad().unwrap().get_data().shape().to_vec()
        );
    }

    #[test]
    fn test_backward_num_grad_1() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let x_var = Array::random_using((2, 3), Uniform::new(0., 10.), &mut rng);
        let x = Variable::new(RawData::from_shape_vec(
            vec![2, 3],
            x_var.flatten().to_vec(),
        ));

        let w_var = Array::random_using((3, 4), Uniform::new(0., 10.), &mut rng);
        let w = Variable::new(RawData::from_shape_vec(
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
        let x = Variable::new(RawData::from_shape_vec(
            vec![10, 1],
            x_var.flatten().to_vec(),
        ));

        let w_var = Array::random_using((1, 5), Uniform::new(0., 10.), &mut rng);
        let w = Variable::new(RawData::from_shape_vec(
            vec![1, 5],
            w_var.flatten().to_vec(),
        ));

        let mut matmul: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(MatmulFunction {})));

        utils::gradient_check(&mut matmul, vec![x.clone(), w.clone()]);
    }
}
