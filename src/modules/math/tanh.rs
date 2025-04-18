// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// Tanh 関数
#[derive(Debug, Clone)]
pub struct TanhFunction;
impl<V: MathOps> Function<V> for TanhFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Tanh".to_string()
    }

    // Tanh の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("tanh(forward)");
        debug!("tanh(forward): tanh({:?})", xs[0].flatten().to_vec());
        let result = vec![xs[0].mapv(|x| {
            let tanh_x = V::to_f64(&x).unwrap().tanh();
            V::from(tanh_x).unwrap()
        })];

        result
    }

    /// 逆伝播
    /// d tanh(x) /dx = 1 - y^2
    fn backward(&self, inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("tanh(backward)");
        debug!(
            "tanh(backward): (1 - {:?} ^2) * {:?}",
            &inputs[0].borrow().get_data().flatten().to_vec(),
            &gys[0].borrow().get_data().flatten().to_vec()
        );

        let tanh_x_pow_2 = &tanh(inputs[0].clone()) ^ 2;

        let gxs =
            vec![&(&Variable::new(RawData::new(V::from(1.0).unwrap())) - &tanh_x_pow_2) * &gys[0]];

        gxs
    }
}

/// Tanh 関数
///
/// Arguments
/// * input (Rc<RefCell<RawData>>): 入力値
///
/// Return
/// * Rc<RefCell<RawData>>: 結果
pub fn tanh<V: MathOps>(input: Variable<V>) -> Variable<V> {
    let mut tanh = FunctionExecutor::new(Rc::new(RefCell::new(TanhFunction)));
    // Tanh の順伝播
    tanh.forward(vec![input.clone()]).get(0).unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PIf32;

    use ndarray_rand::RandomExt;
    use rand::prelude::*;
    use rand::{distributions::Uniform, SeedableRng};
    use rand_isaac::Isaac64Rng;

    #[test]
    fn test_num_grad() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x0_var = Array::random_using((10, 10), Uniform::new(0., 10.), &mut rng);
        let x0 = Variable::new(RawData::from_shape_vec(
            vec![10, 10],
            x0_var.flatten().to_vec(),
        ));

        let mut tanh: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(TanhFunction {})));

        utils::gradient_check(&mut tanh, vec![x0.clone()]);
    }

    /// Tanh 関数のテスト。
    #[test]
    fn test_tanh_1() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        let x = Variable::new(RawData::new(PIf32 / 4.0f32));

        let expected_data = Array::from_elem(IxDyn(&[]), 0.6557942026326724f32);
        let expected_grad = Array::from_elem(IxDyn(&[]), 0.56993395f32);

        // 順伝播、逆伝播を実行する。
        let result = tanh(x.clone());

        // Tanh 結果
        assert_eq!(expected_data, result.borrow().get_data());

        result.backward();

        // 逆伝播結果
        assert_eq!(
            expected_grad,
            x.borrow().get_grad().unwrap().borrow().get_data()
        );
    }
}
