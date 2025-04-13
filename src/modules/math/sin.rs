// ライブラリを一括でインポート
use crate::modules::math::*;

#[allow(unused_imports)]
use core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// Sin 関数
#[derive(Debug, Clone)]
pub struct SinFunction;
impl<V: MathOps> Function<V> for SinFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Sin".to_string()
    }

    // Sin の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("sin(forward)");
        debug!("sin(forward): sin({:?})", xs[0].flatten().to_vec());
        let result = vec![xs[0].mapv(|x| {
            let sin_x = V::to_f64(&x).unwrap().sin();

            V::from(sin_x).unwrap()
        })];

        result
    }

    /// 逆伝播
    /// dy/dx=cos(x) である。
    fn backward(&self, inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("sin(backward)");
        debug!(
            "sin(backward): cos({:?}) * {:?}",
            &inputs[0].borrow().get_data().flatten().to_vec(),
            &gys[0].borrow().get_data().flatten().to_vec()
        );
        let gxs = vec![&cos(inputs[0].clone()) * &gys[0]];
        gxs
    }
}

/// Sin 関数
///
/// Arguments
/// * input (Rc<RefCell<RawVariable>>): 入力値
///
/// Return
/// * Rc<RefCell<RawVariable>>: 結果
pub fn sin<V: MathOps>(input: Variable<V>) -> Variable<V> {
    let mut sin = FunctionExecutor::new(Rc::new(RefCell::new(SinFunction)));
    // Sin の順伝播
    sin.forward(vec![input.clone()]).get(0).unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::RandomExt;
    use rand::{distributions::Uniform, SeedableRng};
    use rand_isaac::Isaac64Rng;
    use std::f32::consts::PI as PIf32;

    #[test]
    fn test_num_grad() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x0_var = Array::random_using(1, Uniform::new(0., 10.), &mut rng);

        let x0 = Variable::new(RawVariable::from_shape_vec(
            vec![1],
            x0_var.flatten().to_vec(),
        ));

        let mut sin: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(SinFunction {})));

        utils::gradient_check(&mut sin, vec![x0.clone()]);
    }

    /// Sin 関数のテスト。
    #[test]
    fn test_sin_1() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        let x = Variable::new(RawVariable::new(PIf32 / 4.0f32));

        let expected_output_data = Array::from_elem(IxDyn(&[]), 0.7071067811865475f32);

        // 順伝播、逆伝播を実行する。
        let result = sin(x.clone());
        result.backward();

        // sin 結果
        assert_eq!(expected_output_data, result.borrow().get_data());
        // 逆伝播結果
        assert_eq!(
            expected_output_data,
            x.borrow().get_grad().unwrap().borrow().get_data()
        );
    }
}
