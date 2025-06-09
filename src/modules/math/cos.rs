// ライブラリを一括でインポート
use crate::modules::math::*;

#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// Cos 関数
#[derive(Debug, Clone)]
pub struct CosFunction;
impl<V: MathOps> Function<V> for CosFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Cos".to_string()
    }

    // Cos の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("cos(forward)");
        debug!("cos(forwad) cos({:?})", xs[0].flatten().to_vec());

        let result = vec![xs[0].mapv(|x| {
            let cos_x = V::to_f64(&x).unwrap().cos();
            V::from(cos_x).unwrap()
        })];

        result
    }

    /// 逆伝播
    /// dy/dx=-sin(x) である。
    fn backward(
        &self,
        inputs: Vec<Variable<V>>,
        _outputs: Vec<Weak<RefCell<RawData<V>>>>,
        gys: Vec<Variable<V>>,
    ) -> Vec<Variable<V>> {
        info!("cos(backward)");
        debug!(
            "cos(backward): -sin({:?}) * {:?}",
            &inputs[0].get_data().flatten().to_vec(),
            &gys[0].get_data().flatten().to_vec()
        );

        let minus_sin_x = &sin(inputs[0].clone()) * -1;

        let gxs = vec![&minus_sin_x * &gys[0]];
        gxs
    }
}

/// Cos 関数
///
/// Arguments
/// * input (Rc<RefCell<RawData>>): 入力値
///
/// Return
/// * Rc<RefCell<RawData>>: 結果
pub fn cos<V: MathOps>(input: Variable<V>) -> Variable<V> {
    let mut cos = FunctionExecutor::new(Rc::new(RefCell::new(CosFunction)));
    // Cos の順伝播
    cos.forward(vec![input.clone()]).first().unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PIf32;

    use ndarray_rand::RandomExt;
    use rand::{distributions::Uniform, SeedableRng};
    use rand_isaac::Isaac64Rng;

    /// 数値微分による近似チェック
    #[test]
    fn test_num_grad_check() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x0_var = Array::random_using((1, 100), Uniform::new(0., 10.), &mut rng);

        let x0 = Variable::new(RawData::from_shape_vec(
            vec![1, 100],
            x0_var.flatten().to_vec(),
        ));

        let mut cos = FunctionExecutor::new(Rc::new(RefCell::new(CosFunction {})));

        utils::gradient_check(&mut cos, vec![x0.clone()]);
    }

    /// Cos 関数のテスト。
    #[test]
    fn test_cos_1() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        let x = Variable::new(RawData::new(PIf32 / 4.0f32));

        let expected_output_data = Array::from_elem(IxDyn(&[]), 0.7071067811865475f32);

        // 順伝播、逆伝播を実行する。
        let result = cos(x.clone());

        // Cos 結果
        assert_eq!(expected_output_data, result.get_data());

        result.backward();

        // 逆伝播結果
        assert_eq!(
            expected_output_data * -1.0,
            x.get_grad().unwrap().get_data()
        );
    }
}
