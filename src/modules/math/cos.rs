// ライブラリを一括でインポート
use crate::modules::math::*;

#[allow(unused_imports)]
use core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

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
    fn backward(&self, inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("cos(backward)");
        debug!(
            "cos(backward): -sin({:?}) * {:?}",
            &inputs[0].borrow().get_data().flatten().to_vec(),
            &gys[0].borrow().get_data().flatten().to_vec()
        );

        let minus_sin_x = &sin(inputs[0].clone()) * -1;

        let gxs = vec![&minus_sin_x * &gys[0]];
        gxs
    }
}

/// Cos 関数
///
/// Arguments
/// * input (Rc<RefCell<RawVariable>>): 入力値
///
/// Return
/// * Rc<RefCell<RawVariable>>: 結果
pub fn cos<V: MathOps>(input: Variable<V>) -> Variable<V> {
    let mut cos = FunctionExecutor::new(Rc::new(RefCell::new(CosFunction)));
    // Cos の順伝播
    cos.forward(vec![input.clone()]).get(0).unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PIf32;

    /// Cos 関数のテスト。
    #[test]
    fn test_cos_1() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        let x = Variable::new(RawVariable::new(PIf32 / 4.0f32));

        let expected_output_data = Array::from_elem(IxDyn(&[]), 0.7071067811865475f32);

        // 順伝播、逆伝播を実行する。
        let result = cos(x.clone());

        // Cos 結果
        assert_eq!(expected_output_data, result.borrow().get_data());

        result.backward();

        // 逆伝播結果
        assert_eq!(
            expected_output_data * -1.0,
            x.borrow().get_grad().unwrap().borrow().get_data()
        );
    }
}
