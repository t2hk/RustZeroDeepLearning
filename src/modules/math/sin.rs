// ライブラリを一括でインポート
use crate::modules::math::*;

use core::fmt::Debug;
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
        let result = vec![xs[0].mapv(|x| {
            let sin_x = V::to_f64(&x).unwrap().sin();
            println!("sin x: {}", sin_x);
            V::from(sin_x).unwrap()
        })];

        result
    }

    /// 逆伝播
    /// dy/dx=cos(x) である。
    fn backward(&self, inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        let x = inputs[0].borrow().get_data();
        let gys_val = gys[0].clone();

        let cos_x = vec![x.mapv(|x| {
            let cos_x = V::to_f64(&x).unwrap().cos();
            println!("cos x: {}", cos_x);
            V::from(cos_x).unwrap()
        })];

        let gxs = cos_x.iter().map(|x| x * &gys_val).collect();
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
    use std::f32::consts::PI as PIf32;

    use super::*;

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
