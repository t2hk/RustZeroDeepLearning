// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// Log 関数
#[derive(Debug, Clone)]
pub struct LogFunction {}

impl<V: MathOps> Function<V> for LogFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Log".to_string()
    }

    // 順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("log(forward)");
        debug!("log(forward) {:?}", &xs[0].flatten().to_vec());

        let result = xs[0].mapv(|x| V::from(x.to_f64().unwrap().ln()).unwrap());

        vec![result]
    }

    /// 逆伝播
    fn backward(&self, inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("log(backward)");

        let result = &gys[0].clone() / &inputs[0].clone();

        debug!(
            "log(backward) gy/x: {:?}",
            result.get_data().flatten().to_vec()
        );
        vec![result]
    }
}

/// log 関数
///
/// Arguments
/// * x (Variable<V>): 変数
///
/// Return
/// * Rc<RefCell<RawData>>: log 結果
pub fn log<V: MathOps>(x: Variable<V>) -> Variable<V> {
    debug!("LogFunction::log");

    let mut log = FunctionExecutor::new(Rc::new(RefCell::new(LogFunction {})));

    // 順伝播
    log.forward(vec![x.clone()]).get(0).unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    // 浮動小数点値を比較するためのヘルパー関数
    fn assert_close(a: f64, b: f64, epsilon: f64) {
        assert!(
            (a - b).abs() < epsilon,
            "Expected {:?} to be close to {:?}",
            a,
            b
        );
    }

    #[test]
    fn test_log_01() {
        let log_fn = LogFunction {};

        // 単一値のテスト
        let input1 = Array::<f64, _>::from_elem(IxDyn(&[1]), 1.0);
        let result1 = log_fn.forward(vec![input1]);
        // ln(1) = 0
        assert_close(result1[0].as_slice().unwrap()[0], 0.0f64, 1e-12f64);

        // 複数値のテスト
        let input2 = Array::from_vec(vec![2.0f64, 7.389f64, 20.085f64]).into_dyn();
        let result2 = log_fn.forward(vec![input2]);
        let expected2 = vec![2.0f64.ln(), 7.389f64.ln(), 20.085f64.ln()];

        for (i, &exp) in expected2.iter().enumerate() {
            assert_close(result2[0].as_slice().unwrap()[i], exp, 1e-12f64);
        }
    }

    #[test]
    fn test_log_02() {
        let input = Variable::new(RawData::from_vec(vec![2.0, 3.0, 4.0]));

        let mut log = FunctionExecutor::new(Rc::new(RefCell::new(LogFunction {})));

        utils::gradient_check(&mut log, vec![input.clone()]);
    }
}
