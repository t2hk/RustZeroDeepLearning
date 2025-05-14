// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, Axis, IxDyn};
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

/// logsumexp 関数
///
/// Arguments:
/// * x (Array<V, IxDyn>): 変数
/// * axis (Axis): 軸
/// Return
/// * Array<V, IxDyn>:
pub fn logsumexp<V: MathOps>(x: Array<V, IxDyn>, axis: Axis) -> Array<V, IxDyn> {
    // def logsumexp(x, axis=1):
    //   xp = cuda.get_array_module(x)
    //   m = x.max(axis=axis, keepdims=True)
    //   y = x - m
    //   xp.exp(y, out=y)
    //   s = y.sum(axis=axis, keepdims=True)
    //   xp.log(s, out=s)
    //   m += s
    //   return m

    let m = x
        .map_axis(axis, |view| {
            // view.iter().max().unwrap().clone()
            let tmp_x: Vec<f64> = view.iter().map(|x| V::to_f64(x).unwrap()).collect();
            let max = tmp_x.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
            V::from(max.to_owned()).unwrap()
        })
        .insert_axis(axis);
    let y = x - m.clone();
    let exp_y = y.mapv_into(|y| V::from(V::to_f64(&y).unwrap().exp()).unwrap());
    let s = exp_y.sum_axis(axis).insert_axis(axis);
    let log_s = s.mapv(|s| V::from(V::to_f64(&s).unwrap().ln()).unwrap());

    let m_s = m + log_s;

    m_s
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
    fn test_logsumexp_01() {
        // python 結果: [1000.69314718]
        let x = Variable::new(RawData::from_vec(vec![1000.0, 1000.0]));
        let y = logsumexp(x.get_data(), Axis(0));
        dbg!(&y);
        assert_eq!(vec![1], y.shape().to_vec());
        let expect = vec![1000.69314718];
        let result = y.flatten().to_vec();
        assert_close(expect[0], result[0], 1e-8f64);
    }

    #[test]
    fn test_logsumexp_02() {
        // python 結果: y: [[2000.69314718 2000.69314718]]
        let x = Variable::new(RawData::from_shape_vec(
            vec![2, 2],
            vec![2000.0, 2000.0, 2000.0, 2000.0],
        ));
        let y = logsumexp(x.get_data(), Axis(0));
        dbg!(&y);
        assert_eq!(vec![1, 2], y.shape().to_vec());
        let expect = vec![2000.69314718, 2000.69314718];
        let result = y.flatten().to_vec();
        assert_close(expect[0], result[0], 1e-8f64);
        assert_close(expect[1], result[1], 1e-8f64);
    }

    #[test]
    fn test_logsumexp_03() {
        // python 結果: y: [[2000.69314718] [2000.69314718]] (shape: [2, 1])
        let x = Variable::new(RawData::from_shape_vec(
            vec![2, 2],
            vec![2000.0, 2000.0, 2000.0, 2000.0],
        ));
        let y = logsumexp(x.get_data(), Axis(1));
        dbg!(&y);
        assert_eq!(vec![2, 1], y.shape().to_vec());
        let expect = vec![2000.69314718, 2000.69314718, 2000.69314718, 2000.69314718];
        let result = y.flatten().to_vec();
        assert_close(expect[0], result[0], 1e-8f64);
        assert_close(expect[1], result[1], 1e-8f64);
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
