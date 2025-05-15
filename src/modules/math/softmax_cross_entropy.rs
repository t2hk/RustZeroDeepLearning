// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, Axis, IxDyn};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// Softmax Cross Entropy 関数
#[derive(Debug, Clone)]
pub struct SoftmaxCrossEntropyFunction {
    t: Variable<usize>,
}

impl<V: MathOps> Function<V> for SoftmaxCrossEntropyFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "SoftmaxCrossEntropy".to_string()
    }

    // 順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        // class SoftmaxCrossEntropy(Function):
        //     def forward(self, x, t):
        //         N = x.shape[0]
        //         log_z = utils.logsumexp(x, axis=1)
        //         log_p = x - log_z
        //         log_p = log_p[np.arange(N), t.ravel()]
        //         y = -log_p.sum() / np.float32(N)
        //         return y

        info!("softmax_cross_entropy(forward)");

        let n = xs[0].shape().to_vec()[0];
        let log_z = logarithm::logsumexp(xs[0].clone(), Axis(1));
        let log_p = xs[0].clone() - log_z;

        let mut tlog_p_vec = vec![];

        for i in 0..n {
            let t_val = self.t.get_data().flatten().to_vec()[i];
            let log_p_val = log_p[[i, t_val]].clone();
            tlog_p_vec.push(log_p_val);
        }
        let tlog_p = Array::from_shape_vec((1, n), tlog_p_vec).unwrap();

        let y = (V::from(-1).unwrap() * tlog_p.sum()) / V::from(n).unwrap();

        vec![RawData::new(y).get_data()]
    }

    /// 逆伝播
    fn backward(
        &self,
        inputs: Vec<Variable<V>>,
        _outputs: Vec<Weak<RefCell<RawData<V>>>>,
        gys: Vec<Variable<V>>,
    ) -> Vec<Variable<V>> {
        // class SoftmaxCrossEntropy(Function):
        //     def backward(self, gy):
        //         x, t = self.inputs
        //         N, CLS_NUM = x.shape
        //         gy *= 1/N
        //         y = softmax(x)
        //         # convert to one-hot
        //         xp = cuda.get_array_module(t.data)
        //         t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        //         y = (y - t_onehot) * gy
        //         return y

        info!("log(softmax_cross_entropy)");

        let n = &inputs[0].get_data().shape().to_vec()[0];
        let cls_num = inputs[0].get_data().shape().to_vec()[1] as usize;

        println!("n: {}, cls_num: {}", n, cls_num);

        let t_data = self.t.get_data();

        let one_div_n = RawData::new(V::from(1).unwrap() / V::from(*n as f64).unwrap()).get_data();
        println!(" 1 / n = {}", one_div_n);

        let gy = gys[0].get_data() * one_div_n;
        let y = softmax(inputs[0].clone(), Axis(1));

        let mut t_onehot = Array::zeros((*n, cls_num));
        for (i, &ref label) in t_data.iter().enumerate() {
            t_onehot[[i, *label]] = V::from(1).unwrap();
        }

        let t_onehot_var = Variable::new(RawData::from_shape_vec(
            t_onehot.shape().to_vec(),
            t_onehot.flatten().to_vec(),
        ));

        let result = &(&y - &t_onehot_var) * &gy;

        vec![result]
    }
}

/// softmax_cross_entropy 関数
///
/// Arguments
/// * x (Variable<V>): 変数
/// * t (Variable<usize>): 教師データ
///
/// Return
/// * Rc<RefCell<RawData>>: log 結果
pub fn softmax_cross_entropy<V: MathOps>(x: Variable<V>, t: Variable<usize>) -> Variable<V> {
    debug!("SoftmaxCrossEntropyFunction::softmax_cross_entropy");

    let mut softmax_cross_entropy =
        FunctionExecutor::new(Rc::new(RefCell::new(SoftmaxCrossEntropyFunction { t: t })));

    // 順伝播
    softmax_cross_entropy
        .forward(vec![x.clone()])
        .get(0)
        .unwrap()
        .clone()
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
    fn test_softmax_cross_entropy_01() {
        // x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        // t = np.array([3, 0]).astype(int)
        // y = F.softmax_cross_entropy(x, t)
        // y2 = CF.softmax_cross_entropy(x, t)
        // res = array_allclose(y.data, y2.data)
        // self.assertTrue(res)
        // y:variable(0.4401896)
        let x = Variable::new(RawData::from_shape_vec(
            vec![2, 4],
            vec![-1., 0., 1., 2., 2., 0., 1., -1.],
        ));
        let t = Variable::new(RawData::from_vec(vec![3, 0]));
        let y = softmax_cross_entropy(x.clone(), t.clone());
        //dbg!(&y);
        assert_close(0.4401896, y.get_data().flatten().to_vec()[0], 1e-4);

        y.backward();
    }

    #[test]
    fn test_softmax_cross_entropy_backward_01() {
        let x = Variable::new(RawData::from_shape_vec(
            vec![2, 4],
            vec![-1., 0., 1., 2., 2., 0., 1., -1.],
        ));
        let t = Variable::new(RawData::from_vec(vec![3, 0]));

        let mut sce =
            FunctionExecutor::new(Rc::new(RefCell::new(SoftmaxCrossEntropyFunction { t: t })));

        utils::gradient_check(&mut sce, vec![x.clone()]);
    }
}
