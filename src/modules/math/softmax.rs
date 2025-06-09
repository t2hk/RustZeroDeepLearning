// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, Array1, Axis, IxDyn};
use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

use super::math::logarithm::log;

/// Softmax 関数
#[derive(Debug, Clone)]
pub struct SoftmaxFunction {
    axis: Axis,
}

impl<V: MathOps> Function<V> for SoftmaxFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Softmax".to_string()
    }

    // 順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        // class Softmax(Function):
        //     def __init__(self, axis=1):
        //         self.axis = axis
        //
        //     def forward(self, x):
        //         xp = cuda.get_array_module(x)
        //         y = x - x.max(axis=self.axis, keepdims=True)
        //         y = xp.exp(y)
        //         y /= y.sum(axis=self.axis, keepdims=True)
        //         return y

        info!("softmax(forward)");

        let x_max = xs[0]
            .map_axis(self.axis, |view| {
                // view.iter().max().unwrap().clone()
                let tmp_x: Vec<f64> = view.iter().map(|x| V::to_f64(x).unwrap()).collect();
                let max = tmp_x.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
                V::from(max.to_owned()).unwrap()
            })
            .insert_axis(self.axis);

        let y = xs[0].clone() - x_max;
        let y_exp = y.mapv_into(|y| V::from(V::to_f64(&y).unwrap().exp()).unwrap());
        let y_exp_sum = y_exp.sum_axis(self.axis).insert_axis(self.axis);
        let y_div_y_exp_sum = y_exp.clone() / y_exp_sum;

        let outputs = vec![y_div_y_exp_sum];

        outputs
    }

    /// 逆伝播
    fn backward(
        &self,
        _inputs: Vec<Variable<V>>,
        outputs: Vec<Weak<RefCell<RawData<V>>>>,
        gys: Vec<Variable<V>>,
    ) -> Vec<Variable<V>> {
        // class Softmax(Function):
        //     def backward(self, gy):
        //         y = self.outputs[0]()
        //         gx = y * gy
        //         sumdx = gx.sum(axis=self.axis, keepdims=True)
        //         gx -= y * sumdx
        //         return gx

        info!("softmax(backward)");

        let y = outputs[0].upgrade().unwrap().borrow().get_data().clone();

        let gx = y.clone() * gys[0].get_data();
        let sumdx = gx.clone().sum_axis(self.axis).insert_axis(self.axis);
        let result_array = gx - (y * sumdx);

        let result = Variable::new(RawData::from_shape_vec(
            result_array.shape().to_vec(),
            result_array.flatten().to_vec(),
        ));

        vec![result]
    }
}

/// softmax 関数
///
/// Arguments
/// * x (Variable<V>): 変数
///
/// Return
/// * Rc<RefCell<RawData>>: log 結果
pub fn softmax<V: MathOps>(x: Variable<V>, axis: Axis) -> Variable<V> {
    debug!("SoftmaxFunction::softmax");

    let mut softmax = FunctionExecutor::new(Rc::new(RefCell::new(SoftmaxFunction { axis: axis })));

    // 順伝播
    let output = softmax.forward(vec![x.clone()]).first().unwrap().clone();
    output
}

/// ソフトマックス関数
///
/// Arguments
/// * x (Variable<V>): 入力変数
///
/// Return
/// * Variable<V>: 分類した確率
pub fn softmax1d<V: MathOps>(x: Variable<V>) -> Variable<V> {
    let y = exp(x);
    let sum_y = sum(y.clone(), None, false);

    return &y / &sum_y;
}

/// ソフトマックス関数
/// バッチとしてまとめられたデータに対して適用できるように軸を指定する。
///
/// Arguments
/// * x (Variable<V>): 入力変数
/// * axis (isize): 軸
///
/// Return
/// * Variable<V>: 分類した確率
pub fn softmax_simple<V: MathOps>(x: Variable<V>, axis: isize) -> Variable<V> {
    let y = exp(x);

    let sum_y = sum(y.clone(), Some(vec![axis]), true);

    return &y / &sum_y;
}

/// 交差エントロピー誤差
///
/// Arguments
/// * x (Variable<V>): ニューラルネットワークのソフトマックス関数を適用する前の出力値
/// * t (Variable<V>): 教師データ(正解となるクラスの番号)
/// Return
/// * V (数値型): 交差エントロピー誤差
pub fn softmax_cross_entropy_simple<V: MathOps>(x: Variable<V>, t: Variable<V>) -> V {
    let n = x.get_data().shape().to_vec()[0];

    let p = softmax_simple(x.clone(), 1);
    // let p_clipped = p.clone().clamp(1e-15, 1.0);
    let p_data = p.get_data();
    let p_clipped = p_data.clamp(V::from(1e-15).unwrap(), V::one());
    p.set_data(p_clipped.clone());

    let log_p = log(p.clone());

    let mut tlog_p_vec = vec![];

    for i in 0..n {
        let t_val = V::to_usize(&t.get_data().flatten().to_vec()[i]).unwrap();
        let log_p_val = log_p.get_data()[[i, t_val]].clone();
        tlog_p_vec.push(log_p_val);
    }
    let tlog_p = Array::from_shape_vec((1, n), tlog_p_vec).unwrap();

    let y = (V::from(-1).unwrap() * tlog_p.sum()) / V::from(n).unwrap();

    y
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use ndarray::Array;

    use super::*;

    #[test]
    fn test_softmax1d_1() {
        // x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        // y = F.softmax1d(Variable(x))
        // y: variable([[0.01349627 0.03668667 0.0997247 ] [0.01349627 0.0997247  0.73687136]])
        let x = Variable::new(RawData::from_shape_vec(
            vec![2, 3],
            vec![0., 1., 2., 0., 2., 4.],
        ));

        let y = softmax1d(x.clone());
        let expect = vec![
            0.01349627, 0.03668667, 0.0997247, 0.01349627, 0.0997247, 0.73687136,
        ];
        let expect_array = Array::from_shape_vec(vec![2, 3], expect).unwrap();

        assert!(y.get_data().abs_diff_eq(&expect_array, 1e-4));
    }

    #[test]
    fn test_softmax1d_2() {
        // x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        // y = F.softmax1d(Variable(x))
        // y: variable([[0.01349627 0.03668667 0.0997247 ] [0.01349627 0.0997247  0.73687136]])
        let x = Variable::new(RawData::from_shape_vec(vec![1, 2], vec![0.2, -0.4]));

        let sigmoid = Rc::new(RefCell::new(SigmoidFunction {}));
        let sgd = Sgd::new(0.2);
        let mut mlp = Mlp::new(vec![10, 3], sigmoid, sgd);

        let y = mlp.forward(vec![x.clone()]);
        let p = softmax1d(y[0].clone());

        println!("p: {:?}", p.get_data());
    }

    #[test]
    fn test_softmax_cross_entropy_simple_01() {
        let x = Variable::new(RawData::from_shape_vec(
            vec![2, 4],
            vec![-1., 0., 1., 2., 2., 0., 1., -1.],
        ));
        let t = Variable::new(RawData::from_vec(vec![3., 0.]));
        let y = softmax_cross_entropy_simple(x.clone(), t.clone());
        println!("y: {}", y);
        // y: variable(0.4401896)
        let expect = 0.4401896;

        assert!(1e-4 > (y - expect));
    }

    #[test]
    fn test_softmax_forward_01() {
        // def test_forward1(self):
        //     x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        //     y2 = CF.softmax(x, axis=1)
        //     y = F.softmax(Variable(x))
        //     res = array_allclose(y.data, y2.data)
        //     self.assertTrue(res)
        // y: variable([[0.09003057 0.24472849 0.66524094] [0.01587624 0.11731042 0.8668133 ]])
        let x = Variable::new(RawData::from_shape_vec(
            vec![2, 3],
            vec![0., 1., 2., 0., 2., 4.],
        ));
        let y = softmax(x.clone(), Axis(1));
        assert_eq!(vec![2, 3], y.get_data().shape().to_vec());

        let expect_values = vec![
            0.09003057, 0.24472849, 0.66524094, 0.01587624, 0.11731042, 0.8668133,
        ];

        let expects = Array::from_shape_vec(y.get_data().shape().to_vec(), expect_values).unwrap();
        assert!(y.get_data().abs_diff_eq(&expects, 1e-4));
    }

    #[test]
    fn test_softmax_backward_01() {
        let x = Variable::new(RawData::from_shape_vec(
            vec![2, 3],
            vec![0., 1., 2., 0., 2., 4.],
        ));
        let y = softmax(x.clone(), Axis(1));
        assert_eq!(vec![2, 3], y.get_data().shape().to_vec());

        let expect_values = vec![
            0.09003057, 0.24472849, 0.66524094, 0.01587624, 0.11731042, 0.8668133,
        ];

        let expects = Array::from_shape_vec(y.get_data().shape().to_vec(), expect_values).unwrap();
        assert!(y.get_data().abs_diff_eq(&expects, 1e-4));

        let mut softmax =
            FunctionExecutor::new(Rc::new(RefCell::new(SoftmaxFunction { axis: Axis(1) })));

        utils::gradient_check(&mut softmax, vec![x.clone()]);
    }
}
