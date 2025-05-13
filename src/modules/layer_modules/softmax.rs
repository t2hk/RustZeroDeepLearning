// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::Axis;

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
}
