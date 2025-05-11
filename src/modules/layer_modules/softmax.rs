// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

pub fn softmax1d<V: MathOps>(x: Variable<V>) -> Variable<V> {
    let y = exp(x);
    let sum_y = sum(y.clone(), None, false);

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
}
