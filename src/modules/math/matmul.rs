// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, ArrayD, Axis, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// Sum 関数
#[derive(Debug, Clone)]
pub struct MatmulFunction {}
impl<V: MathOps> Function<V> for MatmulFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Matmul".to_string()
    }

    // matmul の順伝播
    fn forward(&self, inputs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("matmul(forward)");

        // let x = inputs[0];
        // let w = inputs[1];

        // let x = ArrayD::from_shape_vec(inputs[0].shape().to_vec(), inputs[0].flatten().to_vec())
        //     .unwrap()
        //     .into_dimensionality()
        //     .unwrap();

        // let w = ArrayD::from_shape_vec(inputs[1].shape().to_vec(), inputs[0].flatten().to_vec())
        //     .unwrap()
        //     .into_dimensionality()
        //     .unwrap();

        let x_dim = utils::get_ixdim(&inputs[0]).unwrap();
        match x_dim {
            Ok(FixedDimArray::Dim1(arr)) => arr,
            Ok(FixedDimArray::Dim2(arr)) => arr,
            Ok(FixedDimArray::Dim3(arr)) => arr,
            Ok(FixedDimArray::Dim4(arr)) => arr,
            Ok(FixedDimArray::Dim5(arr)) => arr,
            Ok(FixedDimArray::Dim6(arr)) => arr,
            _ => println!("Invalid or unsupported dimension"),
        }

        dbg!(&x_dim);

        // let x = Array::from_shape_vec(inputs[0].shape().to_vec(), inputs[0].flatten().to_vec())
        //     .unwrap()
        //     .into_dimensionality(x_dim.)
        //     .unwrap();

        // let y = &x.dot(&w);

        inputs
    }

    /// 逆伝播
    fn backward(&self, _inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("matmul(backward)");

        //let reshape_gy = Variable::new(RawVariable::new(gy));

        gys
    }
}

/// matmul 関数
///
/// Arguments
/// * x (Variable<V>): 変数
/// * w (Variable<V>): 変数
///
/// Return
/// * Variable<V>: 結果
pub fn matmul<V: MathOps>(x: Variable<V>, w: Variable<V>) -> Variable<V> {
    let x_shape = x.borrow().get_data().shape().to_vec();
    let mut matmul = FunctionExecutor::new(Rc::new(RefCell::new(MatmulFunction {})));
    // 順伝播
    matmul
        .forward(vec![x.clone(), w.clone()])
        .get(0)
        .unwrap()
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_test() {
        let x = Variable::new(RawVariable::from_shape_vec(vec![1, 6], (1..7).collect()));
        let y = matmul(x.clone(), x.clone());
    }

    /// シンプルな行列の積
    #[test]
    fn test_simple_matmul() {
        let x = Variable::new(RawVariable::from_shape_vec(vec![1, 6], (1..7).collect()));
        let y = sum(x.clone(), None, false);
        y.backward();

        assert_eq!(vec![21], y.borrow().get_data().flatten().to_vec());

        assert_eq!(
            vec![1, 6],
            x.borrow().get_grad().unwrap().borrow().get_data().shape()
        );
        assert_eq!(
            vec![1, 1, 1, 1, 1, 1],
            x.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .flatten()
                .to_vec()
        );

        // 逆伝播結果
        // dbg!(&x.borrow().get_grad().unwrap());
        assert_eq!(
            vec![1, 6],
            x.borrow().get_grad().unwrap().borrow().get_data().shape()
        );
        assert_eq!(
            vec![1, 1, 1, 1, 1, 1],
            x.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .flatten()
                .to_vec()
        );
    }
}
