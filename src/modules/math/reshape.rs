// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// リシェイプ関数
#[derive(Debug, Clone)]
pub struct ReshapeFunction {
    shape: Vec<usize>,   // 変換後
    x_shape: Vec<usize>, // 変換前
}

impl<V: MathOps> Function<V> for ReshapeFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Reshape".to_string()
    }

    /// 順伝播
    /// 関数のインスタンス作成時に変換する Shape(Vec<usize>) を指定すること。
    ///
    /// Arguments
    /// * xs (Vec<Array<V,IxDyn>>): 変換対象のテンソル
    ///
    /// Returns
    /// * Vec<Array<V, IxDyn>>: 累乗の結果
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("reshape(forward)");
        debug!("reshape(forwad) {:?} -> {:?}", &xs[0].shape(), self.shape);

        let y = xs[0].clone().into_shape_clone(self.shape.clone()).unwrap();
        return vec![y];
    }

    /// 逆伝播
    /// 順伝播の入力値と同じ形状に微分値の形状を返還する。
    fn backward(
        &self,
        _inputs: Vec<Variable<V>>,
        _outputs: Vec<Weak<RefCell<RawData<V>>>>,
        gys: Vec<Variable<V>>,
    ) -> Vec<Variable<V>> {
        info!("reshape(backward)");
        debug!(
            "reshape(backward) {:?} -> {:?}",
            gys[0].get_data().shape(),
            &self.x_shape
        );

        dbg!(&gys[0]);
        let reshaped_data = gys[0]
            .get_data()
            .into_shape_clone(self.x_shape.clone())
            .unwrap();

        let mut gy = gys[0].clone();
        gy.set_data(reshaped_data);

        vec![gy]
    }
}

/// リシェイプ関数
///
/// Arguments
/// * input (Variable<V>): テンソル
/// * shape (Vec<usize>): 変換後の形状
///
/// Return
/// * Variable<V>: リシェイプ語の結果
pub fn reshape<V: MathOps>(input: Variable<V>, shape: Vec<usize>) -> Variable<V> {
    let x_shape = input.get_data().shape().to_vec();

    let mut reshape = FunctionExecutor::new(Rc::new(RefCell::new(ReshapeFunction {
        x_shape: x_shape,
        shape: shape,
    })));

    // 順伝播
    reshape.forward(vec![input]).get(0).unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 行列の形状変更のテスト
    #[test]
    fn test_reshape() {
        let input_shape = vec![2, 3];
        let x = Variable::new(RawData::from_shape_vec(
            input_shape.clone(),
            vec![1, 2, 3, 4, 5, 6],
        ));

        let y = reshape(x.clone(), vec![6]);

        // 形状変更後の確認
        assert_eq!(vec![6], y.get_data().shape().to_vec());
        assert_eq!(vec![1, 2, 3, 4, 5, 6], y.get_data().flatten().to_vec());

        y.backward();

        let x_grad = x.get_grad().unwrap();
        // dbg!(&x_grad);

        // 微分値が入力値と同じ形状で、全て1であることを確認する。
        assert_eq!(input_shape, x_grad.get_data().shape().to_vec());
        assert_eq!(vec![1, 1, 1, 1, 1, 1], x_grad.get_data().flatten().to_vec());
    }
}
