// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// 転置関数
#[derive(Debug, Clone)]
pub struct TransposeFunction {
    axes: Option<Vec<usize>>,
}

impl<V: MathOps> Function<V> for TransposeFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Transpose".to_string()
    }

    /// 順伝播
    ///
    /// Arguments
    /// * xs (Vec<Array<V,IxDyn>>): 変換対象のテンソル
    ///
    /// Returns
    /// * Vec<Array<V, IxDyn>>: 転置の結果
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        if let Some(axes) = self.axes.clone() {
            info!("transpose_axes(forward)");
            let y = xs[0].clone().permuted_axes(axes).to_owned();

            debug!(
                "transpose_axes(forwad) {:?} -> {:?}",
                &xs[0].shape(),
                y.shape()
            );

            vec![y]
        } else {
            info!("transpose(forward)");
            let y = xs[0].clone().t().to_owned();

            debug!("transpose(forwad) {:?} -> {:?}", &xs[0].shape(), y.shape());

            vec![y]
        }
    }

    /// 逆伝播
    fn backward(&self, _inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        if let Some(axes) = self.axes.clone() {
            info!("transpose_axes(backward)");

            let axes_len = axes.len();
            let inv_axes: Vec<usize> = axes.iter().map(|&ax| ax.rem_euclid(axes_len)).collect();

            let mut indices: Vec<usize> = (0..axes_len).collect();
            indices.sort_by_key(|&i| inv_axes[i]);

            let permuted = transpose_axes(gys[0].clone(), indices.clone());
            vec![permuted]
        } else {
            info!("transpose(backward)");
            let t_gy = transpose(gys[0].clone());

            debug!(
                "transpose(backward) {:?} -> {:?}",
                gys[0].borrow().get_data().shape(),
                t_gy.borrow().get_data().shape(),
            );

            vec![t_gy]
        }
    }
}

/// 転置関数
///
/// Arguments
/// * input (Variable<V>): 転置対象
///
/// Return
/// * Variable<V>: 転置の結果
pub fn transpose<V: MathOps>(input: Variable<V>) -> Variable<V> {
    let mut transpose =
        FunctionExecutor::new(Rc::new(RefCell::new(TransposeFunction { axes: None })));

    // 順伝播
    transpose.forward(vec![input]).get(0).unwrap().clone()
}

/// 転置関数
///
/// Arguments
/// * input (Variable<V>): 転置対象
/// * axes (Option<Vec<usize>>): 入れ替える軸
///
/// Return
/// * Variable<V>: 転置の結果
pub fn transpose_axes<V: MathOps>(input: Variable<V>, axes: Vec<usize>) -> Variable<V> {
    let mut transpose = FunctionExecutor::new(Rc::new(RefCell::new(TransposeFunction {
        axes: Some(axes),
    })));

    // 順伝播
    transpose.forward(vec![input]).get(0).unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray_rand::RandomExt;
    use rand::{distributions::Uniform, SeedableRng};
    use rand_isaac::Isaac64Rng;

    #[test]
    fn test_num_grad() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let x0_var = Array::random_using((4, 3, 2), Uniform::new(0., 10.), &mut rng);
        let x0 = Variable::new(RawData::from_shape_vec(
            vec![4, 3, 2],
            x0_var.flatten().to_vec(),
        ));

        let mut transpose: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(TransposeFunction {
                axes: Some(vec![1, 0, 2]),
            })));

        utils::gradient_check(&mut transpose, vec![x0.clone()]);
    }

    #[test]
    fn test_transpose_axes() {
        let input_shape = vec![4, 3, 2];
        let x = Variable::new(RawData::from_shape_vec(
            input_shape.clone(),
            (0..=23).collect::<Vec<i32>>(),
        ));
        // dbg!(&x);

        let y = transpose_axes(x.clone(), vec![1, 0, 2]);
        // dbg!(&y);

        let y_data = y.borrow().get_data();
        // 正しく軸が入れ替わっていることを確認 (4, 3, 2) -> (3, 4, 2)
        assert_eq!(vec![3, 4, 2], y_data.shape().to_vec());

        let expect_rows = vec![
            [0, 1],
            [6, 7],
            [12, 13],
            [18, 19],
            [2, 3],
            [8, 9],
            [14, 15],
            [20, 21],
            [4, 5],
            [10, 11],
            [16, 17],
            [22, 23],
        ];

        let y_rows = y_data.rows();
        // 全ての行に対してイテレーションし、転置後の値が正しいかチェックする。
        for (i, row) in y_rows.into_iter().enumerate() {
            assert_eq!(
                format!("{:?}", expect_rows[i]),
                format!("{:?}", row.flatten().to_vec())
            );
        }

        y.backward();
        let x_grad = x.borrow().get_grad().unwrap().borrow().get_data();

        // 逆伝播後の勾配が入力値の形状と一致することを確認する。
        assert_eq!(vec![4, 3, 2], x_grad.shape().to_vec());
        //dbg!(&x_grad);
    }

    /// 行列の転置のテスト
    #[test]
    fn test_transpose() {
        let input_shape = vec![2, 3];
        let x = Variable::new(RawData::from_shape_vec(
            input_shape.clone(),
            vec![1, 2, 3, 4, 5, 6],
        ));
        dbg!(&x);

        let y = transpose(x.clone());

        // 転置後の確認
        assert_eq!(vec![3, 2], y.borrow().get_data().shape().to_vec());
        assert_eq!(
            vec![1, 4, 2, 5, 3, 6],
            y.borrow().get_data().flatten().to_vec()
        );

        y.backward();

        let x_grad = x.borrow().get_grad().unwrap();
        dbg!(&x_grad);

        // 微分値が入力値と同じ形状で、全て1であることを確認する。
        assert_eq!(input_shape, x_grad.borrow().get_data().shape().to_vec());
        assert_eq!(
            vec![1, 1, 1, 1, 1, 1],
            x_grad.borrow().get_data().flatten().to_vec()
        );
    }
}
