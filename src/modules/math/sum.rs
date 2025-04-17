// ライブラリを一括でインポート
use crate::modules::math::*;
#[allow(unused_imports)]
use ::core::fmt::Debug;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, Axis, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// Sum 関数
#[derive(Debug, Clone)]
pub struct SumFunction {
    x_shape: Vec<usize>,
    // axis: Option<Axis>,
    axis: Option<Vec<isize>>,
    keepdims: bool,
}
impl<V: MathOps> Function<V> for SumFunction {
    /// 関数名を取得する。
    ///
    /// Return
    /// ＊String: 関数の名前
    fn get_name(&self) -> String {
        "Sum".to_string()
    }

    // Sum の順伝播
    fn forward(&self, x: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        info!("sum(forward)");

        let mut result;

        if let Some(axis) = self.axis.clone() {
            if self.keepdims {
                let tmp_x = x[0].sum_axis(Axis(axis[0] as usize));
                result = tmp_x.insert_axis(Axis(axis[0] as usize));
                debug!(
                    "sum(backward) {:?} -> {:?}",
                    x[0].flatten().to_vec(),
                    result.flatten().to_vec()
                );
            } else {
                result = x[0].sum_axis(Axis(axis[0] as usize));

                debug!(
                    "sum(backward) {:?} -> {:?}",
                    x[0].flatten().to_vec(),
                    result.flatten().to_vec()
                );
            }
        } else {
            if self.keepdims {
                let dims = x[0].shape().len();
                let mut tmp_x = x[0].sum_axis(Axis(0));
                for _i in 1..dims {
                    tmp_x = tmp_x.sum_axis(Axis(0));
                }

                result = tmp_x.insert_axis(Axis(0));
                for _i in 1..dims {
                    result = result.insert_axis(Axis(0));
                }
                debug!(
                    "sum(backward) {:?} -> {:?}",
                    x[0].flatten().to_vec(),
                    result.flatten().to_vec()
                );
            } else {
                result = Array::from_elem(IxDyn(&[]), x[0].sum());
                debug!(
                    "sum(backward) {:?} -> {:?}",
                    x[0].flatten().to_vec(),
                    result.flatten().to_vec()
                );
            }
        }

        vec![result]
    }

    /// 逆伝播
    fn backward(&self, _inputs: Vec<Variable<V>>, gys: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("sum(backward)");
        let gy = utils::reshape_sum_backward(
            gys[0].clone(),
            self.x_shape.clone(),
            self.axis.clone(),
            self.keepdims,
        );

        //let reshape_gy = Variable::new(RawVariable::new(gy));

        let gx = broadcast_to(gy.clone(), self.x_shape.clone());
        println!("self axis: {:?}", self.axis);

        debug!(
            "sum(backward) {:?} -> {:?}",
            gys[0].borrow().get_data().flatten().to_vec(),
            gx.borrow().get_data().flatten().to_vec()
        );
        vec![gx]
    }
}

/// Sum 関数
///
/// Arguments
/// * x (Rc<RefCell<Variable>>): 対象の変数
/// * axis (Option<Axis>): 軸
/// * keepdims (bool): 次元を維持するか
///
/// Return
/// * Variable<V>: 結果
pub fn sum<V: MathOps>(x: Variable<V>, axis: Option<Vec<isize>>, keepdims: bool) -> Variable<V> {
    let x_shape = x.borrow().get_data().shape().to_vec();
    let mut sum = FunctionExecutor::new(Rc::new(RefCell::new(SumFunction {
        x_shape: x_shape,
        axis: axis,
        keepdims: keepdims,
    })));
    // 順伝播
    sum.forward(vec![x.clone()]).get(0).unwrap().clone()
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
        let x0_var = Array::random_using((10, 10), Uniform::new(0., 10.), &mut rng);
        let x0 = Variable::new(RawVariable::from_shape_vec(
            vec![10, 10],
            x0_var.flatten().to_vec(),
        ));

        let x_shape = x0.borrow().get_data().shape().to_vec();

        let mut sum: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(SumFunction {
                x_shape: x_shape,
                axis: None,
                keepdims: false,
            })));

        utils::gradient_check(&mut sum, vec![x0.clone()]);
    }

    /// シンプルな全要素の和
    #[test]
    fn test_simple_sum() {
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

    /// シンプルな全要素の和
    #[test]
    fn test_simple_sum2() {
        let x = Variable::new(RawVariable::from_shape_vec(vec![2, 3], (1..7).collect()));
        let y = sum(x.clone(), Some(vec![0]), false);
        y.backward();

        // 順伝播結果
        assert_eq!(vec![5, 7, 9], y.borrow().get_data().flatten().to_vec());
        assert_eq!(vec![3], y.borrow().get_data().shape().to_vec());

        // 逆伝播結果
        // dbg!(&x.borrow().get_grad().unwrap());
        assert_eq!(
            vec![2, 3],
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

    /// keepdims を指定した全要素の和
    #[test]
    fn test_sum_keepdims() {
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 3, 4],
            (0..24).collect(),
        ));
        let y = sum(x.clone(), None, true);

        let tmp = y.borrow().get_data();

        assert_eq!(vec![276], tmp.flatten().to_vec());
        assert_eq!(vec![1, 1, 1], tmp.shape().to_vec());
    }

    /// keepdims を指定しない Axis(0) の和
    #[test]
    fn test_sum_axis0() {
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 3, 4],
            (0..24).collect(),
        ));
        let y = sum(x.clone(), Some(vec![0]), false);

        let tmp = y.borrow().get_data();
        dbg!(&tmp);

        assert_eq!(
            vec![12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
            tmp.flatten().to_vec()
        );
        assert_eq!(vec![3, 4], tmp.shape().to_vec());
    }

    /// keepdims を指定した Axis(0) の和
    #[test]
    fn test_sum_keepdims_axis0() {
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 3, 4],
            (0..24).collect(),
        ));
        let y = sum(x.clone(), Some(vec![0]), true);

        let tmp = y.borrow().get_data();
        dbg!(&tmp);

        assert_eq!(
            vec![12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
            tmp.flatten().to_vec()
        );
        assert_eq!(vec![1, 3, 4], tmp.shape().to_vec());

        y.backward();

        // 逆伝播結果
        // dbg!(&x.borrow().get_grad().unwrap());
        assert_eq!(
            vec![2, 3, 4],
            x.borrow().get_grad().unwrap().borrow().get_data().shape()
        );

        let grad: Vec<i32> = std::iter::repeat(1).take(24).collect();
        assert_eq!(
            grad,
            x.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .flatten()
                .to_vec()
        );
    }

    /// keepdims を指定しない Axis(1) の和
    #[test]
    fn test_sum_axis1() {
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 3, 4],
            (0..24).collect(),
        ));
        let y = sum(x.clone(), Some(vec![1]), false);

        let tmp = y.borrow().get_data();
        dbg!(&tmp);

        assert_eq!(vec![12, 15, 18, 21, 48, 51, 54, 57], tmp.flatten().to_vec());
        assert_eq!(vec![2, 4], tmp.shape().to_vec());

        y.backward();

        // 逆伝播結果
        // dbg!(&x.borrow().get_grad().unwrap());
        assert_eq!(
            vec![2, 3, 4],
            x.borrow().get_grad().unwrap().borrow().get_data().shape()
        );

        let grad: Vec<i32> = std::iter::repeat(1).take(24).collect();
        assert_eq!(
            grad,
            x.borrow()
                .get_grad()
                .unwrap()
                .borrow()
                .get_data()
                .flatten()
                .to_vec()
        );
    }

    /// keepdims を指定した Axis(1) の和
    #[test]
    fn test_sum_keepdims_axis1() {
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![2, 3, 4],
            (0..24).collect(),
        ));
        let y = sum(x.clone(), Some(vec![1]), true);

        let tmp = y.borrow().get_data();
        dbg!(&tmp);

        assert_eq!(vec![12, 15, 18, 21, 48, 51, 54, 57], tmp.flatten().to_vec());
        assert_eq!(vec![2, 1, 4], tmp.shape().to_vec());

        y.backward();

        // 逆伝播結果
        // dbg!(&x.borrow().get_grad().unwrap());
        assert_eq!(
            vec![2, 3, 4],
            x.borrow().get_grad().unwrap().borrow().get_data().shape()
        );

        let grad: Vec<i32> = std::iter::repeat(1).take(24).collect();
        assert_eq!(
            grad,
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
