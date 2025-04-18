// ライブラリを一括でインポート
use crate::modules::*;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, ArrayD, IntoDimension, IxDyn};
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

use super::math::{reshape, transpose};

/// Variable 構造体
/// RawData 構造体のラッパーである。
/// 順伝播や逆伝播について、所有権の共有や内部可変が必要であるため
/// Rc と RefCell で RawData を保持する。
#[derive(Debug, Clone)]
pub struct Variable<V: MathOps> {
    raw: Rc<RefCell<RawData<V>>>,
}
impl<V: MathOps> Variable<V> {
    /// コンストラクタ
    ///
    /// Arguments:
    /// * raw (RawData<V>): ラップする RawData
    ///
    /// Return:
    /// * Variable<V>: RawData をラップしたインスタンス
    pub fn new(raw: RawData<V>) -> Self {
        Variable {
            raw: Rc::new(RefCell::new(raw)),
        }
    }

    /// Rc、RefCell による参照の共有や内部可変に対応した RawData を取得する。
    pub fn raw(&self) -> Rc<RefCell<RawData<V>>> {
        // self.raw.clone()
        Rc::clone(&self.raw)
    }

    /// RawData の borrow を返す。
    pub fn borrow(&self) -> Ref<RawData<V>> {
        self.raw.borrow()
    }

    /// RawData の borrow_mut を返す。
    pub fn borrow_mut(&self) -> RefMut<RawData<V>> {
        self.raw.borrow_mut()
    }

    /// RawData の as_ref を返す。
    pub fn as_ref(&self) -> &RefCell<RawData<V>> {
        self.raw.as_ref()
    }

    /// 変数を設定する。
    ///
    /// Arguments:
    /// * data (Array<V, IxDyn>): 変数
    pub fn set_data(&mut self, data: Array<V, IxDyn>) {
        self.raw().borrow_mut().set_data(data);
    }

    /// 逆伝播を実行する。
    pub fn backward(&self) {
        self.raw.as_ref().clone().borrow().backward();
    }

    pub fn detail(&self) -> String {
        let data_detail = format!("data: {:?}", self.raw().borrow().get_data());
        let grad_detail = format!(
            "grad: {:?}",
            match self.raw().borrow().get_grad() {
                Some(grad) => format!("{:?}", grad.borrow().get_data()),
                None => "None".to_string(),
            }
        );
        let detail = format!(
            "variable name: {},generation: {}, data:{:?}, grad: {:?}",
            match self.raw().borrow().get_name() {
                Some(name) => name,
                None => "None".to_string(),
            },
            self.raw().borrow().get_generation(),
            data_detail,
            grad_detail
        );
        detail.to_string()
    }

    /// リシェイプ
    ///
    /// Arguments:
    /// * shape (Vec<usize>): 変更後の形状
    ///
    /// Return:
    /// * Variable<RawData<V>>
    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        return reshape(self.clone(), shape.clone());
    }

    /// 転値
    ///
    /// Return:
    /// * Variable<RawData<V>>: 転値結果
    pub fn transpose(&self) -> Self {
        return transpose(self.clone());
    }
}

/// RawData 構造体を生成するためのトレイト
/// * create_variable: RawData 構造体を生成する
pub trait CreateVariable<V: MathOps> {
    fn create_variable(&self) -> RawData<V>;
}

/// CreateVariable トレイトの Array<f64, IxDyn> 用の実装
impl<V: MathOps> CreateVariable<V> for Array<V, IxDyn> {
    fn create_variable(&self) -> RawData<V> {
        RawData::new(self.clone())
    }
}

/// CreateVariable トレイトの 数値用の実装
impl<V: MathOps> CreateVariable<V> for V {
    fn create_variable(&self) -> RawData<V> {
        RawData::new(Array::from_elem(IxDyn(&[]), self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    #[test]
    /// 変数の型名に関するテスト。
    fn test_get_dtype() {
        let var_i8 = RawData::new(10i8);
        let var_i16 = RawData::new(10i16);
        let var_i32 = RawData::new(10i32);
        let var_i64 = RawData::new(10i64);
        let var_f32 = RawData::new(10.0f32);
        let var_f64 = RawData::new(10.0f64);
        let var_u8 = RawData::new(10u8);
        let var_u16 = RawData::new(10u16);
        let var_u32 = RawData::new(10u32);
        let var_u64 = RawData::new(10u64);
        assert_eq!("i8", var_i8.get_dtype());
        assert_eq!("i16", var_i16.get_dtype());
        assert_eq!("i32", var_i32.get_dtype());
        assert_eq!("i64", var_i64.get_dtype());
        assert_eq!("f32", var_f32.get_dtype());
        assert_eq!("f64", var_f64.get_dtype());
        assert_eq!("u8", var_u8.get_dtype());
        assert_eq!("u16", var_u16.get_dtype());
        assert_eq!("u32", var_u32.get_dtype());
        assert_eq!("u64", var_u64.get_dtype());
    }

    /// 変数の size, shape, ndim のテスト
    #[test]
    fn test_variable_params() {
        let var0 = RawData::new(1.0);
        assert_eq!(1, var0.get_size());
        let a: [usize; 0] = [];
        assert_eq!(&a, var0.get_shape());
        assert_eq!(0, var0.get_ndim());

        let var1 = RawData::from_shape_vec(vec![1], vec![1.0]);

        let sh2x2 = vec![2, 2];
        let val2x2 = vec![1., 2., 3., 4.];
        let var2x2 = RawData::from_shape_vec(sh2x2, val2x2);

        assert_eq!(4, var2x2.get_size());
        assert_eq!([2, 2], var2x2.get_shape());
        assert_eq!(2, var2x2.get_ndim());
        dbg!(&var2x2.get_shape());

        let sh10x20x30x40x50 = vec![10, 20, 30, 40, 50];
        let val10x20x30x40x50: Vec<f64> = (1..=12000000).map(|x| x as f64).collect();

        let var10x20x30x40x50 = RawData::from_shape_vec(sh10x20x30x40x50, val10x20x30x40x50);
        assert_eq!(12000000, var10x20x30x40x50.get_size());
        assert_eq!([10, 20, 30, 40, 50], var10x20x30x40x50.get_shape());
        assert_eq!(5, var10x20x30x40x50.get_ndim());
    }

    #[test]
    /// 任意の形状に関するテスト。
    fn test_dyndim_array() {
        let shape = vec![2, 2, 2];
        let values = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let var = RawData::from_shape_vec(shape, values);
        // dbg!(&var);
        assert_eq!(&[2, 2, 2], var.get_data().shape());
    }

    /// 変数の名前のテスト。
    #[test]
    fn test_variable_name() {
        let mut val = RawData::new(Array::from_elem(IxDyn(&[100, 100, 100]), 1.0));

        assert_eq!(None, val.get_name());

        val.set_name("test_val".to_string());
        assert_eq!(Some("test_val".to_string()), val.get_name());
    }

    /// linspace のテスト
    #[test]
    fn test_linspace() {
        let start = -5.0;
        let end = 5.0;
        let n = 20usize;
        let var = RawData::linspace(start, end, n);
        assert_eq!(n, var.get_size());
        assert_eq!(&[n], var.get_shape());
        let max = var.get_data().mapv(|x| x).get(n - 1).unwrap().clone();
        let min = var.get_data().mapv(|x| x).get(0).unwrap().clone();
        assert_eq!(start, min);
        assert_eq!(end, max);
    }

    /// Variable のリシェイプのテスト
    #[test]
    fn test_variable_reshape() {
        let x = Variable::new(RawData::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]));

        let r1 = x.reshape(vec![6]);
        assert_eq!(vec![6], r1.borrow().get_data().shape().to_vec());
        assert_eq!(
            vec![1, 2, 3, 4, 5, 6],
            r1.borrow().get_data().flatten().to_vec()
        );

        let r2 = r1.reshape(vec![3, 2]);
        assert_eq!(vec![3, 2], r2.borrow().get_data().shape().to_vec());
        assert_eq!(
            vec![1, 2, 3, 4, 5, 6],
            r2.borrow().get_data().flatten().to_vec()
        );
    }

    /// 行列の転値のテスト
    #[test]
    fn test_transpose() {
        let input_shape = vec![2, 3];
        let x = Variable::new(RawData::from_shape_vec(
            input_shape.clone(),
            vec![1, 2, 3, 4, 5, 6],
        ));
        dbg!(&x);

        let y = x.transpose();

        // 転値後の確認
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
