// ライブラリを一括でインポート
use crate::modules::*;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::cell::RefCell;
use std::rc::Rc;

/// Parameter 構造体
/// RawData 構造体のラッパーである。
/// 順伝播や逆伝播について、所有権の共有や内部可変が必要であるため
/// Rc と RefCell で RawData を保持する。
#[derive(Debug, Clone)]
pub struct Parameter<V: MathOps> {
    raw: Rc<RefCell<RawData<V>>>,
}

impl<V: MathOps> RawDataProcessor<V> for Parameter<V> {
    //impl<V: MathOps> Parameter<V> {
    /// コンストラクタ
    ///
    /// Arguments:
    /// * raw (RawData<V>): ラップする RawData
    ///
    /// Return:
    /// * Parameter<V>: RawData をラップしたインスタンス
    fn new(raw: RawData<V>) -> Self {
        Parameter {
            raw: Rc::new(RefCell::new(raw)),
        }
    }

    /// Rc、RefCell による参照の共有や内部可変に対応した RawData を取得する。
    fn raw(&self) -> Rc<RefCell<RawData<V>>> {
        // self.raw.clone()
        Rc::clone(&self.raw)
    }

    /// RawData の as_ref を返す。
    fn as_ref(&self) -> &RefCell<RawData<V>> {
        self.raw.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, IxDyn};
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
    fn test_parameter_params() {
        let var0 = RawData::new(1.0);
        assert_eq!(1, var0.get_size());
        let a: [usize; 0] = [];
        assert_eq!(&a, var0.get_shape());
        assert_eq!(0, var0.get_ndim());

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
    fn test_parameter_name() {
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
}
