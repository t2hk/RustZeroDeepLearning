// ライブラリを一括でインポート
use crate::modules::*;

use ndarray::{Array, ArrayD, IntoDimension, IxDyn};
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

/// RawVariable 構造体
/// 変数自体、および逆伝播に必要な情報を保持する。
///
/// * data (Array<f64, IxDyn>): 変数
/// * name (Option<String>): 変数の名前
/// * grad (Option<Array<f64, IxDyn>): 変数に対応した微分した値。逆伝播によって実際に微分が計算されたときに値を設定する。
/// * creator (Option<Rc<RefCell<FunctionExecutor>>>): この変数を生成した関数
/// * generation (i32): 計算グラフ上の世代
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawVariable<V: MathOps> {
    data: Array<V, IxDyn>,
    name: Option<String>,
    grad: Option<Array<V, IxDyn>>,
    creator: Option<Rc<RefCell<FunctionExecutor<V>>>>,
    generation: i32,
}

// impl<V: MathOps> PartialOrd for RawVariable<V> {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         let lhs_data = &self.data[[]];
//         let rhs_data = &other.data[[]];

//         if lhs_data == rhs_data {
//             Some(Ordering::Equal)
//         } else if lhs_data > rhs_data {
//             Some(Ordering::Greater)
//         } else {
//             Some(Ordering::Less)
//         }
//     }
// }

// impl<V: MathOps> Ord for RawVariable<V> {
//     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//         self.data[[]].cmp(&other.data[[]])
//     }
// }

/// Variable 構造体
/// RawVariable 構造体のラッパーである。
/// 順伝播や逆伝播について、所有権の共有や内部可変が必要であるため
/// Rc と RefCell で RawVariable を保持する。
#[derive(Debug, Clone)]
pub struct Variable<V: MathOps> {
    raw: Rc<RefCell<RawVariable<V>>>,
}
impl<V: MathOps> Variable<V> {
    /// コンストラクタ
    ///
    /// Arguments:
    /// * raw (RawVariable<V>): ラップする RawVariable
    ///
    /// Return:
    /// * Variable<V>: RawVariable をラップしたインスタンス
    pub fn new(raw: RawVariable<V>) -> Self {
        Variable {
            raw: Rc::new(RefCell::new(raw)),
        }
    }

    /// Rc、RefCell による参照の共有や内部可変に対応した RawVariable を取得する。
    pub fn raw(&self) -> Rc<RefCell<RawVariable<V>>> {
        // self.raw.clone()
        Rc::clone(&self.raw)
    }

    /// RawVariable の borrow を返す。
    pub fn borrow(&self) -> Ref<RawVariable<V>> {
        self.raw.borrow()
    }

    /// RawVariable の borrow_mut を返す。
    pub fn borrow_mut(&self) -> RefMut<RawVariable<V>> {
        self.raw.borrow_mut()
    }

    /// RawVariable の as_ref を返す。
    pub fn as_ref(&self) -> &RefCell<RawVariable<V>> {
        self.raw.as_ref()
    }

    /// 変数を設定する。
    ///
    /// Arguments:
    /// * data (Array<V, IxDyn>): 変数
    pub fn set_data(&mut self, data: Array<V, IxDyn>) {
        self.raw().borrow_mut().data = data;
    }

    /// 逆伝播を実行する。
    pub fn backward(&self) {
        self.raw.as_ref().clone().borrow().backward();
    }
}

impl<V: MathOps> RawVariable<V> {
    /// RawVariable のコンストラクタ。
    ///
    /// # Arguments
    /// * data - 変数    
    pub fn new<T: CreateVariable<V>>(data: T) -> RawVariable<V> {
        CreateVariable::create_variable(&data)
    }

    /// RawVariable を次元と値から生成する。
    /// 以下のように使用する。
    ///   let dim = vec![2, 2, 2];
    ///   let values = vec![1., 2., 3., 4., 5., 6., 7., 8.];
    ///   let variable = RawVariable::new(dim, values);
    ///
    /// Arguments
    /// * shape (Vec<i32>): 次元
    /// * values (Vec<f64>): 変数
    ///
    /// Returns
    /// * Result<Self, ShapeError>
    pub fn from_shape_vec<Sh>(shape: Sh, values: Vec<V>) -> Self
    where
        Sh: IntoDimension<Dim = IxDyn>,
    {
        let dim = shape.into_dimension();
        let array = ArrayD::from_shape_vec(dim, values).expect("Shape error while creating array");
        Self {
            data: array,
            name: None,
            grad: None,
            creator: None,
            generation: 0,
        }
    }

    /// この変数を生成した関数を設定する。
    ///
    /// Arguments
    /// * creator (Rc<RefCell<FunctionExecutor>>): 関数のラッパー
    pub fn set_creator(&mut self, creator: Rc<RefCell<FunctionExecutor<V>>>) {
        self.creator = Some(Rc::clone(&creator));
        self.generation = creator.borrow().get_generation() + 1;
    }

    /// この変数を算出した関数を取得する。
    ///
    /// Return
    /// * Option<Rc<RefCell<FunctionExecutor<V>>>>: 関数
    pub fn get_creator(&self) -> Option<Rc<RefCell<FunctionExecutor<V>>>> {
        if let Some(creator) = self.creator.clone() {
            // Some(Rc::clone(&self.creator.clone().unwrap()))
            Some(Rc::clone(&creator.clone()))
        } else {
            None
        }
    }

    /// 微分をリセットする。
    pub fn clear_grad(&mut self) {
        self.grad = None;
    }

    /// 変数の盛大を取得する。
    ///
    /// Return
    /// i32: 世代
    pub fn get_generation(&self) -> i32 {
        self.generation
    }

    /// 生成した関数の世代を取得する。
    ///
    /// Return
    /// i32: 生成した関数の世代
    pub fn get_creator_generation(&self) -> i32 {
        // self.creator.clone().unwrap().borrow().generation
        self.creator.as_ref().unwrap().borrow().get_generation()
    }

    /// 値を取得する。
    ///
    /// Return
    /// * Array<f64, IxDyn>: 値
    pub fn get_data(&self) -> Array<V, IxDyn> {
        self.data.clone()
    }

    /// 変数の名前を設定する。
    ///
    /// Arguments
    /// * name (String): 変数名
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name.to_string());
    }

    /// 変数の名前を取得する。
    ///
    /// Return
    /// * String: 名前
    pub fn get_name(&self) -> Option<String> {
        self.name.clone()
    }

    /// 微分値を取得する。逆伝播を実行した場合のみ値が返る。
    ///
    /// Return
    /// * Array<f64, IxDyn>: 微分値
    pub fn get_grad(&self) -> Option<Array<V, IxDyn>> {
        match self.grad.as_ref() {
            Some(grad) => Some(grad.clone()),
            None => None,
        }
    }

    /// 微分値を設定する。
    ///
    /// Arguments
    /// * grad (Array<V, IxDyn)): 微分値
    pub fn set_grad(&mut self, grad: Array<V, IxDyn>) {
        self.grad = Some(grad);
    }

    /// 要素の数
    pub fn get_size(&self) -> usize {
        self.data.len()
    }

    /// 次元ごとの要素数
    pub fn get_shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// 次元数
    pub fn get_ndim(&self) -> usize {
        self.data.ndim()
    }

    /// 型
    pub fn get_dtype(&self) -> String {
        format!("{}", std::any::type_name::<V>())
    }

    /// この変数を出力結果とした場合の逆伝播を行う。
    pub fn backward(&self) {
        let mut creators = FunctionExecutor::extract_creators(vec![Variable::new(self.clone())]);
        // for (_gen, creator) in creators.iter() {
        //     creator.borrow().backward();
        // }

        // 優先度の高い順に関数を取得し、逆伝播を実行する。
        while let Some(creator) = creators.pop() {
            creator.1.borrow().backward();
        }
    }
}

/// RawVariable 構造体を生成するためのトレイト
/// * create_variable: RawVariable 構造体を生成する
pub trait CreateVariable<V: MathOps> {
    fn create_variable(&self) -> RawVariable<V>;
}

/// CreateVariable トレイトの Array<f64, IxDyn> 用の実装
impl<V: MathOps> CreateVariable<V> for Array<V, IxDyn> {
    fn create_variable(&self) -> RawVariable<V> {
        RawVariable {
            data: self.clone(),
            name: None,
            grad: None,
            creator: None,
            generation: 0,
        }
    }
}

/// CreateVariable トレイトの 数値用の実装
impl<V: MathOps> CreateVariable<V> for V {
    fn create_variable(&self) -> RawVariable<V> {
        RawVariable {
            // data: Array::from_elem(IxDyn(&[]), *self),
            data: Array::from_elem(IxDyn(&[]), self.clone()),
            name: None,
            grad: None,
            creator: None,
            generation: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    #[test]
    /// 変数の型名に関するテスト。
    fn test_get_dtype() {
        let var_i8 = RawVariable::new(10i8);
        let var_i16 = RawVariable::new(10i16);
        let var_i32 = RawVariable::new(10i32);
        let var_i64 = RawVariable::new(10i64);
        let var_f32 = RawVariable::new(10.0f32);
        let var_f64 = RawVariable::new(10.0f64);
        let var_u8 = RawVariable::new(10u8);
        let var_u16 = RawVariable::new(10u16);
        let var_u32 = RawVariable::new(10u32);
        let var_u64 = RawVariable::new(10u64);
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
        let var0 = RawVariable::new(1.0);
        assert_eq!(1, var0.get_size());
        let a: [usize; 0] = [];
        assert_eq!(&a, var0.get_shape());
        assert_eq!(0, var0.get_ndim());

        let var1 = RawVariable::from_shape_vec(vec![1], vec![1.0]);

        let sh2x2 = vec![2, 2];
        let val2x2 = vec![1., 2., 3., 4.];
        let var2x2 = RawVariable::from_shape_vec(sh2x2, val2x2);

        assert_eq!(4, var2x2.get_size());
        assert_eq!([2, 2], var2x2.get_shape());
        assert_eq!(2, var2x2.get_ndim());
        dbg!(&var2x2.get_shape());

        let sh10x20x30x40x50 = vec![10, 20, 30, 40, 50];
        let val10x20x30x40x50: Vec<f64> = (1..=12000000).map(|x| x as f64).collect();

        let var10x20x30x40x50 = RawVariable::from_shape_vec(sh10x20x30x40x50, val10x20x30x40x50);
        assert_eq!(12000000, var10x20x30x40x50.get_size());
        assert_eq!([10, 20, 30, 40, 50], var10x20x30x40x50.get_shape());
        assert_eq!(5, var10x20x30x40x50.get_ndim());
    }

    #[test]
    /// 任意の形状に関するテスト。
    fn test_dyndim_array() {
        let shape = vec![2, 2, 2];
        let values = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let var = RawVariable::from_shape_vec(shape, values);
        // dbg!(&var);
        assert_eq!(&[2, 2, 2], var.get_data().shape());
    }

    /// 変数の名前のテスト。
    #[test]
    fn test_variable_name() {
        let mut val = RawVariable::new(Array::from_elem(IxDyn(&[100, 100, 100]), 1.0));

        assert_eq!(None, val.get_name());

        val.set_name("test_val".to_string());
        assert_eq!(Some("test_val".to_string()), val.get_name());
    }
}
