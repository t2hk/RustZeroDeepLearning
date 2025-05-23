// ライブラリを一括でインポート
use crate::modules::*;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, ArrayD, IntoDimension, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

// #[derive(Debug, Clone)]
// pub enum DataType<V: MathOps> {
//     Variable(Variable<V>),
//     Parameter(Parameter<V>),
// }

/// RawData 構造体
/// 変数自体、および逆伝播に必要な情報を保持する。
///
/// * data (Array<f64, IxDyn>): 変数
/// * name (Option<String>): 変数の名前
/// * grad (Option<Array<f64, IxDyn>): 変数に対応した微分した値。逆伝播によって実際に微分が計算されたときに値を設定する。
/// * creator (Option<Rc<RefCell<FunctionExecutor>>>): この変数を生成した関数
/// * generation (i32): 計算グラフ上の世代
#[derive(Debug, Clone)]
pub struct RawData<V: MathOps> {
    data: Array<V, IxDyn>,
    name: Option<String>,
    grad: Option<Variable<V>>,
    creator: Option<Rc<RefCell<FunctionExecutor<V>>>>,
    generation: i32,
}

impl<V: MathOps> RawData<V> {
    /// RawData のコンストラクタ。
    ///
    /// # Arguments
    /// * data - 変数    
    pub fn new<T: CreateRawData<V>>(data: T) -> RawData<V> {
        CreateRawData::create_raw_data(&data)
    }

    /// RawData を次元と値から生成する。
    /// 以下のように使用する。
    ///   let dim = vec![2, 2, 2];
    ///   let values = vec![1., 2., 3., 4., 5., 6., 7., 8.];
    ///   let variable = RawData::new(dim, values);
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

    /// RawData をベクトルから生成する。
    ///
    /// Arguments
    /// * values (Vec<V>): 変数
    ///
    /// Returns
    /// * Self
    pub fn from_vec(values: Vec<V>) -> Self {
        let array = Array::from_vec(values).into_dyn();
        Self {
            data: array,
            name: None,
            grad: None,
            creator: None,
            generation: 0,
        }
    }

    /// 要素数による等差数列を生成する。
    ///
    /// Arguments:
    /// * start (V): 開始値
    /// * end (V): 終了値
    /// * n (usize): 要素数
    /// Return:
    /// * Self: 指定した開始値から終了値までの要素数を持つ等差数列の RawData
    pub fn linspace(start: V, end: V, n: usize) -> Self {
        let base = Array::linspace(start.to_f64().unwrap(), end.to_f64().unwrap(), n);
        let shape = base.shape();
        let values = base.to_vec().iter().map(|x| V::from(*x).unwrap()).collect();

        let array = Array::from_shape_vec(IxDyn(shape), values).unwrap();

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
        // debug!(
        //     "[set_creator] {:?} in:{:?}",
        //     &creator.borrow().get_creator().borrow().get_name(),
        //     &creator.borrow().get_inputs()[0].borrow().get_data()
        // );

        self.creator = Some(Rc::clone(&creator));
        self.generation = creator.borrow().get_generation() + 1;
    }

    /// この変数を算出した関数を取得する。
    ///
    /// Return
    /// * Option<Rc<RefCell<FunctionExecutor<V>>>>: 関数
    pub fn get_creator(&self) -> Option<Rc<RefCell<FunctionExecutor<V>>>> {
        if let Some(creator) = self.creator.clone() {
            Some(Rc::clone(&creator.clone()))
        } else {
            None
        }
    }

    /// 微分をリセットする。
    pub fn clear_grad(&mut self) {
        self.grad = None;
    }

    /// 変数の世代を取得する。
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

    pub fn set_data(&mut self, data: Array<V, IxDyn>) {
        self.data = data;
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
    /// * Variable<V>: 微分値
    pub fn get_grad(&self) -> Option<Variable<V>> {
        match self.grad.as_ref() {
            Some(grad) => Some(grad.clone()),
            None => None,
        }
    }

    /// 微分値を設定する。
    ///
    /// Arguments
    /// * grad (Variable<V>): 微分値
    pub fn set_grad(&mut self, grad: Variable<V>) {
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
        debug!(
            "[backward] self name: {}",
            self.get_name().unwrap_or("none".to_string())
        );
        let mut creators = FunctionExecutor::extract_creators(vec![Variable::new(self.clone())]);

        // 優先度の高い順に関数を取得し、逆伝播を実行する。
        while let Some(creator) = creators.pop() {
            // debug!("{:?}", creator.1.borrow().detail());
            creator.1.borrow().backward();
        }
    }
}

/// RawData 構造体を生成するためのトレイト
/// * create_raw_data: RawData 構造体を生成する
pub trait CreateRawData<V: MathOps> {
    fn create_raw_data(&self) -> RawData<V>;
}

/// CreateRawData トレイトの Array<f64, IxDyn> 用の実装
impl<V: MathOps> CreateRawData<V> for Array<V, IxDyn> {
    fn create_raw_data(&self) -> RawData<V> {
        RawData {
            data: self.clone(),
            name: None,
            grad: None,
            creator: None,
            generation: 0,
        }
    }
}

/// CreateRawData トレイトの 数値用の実装
impl<V: MathOps> CreateRawData<V> for V {
    fn create_raw_data(&self) -> RawData<V> {
        RawData {
            // data: Array::from_elem(IxDyn(&[]), *self),
            data: Array::from_elem(IxDyn(&[]), self.clone()),
            name: None,
            grad: None,
            creator: None,
            generation: 0,
        }
    }
}

/// RawData に関する処理トレート
pub trait RawDataProcessor<V: MathOps> {
    /// コンストラクタ
    ///
    /// Arguments:
    /// * raw (RawData<V>): ラップする RawData
    ///
    /// Return:
    /// * Self: RawData をラップしたインスタンス
    fn new(raw: RawData<V>) -> Self;

    /// Rc、RefCell による参照の共有や内部可変に対応した RawData を取得する。
    fn raw(&self) -> Rc<RefCell<RawData<V>>>;

    // /// RawData の borrow を返す。
    // fn borrow(&self) -> Ref<RawData<V>>;

    // /// RawData の borrow_mut を返す。
    // fn borrow_mut(&self) -> RefMut<RawData<V>>;

    /// RawData の as_ref を返す。
    fn as_ref(&self) -> &RefCell<RawData<V>>;

    /// 変数を取得する。
    ///
    /// Return
    /// * Array<V, IxDyn>: 変数
    fn get_data(&self) -> Array<V, IxDyn> {
        self.raw().borrow().get_data()
    }

    /// 変数の名前を設定する。
    ///
    /// Arguments
    /// * name (String): 変数名
    fn set_name(&self, name: String) {
        self.raw().borrow_mut().set_name(name);
    }

    /// 変数の名前を取得する。
    ///
    /// Return
    /// * String: 名前
    fn get_name(&self) -> Option<String> {
        self.raw().borrow().get_name()
    }
    /// 微分値を設定する。
    ///
    /// Arguments
    /// * grad (Variable<V>): 微分値
    fn set_grad(&self, grad: Variable<V>) {
        self.raw().borrow_mut().set_grad(grad);
    }

    /// 微分値(勾配)を取得する。
    /// 逆伝播を実行済みの場合のみ値が存在する。
    ///
    /// Return
    /// * Option<Variable<V>>: 微分値(勾配)
    fn get_grad(&self) -> Option<Variable<V>> {
        self.raw().borrow().get_grad()
    }

    /// 微分値を消去する。    
    fn clear_grad(&self) {
        self.raw().borrow_mut().clear_grad();
    }

    /// 変数の世代を取得する。
    ///
    /// Return
    /// i32: 世代
    fn get_generation(&self) -> i32 {
        self.raw().borrow().get_generation()
    }

    /// この変数を生成した関数を設定する。
    ///
    /// Arguments
    /// * creator (Rc<RefCell<FunctionExecutor>>): 関数のラッパー
    fn set_creator(&self, creator: Rc<RefCell<FunctionExecutor<V>>>) {
        self.raw().borrow_mut().set_creator(creator);
    }

    /// この変数を算出した関数を取得する。
    /// creator が設定されていない場合は None を返す。
    ///
    /// Return
    /// * Option<Rc<RefCell<FunctionExecutor<V>>>>: 関数
    fn get_creator(&self) -> Option<Rc<RefCell<FunctionExecutor<V>>>> {
        self.raw().borrow().get_creator()
    }

    /// この変数を生成した関数の世代を取得する。
    /// creator が設定されていない場合は None を返す。
    ///
    /// Return
    /// * Option<i32>: 関数の世代
    fn get_creator_generation(&self) -> Option<i32> {
        if let Some(creator) = self.raw().borrow().get_creator() {
            Some(creator.borrow().get_generation())
        } else {
            None
        }
    }

    /// 変数を設定する。
    ///
    /// Arguments:
    /// * data (Array<V, IxDyn>): 変数
    fn set_data(&self, data: Array<V, IxDyn>) {
        self.raw().borrow_mut().set_data(data);
    }

    /// 逆伝播を実行する。
    fn backward(&self) {
        self.raw().as_ref().clone().borrow().backward();
    }

    // /// リシェイプ
    // ///
    // /// Arguments:
    // /// * shape (Vec<usize>): 変更後の形状
    // ///
    // /// Return:
    // /// * Variable<RawData<V>>
    // fn reshape(&self, shape: Vec<usize>) -> Self;

    // /// 転値
    // ///
    // /// Return:
    // /// * Variable<RawData<V>>: 転値結果
    // fn transpose(&self) -> Self;

    /// 詳細出力
    fn detail(&self) -> String {
        let data_detail = format!("data: {:?}", self.raw().borrow().get_data());
        let grad_detail = format!(
            "grad: {:?}",
            match self.raw().borrow().get_grad() {
                // Some(grad) => format!("{:?}", grad.borrow().get_data()),
                Some(grad) => format!("{:?}", grad.get_data()),
                None => "None".to_string(),
            }
        );
        let detail = format!(
            "name: {},generation: {}, data:{:?}, grad: {:?}",
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
}

#[cfg(test)]
mod tests {
    use super::*;
    // use rand::prelude::*;
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
        assert_eq!(vec![6], r1.get_data().shape().to_vec());
        assert_eq!(vec![1, 2, 3, 4, 5, 6], r1.get_data().flatten().to_vec());

        let r2 = r1.reshape(vec![3, 2]);
        assert_eq!(vec![3, 2], r2.get_data().shape().to_vec());
        assert_eq!(vec![1, 2, 3, 4, 5, 6], r2.get_data().flatten().to_vec());
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
        assert_eq!(vec![3, 2], y.get_data().shape().to_vec());
        assert_eq!(vec![1, 4, 2, 5, 3, 6], y.get_data().flatten().to_vec());

        y.backward();

        let x_grad = x.get_grad().unwrap();
        dbg!(&x_grad);

        // 微分値が入力値と同じ形状で、全て1であることを確認する。
        assert_eq!(input_shape, x_grad.clone().get_data().shape().to_vec());
        assert_eq!(vec![1, 1, 1, 1, 1, 1], x_grad.get_data().flatten().to_vec());
    }
}
