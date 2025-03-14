use core::fmt::Debug;
use ndarray::{ArrayD, IntoDimension, IxDyn, ShapeError};
use rand::TryRngCore;
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::error::Error;
use std::ops::Deref;
use std::rc::{Rc, Weak};

#[derive(Debug, Clone)]

/// Variable 構造体
/// * data (ArrayD<f64>): 変数
/// * name (Option<String>): 変数の名前
/// * grad (Option<ArrayD<f64>): 変数に対応した微分した値。逆伝播によって実際に微分が計算されたときに値を設定する。
/// * creator (Option<Rc<RefCell<FunctionExecutor>>>): この変数を生成した関数
/// * generation (i32): 計算グラフ上の世代
struct Variable<T> {
    data: Option<T>,
    name: Option<String>,
    grad: Option<T>,
    // creator: Option<Rc<RefCell<FunctionExecutor>>>,
    generation: i32,
}

impl Variable<ArrayD<f64>> {
    /// Variable のコンストラクタ。
    /// 以下のように使用する。
    ///   let dim = vec![2, 2, 2];
    ///   let values = vec![1., 2., 3., 4., 5., 6., 7., 8.];
    ///   let variable = Variable::new(dim, values);
    ///
    /// Arguments
    /// * shape (Vec<i32>): 次元
    /// * values (Vec<f64>): 変数
    ///
    /// Returns
    /// * Result<Self, ShapeError>
    fn new<Sh>(shape: Sh, values: Vec<f64>) -> Result<Self, ShapeError>
    where
        Sh: IntoDimension<Dim = IxDyn>,
    {
        let dim = shape.into_dimension();
        let array = ArrayD::from_shape_vec(dim, values)?;
        Ok(Variable {
            data: Some(array),
            name: None,
            grad: None,
            generation: 0,
        })
    }

    /// 変数の名前を設定する。
    ///
    /// Arguments
    /// * name (String): 変数の名前
    fn set_name(&mut self, name: String) {
        self.name = Some(name.to_string());
    }
}

fn type_of<T>(_: T) -> String {
    let a = std::any::type_name::<T>();
    return a.to_string();
}

fn main() {
    let dim = vec![2, 2, 2];
    let values = vec![1., 2., 3., 4., 5., 6., 7., 8.];

    let val = Variable::new(dim, values);
    dbg!(&val);
}
