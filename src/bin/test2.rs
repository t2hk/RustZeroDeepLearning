use core::fmt::Debug;
use ndarray::{Array, ArrayD, IntoDimension, IxDyn, ShapeError};
use rand::TryRngCore;
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::error::Error;
use std::ops::Deref;
use std::rc::{Rc, Weak};

/// Variable 構造体
/// * data (ArrayD<f64>): 変数
/// * name (Option<String>): 変数の名前
/// * grad (Option<ArrayD<f64>): 変数に対応した微分した値。逆伝播によって実際に微分が計算されたときに値を設定する。
/// * creator (Option<Rc<RefCell<FunctionExecutor>>>): この変数を生成した関数
/// * generation (i32): 計算グラフ上の世代
#[derive(Debug, Clone)]
struct Variable {
    data: Array<f64, IxDyn>,
    name: Option<String>,
    grad: Option<Array<f64, IxDyn>>,
    //creator: Option<Rc<RefCell<FunctionExecutor>>>,
    generation: i32,
}
impl Variable {
    /// Variable のコンストラクタ。
    ///
    /// # Arguments
    /// * data - 変数    
    fn new<T: CreateVariable>(data: T) -> Variable {
        CreateVariable::create_variable(&data)
    }

    /// Variable を次元と値から生成する。
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
    fn from_shape_vec<Sh>(shape: Sh, values: Vec<f64>) -> Result<Self, ShapeError>
    where
        Sh: IntoDimension<Dim = IxDyn>,
    {
        let dim = shape.into_dimension();
        let array = ArrayD::from_shape_vec(dim, values)?;
        Ok(Variable {
            data: array,
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

/// Variable 構造体を生成するためのトレイト
/// * create_variable: Variable 構造体を生成する
/// Variable 構造体を生成するためのトレイト
/// * create_variable: Variable 構造体を生成する
trait CreateVariable {
    fn create_variable(&self) -> Variable;
}

/// CreateVariable トレイトの Array<f64, IxDyn> 用の実装
impl CreateVariable for Array<f64, IxDyn> {
    fn create_variable(&self) -> Variable {
        Variable {
            data: self.clone(),
            name: None,
            grad: None,
            // creator: None,
            generation: 0,
        }
    }
}

/// CreateVariable トレイトの f64 用の実装
impl CreateVariable for f64 {
    fn create_variable(&self) -> Variable {
        Variable {
            data: Array::from_elem(IxDyn(&[]), *self),
            name: None,
            grad: None,
            //creator: None,
            generation: 0,
        }
    }
}

fn type_of<T>(_: T) -> String {
    let a = std::any::type_name::<T>();
    return a.to_string();
}

fn main() {
    let dim = vec![2, 2, 2];
    let values = vec![1., 2., 3., 4., 5., 6., 7., 8.];

    let val = Variable::from_shape_vec(dim, values);
    dbg!(&val);

    println!("dim: {:?}", val.clone().unwrap().data.dim());
    println!("shape: {:?}", val.clone().unwrap().data.shape());
    // println!("shape: {:?}", val.clone().unwrap().data.unwrap().a);

    let x1 = Rc::new(RefCell::new(Variable::new(1.0)));
    dbg!(&x1);
}
