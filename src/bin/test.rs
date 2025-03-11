use core::fmt::Debug;
use ndarray::{Array, Dimension, IxDyn, ShapeBuilder};
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::error::Error;
use std::rc::{Rc, Weak};

#[derive(Debug, Clone)]
struct Variable<A, D: Dimension> {
    data: Array<A, D>,
    name: Option<String>,
    grad: Option<Array<A, D>>,
    // creator: Option<Rc<RefCell<FunctionExecutor>>>,
    generation: i32,
}

impl<A, D: Dimension> Variable<A, D> {
    /// Variable のコンストラクタ。
    ///
    /// # Arguments
    /// * data - 変数    
    fn new<T: CreateVariable<A, D>>(data: T) -> Variable<A, D> {
        CreateVariable::create_variable(&data)
    }

    /// ベクトルから配列を生成して構造体を構築
    fn from_shape_vec<Sh>(
        shape: Sh,
        data: Vec<A>,
        metadata: Option<String>,
    ) -> Result<Self, Box<dyn Error>>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let arr = Array::from_shape_vec(shape, data)?;
        Ok(Self {
            data: arr,
            name: None,
            grad: None,
            // creator: None,
            generation: 0,
        })
    }
}

/// Variable 構造体を生成するためのトレイト
/// * create_variable: Variable 構造体を生成する
trait CreateVariable<A, D: Dimension> {
    fn create_variable(&self) -> Variable<A, D>;
}

fn main() {
    let vector_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix = Variable::<f64, ndarray::Ix2>::from_shape_vec(
        (2, 3), // 2x3の2次元配列
        vector_data,
        Some("サンプルデータ".to_string()),
    )
    .unwrap();
    dbg!(&matrix);

    let arr = Array::from_elem(IxDyn(&[]), 2.0);
    let var = Variable::<i32, IxDyn>::from_shape_vec(IxDyn(&[]), vec![2], Some("hoge".to_string()))
        .unwrap();
    dbg!(&var);
}
