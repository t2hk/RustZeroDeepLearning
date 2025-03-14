use core::fmt::Debug;
use ndarray::{Array, Dimension, IxDyn, ShapeBuilder};
use num;
use rand::TryRngCore;
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::error::Error;
use std::ops::Deref;
use std::rc::{Rc, Weak};

#[derive(Debug, Clone)]
struct Variable<A, D: Dimension> {
    data: Array<A, D>,
    name: Option<String>,
    grad: Option<Array<A, D>>,
    // creator: Option<Rc<RefCell<FunctionExecutor>>>,
    generation: i32,
}

impl<A, D> Variable<A, D>
where
    D: Dimension,
{
    fn new(shape: D) -> Self
    where
        A: Default,
    {
        Variable {
            data: Array::default(shape),
            name: None,
            grad: None,
            generation: 0,
        }
    }

    /// Variable のコンストラクタ。
    ///
    /// # Arguments
    /// * data - 変数    
    // fn new(&self, num: f64) -> Result<Variable<A, D>, Box<dyn Error>> {
    //     num.create_variable((1,))
    // }
}

// 数値型用トレイトの定義
trait CreateVariable<D: Dimension> {
    fn create_variable<Sh>(&self, shape: Sh) -> Result<Variable<f64, D>, Box<dyn Error>>
    where
        Sh: ShapeBuilder<Dim = D>;
}


impl<D: Dimension> CreateVariable<D> for f64 {
    fn create_variable<Sh>(&self, shape: Sh) -> Result<Variable<f64, D>, Box<dyn Error>>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Ok(Variable {
            data: Array::from_elem(shape, self.clone()),
            name: None,
            grad: None,
            generation: 0,
        })
    }
}

impl<D: Dimension> CreateVariable<D> for Vec<f64> {
    fn create_variable<Sh>(&self, shape: Sh) -> Result<Variable<f64, D>, Box<dyn Error>>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Ok(Variable {
            data: Array::from_shape_vec(shape, self.clone())?,
            name: None,
            grad: None,
            generation: 0,
        })
    }
}

fn type_of<T>(_: T) -> String {
    let a = std::any::type_name::<T>();
    return a.to_string();
}

fn main() {
    let vector_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // let matrix = Variable::<f64, ndarray::Ix2>::from_shape_vec(
    //     (2, 3), // 2x3の2次元配列
    //     vector_data,
    //     Some("サンプルデータ".to_string()),
    // )
    // .unwrap();
    // dbg!(&matrix);

    //    let arr = Array::from_elem(IxDyn(&[]), 2.0);
    // //let var = Variable::<i32, IxDyn>::from_shape_vec(IxDyn(&[]), vec![2], Some("hoge".to_string()))
    //     .unwrap();
    // dbg!(&var);

    // let arr_val = Variable::new(&arr);

    // println!("{}", type_of(&arr));
    // println!("{}", type_of(&var));

    let vector1 = 5.0.create_variable((3, 3, 3)); // 3要素のベクトル
    let vector2 = 3.0.create_variable((3,)); // 3要素のベクトル
    let matrix = 1.0.create_variable((2, 2)); // 2x2行列
    let matrix2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].create_variable((2, 2, 2));

    dbg!(&vector1);
    dbg!(&vector2);
    dbg!(&matrix);
    dbg!(&matrix2);

    let val1 = Variable::new();
}
