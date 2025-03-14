use core::fmt::Debug;
use ndarray::{Array, Array2, DataOwned, Dimension, IxDyn, ShapeBuilder};
use num::Num;
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
    A: Clone + 'static,
    D: Dimension,
{
}

// 数値型用トレイトの定義
trait CreateVariable<A, D: Dimension> {
    fn create_variable<Sh>(value: A, shape: Sh) -> Result<Variable<A, D>, Box<dyn Error>>
    where
        Sh: ShapeBuilder<Dim = D>;
}

// 全数値型に対する実装
impl<D> CreateVariable<f64, D> for Variable<f64, D>
where
    D: Dimension,
{
    fn create_variable<Sh>(value: f64, shape: Sh) -> Result<Variable<f64, D>, Box<dyn Error>>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Ok(Variable {
            data: Array::from_shape_vec(shape, vec![value])?,
            name: None,
            grad: None,
            generation: 0,
        })
    }
}

impl<D> CreateVariable<Vec<f64>, D> for Variable<Vec<f64>, D>
where
    D: Dimension,
{
    fn create_variable<Sh>(
        values: Vec<f64>,
        shape: Sh,
    ) -> Result<Variable<Vec<f64>, D>, Box<dyn Error>>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Ok(Variable {
            data: Array::from_shape_vec(shape, vec![values])?,
            name: None,
            grad: None,
            generation: 0,
        })
    }
}

#[derive(Debug, Clone)]
struct Variable2<T> {
    data: Option<T>,
}

impl<T> Variable2<T> {
    fn new() -> Variable2<T> {
        Variable2 { data: None }
    }
}

// f64型の2x2配列に特化した実装
impl Variable2<Array2<f64>> {
    fn new_2x2() -> Variable2<Array2<f64>> {
        Variable2 {
            data: Some(Array2::from_shape_vec((2, 2), vec![1., 2., 3., 4.]).unwrap()),
        }
    }

    fn new2(shape: (usize, usize), values: Vec<f64>) -> Variable2<Array2<f64>> {
        Variable2 {
            data: Some(Array2::from_shape_vec(shape, values).unwrap()),
        }
    }
}

impl Variable2<Array<, IxDyn>> {
    fn new3<Sh, S, A>(shape: Sh, values: Vec<A>) -> Variable2<Array<A, IxDyn>>
    where
        Sh: ShapeBuilder<Dim = IxDyn>,
        S: DataOwned<Elem = A>,
    {
        Variable2 {
            data: Array::from_shape_vec(shape, values).ok(),
        }
    }
}

fn main() {
    let vector_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let generic_var: Variable2<i32> = Variable2::new();
    let array_var: Variable2<Array2<f64>> = Variable2::new_2x2();
    let array_var2: Variable2<Array2<f64>> = Variable2::new2((2, 3), vec![1., 2., 3., 4., 5., 6.]);
    let array_var3: Variable2<Array2<f64>> =
        Variable2::new3((2, 2, 2), vec![1., 2., 3., 4., 5., 6., 7., 8.]);

    dbg!(&generic_var);
    dbg!(&array_var2);
    dbg!(&array_var3);

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

    // let vector1 = 5.0.create_variable((3, 3, 3)); // 3要素のベクトル
    // let vector2 = 3.0.create_variable((3,)); // 3要素のベクトル
    // let matrix = 2.0.create_variable((2, 2)); // 2x2行列
    // let matrix2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].create_variable((2, 2, 2));

    // dbg!(&vector1);
    // dbg!(&vector2);
    // dbg!(&matrix);
    // dbg!(&matrix2);

    // let val1 = Variable::new(1.0);
    // dbg!(&val1);
    // let val2 = Variable::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]); //.with_shape((2, 2, 2));
    // dbg!(&val2);

    //let val1 = Variable::new();
}
