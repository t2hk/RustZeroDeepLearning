/// ステップ1 箱としての変数
use ndarray::{Array, IxDyn};

/// Variable 構造体
struct Variable {
  data: Array<f64, IxDyn>,
}

impl Variable {
  fn new(data: Array<f64, IxDyn>) -> Variable {
      Variable { data }
  }
}

fn main() {
  // xを定義
  let data1 = Array::from_elem(IxDyn(&[]), 0.5);
  let data2 = Array::from_elem(IxDyn(&[]), 1.0);
  let x = Variable::new(data1);
  let y = Variable::new(data2);

  println!("x: {}", x.data);
  println!("y: {}", y.data);
}