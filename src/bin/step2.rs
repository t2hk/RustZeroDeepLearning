//! ステップ2 変数を生み出す関数
//!
//! 実行方法
//! ```
//! cargo run --bin [RSファイル]
//! ```

use ndarray::{Array, IxDyn};

/// Variable 構造体
struct Variable {
    data: Array<f64, IxDyn>,
}

impl Variable {
    /// Variable のコンストラクタ。
    ///
    /// # Argumemnts
    /// * data - 変数
    fn new(data: Array<f64, IxDyn>) -> Variable {
        Variable { data }
    }
}

/// Function トレイト
/// Variable を入力し、処理を実行して結果を Variable で返却する。
trait Function {
    /// 関数を実行する。
    /// 関数の実装は Function を継承して行う。
    ///
    /// # Arguments
    /// * input (Variable) - 変数
    ///
    /// # Return
    /// * Variable - 処理結果
    fn call(&self, input: &Variable) -> Variable {
        let x = &input.data;

        let y = self.forward(x);

        let output = Variable::new(y);
        output
    }

    /// 計算を行う。継承して実装すること。
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn>;
}

/// 二乗する。
struct Square;
impl Function for Square {
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        x.mapv(|x| x.powi(2))
    }
}

fn main() {
    // xを定義
    let data1 = Array::from_elem(IxDyn(&[]), 0.5);
    let x = Variable::new(data1);

    println!("x: {}", x.data);

    let f = Square {};
    let y = f.call(&x);
    println!("y: {}", y.data);
}

#[cfg(test)]
/// ゼロの二乗のテスト。
#[test]
fn test_zero_square() {
    let x = Variable::new(Array::from_elem(IxDyn(&[]), 0.0));
    let f = Square {};
    let y = f.call(&x);
    let expected = Array::from_elem(IxDyn(&[]), 0.0);
    assert_eq!(expected, y.data);
}

/// 1の二乗のテスト。
#[test]
fn test_one_square() {
    let x = Variable::new(Array::from_elem(IxDyn(&[]), 1.0));
    let f = Square {};
    let y = f.call(&x);
    let expected = Array::from_elem(IxDyn(&[]), 1.0);
    assert_eq!(expected, y.data);
}

/// 10の二乗のテスト。
#[test]
fn test_ten_square() {
    let x = Variable::new(Array::from_elem(IxDyn(&[]), 10.0));
    let f = Square {};
    let y = f.call(&x);
    let expected = Array::from_elem(IxDyn(&[]), 100.0);
    assert_eq!(expected, y.data);
}

/// 負の値の二乗のテスト。
#[test]
fn test_negative_square() {
    let x = Variable::new(Array::from_elem(IxDyn(&[]), -5.0));
    let f = Square {};
    let y = f.call(&x);
    let expected = Array::from_elem(IxDyn(&[]), 25.0);
    assert_eq!(expected, y.data);
}
