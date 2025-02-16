//! ステップ4 数値微分
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

/// Exp 関数
struct Exp;
impl Function for Exp {
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let e = std::f64::consts::E;
        x.mapv(|x| e.powf(x))
    }
}

/// 中心差分近似による数値微分
///
/// {f(x+h) - f(x-h)} / (2h)
///
/// Arguments:
/// * f (Function): 微分対象の関数
/// * x (Variable): 微分を求める変数
/// * eps (f64): 微小値 (1e-4)
///
/// Return
/// Variable: 中心差分近似による微分結果
fn numerical_diff<F: Function>(f: F, x: Variable, eps: f64) -> Variable {
    let eps_array = Array::from_elem(IxDyn(&[]), eps);

    let x0 = Variable::new(&x.data - &eps_array);
    let x1 = Variable::new(&x.data + &eps_array);
    let y0 = f.call(&x0);
    let y1 = f.call(&x1);

    Variable::new((&y1.data - &y0.data) / (&eps_array * 2.0))
}

/// ステップ4 合成関数
struct CompositionCunctionStep4;
impl Function for CompositionCunctionStep4 {
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let a = Square {};
        let b = Exp {};
        let c = Square {};

        let val_x = Variable::new(x.clone());
        let a = a.call(&val_x);
        let b = b.call(&a);
        let y = c.call(&b);
        y.data
    }
}

fn main() {
    // ステップ2 二乗の計算
    let x_step2 = Variable::new(Array::from_elem(IxDyn(&[]), 0.5));
    let f_step2 = Square {};
    let y_step2 = f_step2.call(&x_step2);
    println!("[step2] x: {} y: {}", x_step2.data, y_step2.data);

    // ステップ3 関数の連結
    let a_square_step3 = Square {};
    let b_exp_step3 = Exp {};
    let c_square_step3 = Square {};

    let x_step3 = Variable::new(Array::from_elem(IxDyn(&[]), 0.5));
    let a_step3 = a_square_step3.call(&x_step3);
    let b_step3 = b_exp_step3.call(&a_step3);
    let y_step3 = c_square_step3.call(&b_step3);

    println!("[step3] x: {} y: {}", x_step3.data, y_step3.data);

    // ステップ4 数値微分
    let f_step4 = Square {};
    let x_step4 = Variable::new(Array::from_elem(IxDyn(&[]), 2.0));
    let dy = numerical_diff(f_step4, x_step4, 1e-4);

    println!("[step4] dy: {}", dy.data);

    // ステップ4 合成関数の微分
    let x_step4 = Variable::new(Array::from_elem(IxDyn(&[]), 0.5));
    let dy_step4 = numerical_diff(CompositionCunctionStep4, x_step4, 1e-4);
    println!("[step4] dy: {}", dy_step4.data);
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Exp 関数のテスト。
    #[test]
    fn test_exp_2() {
        let x = Variable::new(Array::from_elem(IxDyn(&[]), 2.0));
        let f = Exp {};
        let y = f.call(&x);
        // let expected = Array::from_elem(IxDyn(&[]), 7.38905609893065);
        let e = std::f64::consts::E;
        // let result = e.powi(2);
        let expected = Array::from_elem(IxDyn(&[]), e.powi(2));

        assert_eq!(expected, y.data);
    }

    #[test]
    fn test_num_diff() {
        let f = Square {};
        let x = Variable::new(Array::from_elem(IxDyn(&[]), 2.0));
        let dy = numerical_diff(f, x, 1e-4);
        assert_eq!(Array::from_elem(IxDyn(&[]), 4.000000000004), dy.data);
    }
}
