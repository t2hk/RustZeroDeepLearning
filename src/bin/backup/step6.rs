//! ステップ6 手作業によるバックプロパゲーション
//!
//! 実行方法
//! ```
//! cargo run --bin [RSファイル]
//! ```

use ndarray::{Array, IxDyn};

/// Variable 構造体
/// * data (Array<f64, IxDyn>): 変数
/// * grad (Option<Array<f64, IxDyn>): 変数に対応した微分した値。逆伝播によって実際に微分が計算されたときに値を設定する。
struct Variable {
    data: Array<f64, IxDyn>,
    grad: Option<Array<f64, IxDyn>>,
}

impl Variable {
    /// Variable のコンストラクタ。
    ///
    /// # Arguments
    /// * data - 変数
    /// * grad - data を微分した値
    fn new(data: Array<f64, IxDyn>) -> Variable {
        Variable {
            data: data,
            grad: None,
        }
    }
}

/// Function トレイト
/// Variable を入力し、処理を実行して結果を Variable で返却する。
trait Function<'a> {
    /// 関数を実行する。
    /// 関数の実装は Function を継承して行う。
    ///
    /// # Arguments
    /// * input (Variable) - 変数
    ///
    /// # Return
    /// * Variable - 処理結果
    fn call(&mut self, input: &'a Variable) -> Variable {
        let x = &input.data;
        let y = self.forward(x);

        let output = Variable::new(y);
        self.keep_input(input);
        output
    }

    fn keep_input(&mut self, input: &'a Variable);

    /// 通常の計算を行う順伝播。継承して実装すること。
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn>;

    /// 微分の計算を行う逆伝播。継承して実装すること。
    fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn>;
}

/// 二乗する。
struct Square<'a> {
    input: Option<&'a Variable>,
}
impl<'a> Function<'a> for Square<'a> {
    /// 順伝播
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        x.mapv(|x| x.powi(2))
    }

    fn keep_input(&mut self, input: &'a Variable) {
        self.input = Some(input);
    }

    /// 逆伝播
    /// y=x^2 の微分であるため、dy/dx=2x である。
    fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let x = &self.input.unwrap().data;
        let gx = 2.0 * x * gy;
        gx
    }
}

/// Exp 関数
struct Exp<'a> {
    input: Option<&'a Variable>,
}
impl<'a> Function<'a> for Exp<'a> {
    // Exp (y=e^x) の順伝播
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let e = std::f64::consts::E;
        x.mapv(|x| e.powf(x))
    }

    fn keep_input(&mut self, input: &'a Variable) {
        self.input = Some(input);
    }

    /// 逆伝播
    /// dy/dx=e^x である。
    fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let e = std::f64::consts::E;
        let x = &self.input.unwrap().data;
        let gx = x.mapv(|x| e.powf(x)) * gy;
        gx
    }
}

fn main() {
    // 順伝播
    // x -> Square -> a -> Exp -> b -> Square -> y
    let mut a_square = Square { input: None };
    let mut b_exp = Exp { input: None };
    let mut c_square = Square { input: None };

    let mut x = Variable::new(Array::from_elem(IxDyn(&[]), 0.5));
    println!("x: {}", x.data);
    let mut a = a_square.call(&x);
    println!("a: {}", a.data);
    let mut b = b_exp.call(&a);
    println!("b: {}", b.data);
    let mut y = c_square.call(&b);
    println!("y: {}", y.data);

    // 逆伝播
    //x.grad <- a.backword <- a.grad <- b.backward <- b.grad <- c.backward <- y.grad(=1.0)
    y.grad = Some(Array::from_elem(IxDyn(&[]), 1.0));
    b.grad = Some(c_square.backward(&y.grad.unwrap()));
    a.grad = Some(b_exp.backward(&b.grad.unwrap()));
    x.grad = Some(a_square.backward(&a.grad.unwrap()));

    println!("x.grad: {}", x.grad.unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ゼロの二乗のテスト。
    #[test]
    fn test_zero_square() {
        let x = Variable::new(Array::from_elem(IxDyn(&[]), 0.0));
        let mut f = Square { input: None };
        let y = f.call(&x);
        let expected = Array::from_elem(IxDyn(&[]), 0.0);
        assert_eq!(expected, y.data);
    }

    /// 1の二乗のテスト。
    #[test]
    fn test_one_square() {
        let x = Variable::new(Array::from_elem(IxDyn(&[]), 1.0));
        let mut f = Square { input: None };
        let y = f.call(&x);
        let expected = Array::from_elem(IxDyn(&[]), 1.0);
        assert_eq!(expected, y.data);
    }

    /// 10の二乗のテスト。
    #[test]
    fn test_ten_square() {
        let x = Variable::new(Array::from_elem(IxDyn(&[]), 10.0));
        let mut f = Square { input: None };
        let y = f.call(&x);
        let expected = Array::from_elem(IxDyn(&[]), 100.0);
        assert_eq!(expected, y.data);
    }

    /// 負の値の二乗のテスト。
    #[test]
    fn test_negative_square() {
        let x = Variable::new(Array::from_elem(IxDyn(&[]), -5.0));
        let mut f = Square { input: None };
        let y = f.call(&x);
        let expected = Array::from_elem(IxDyn(&[]), 25.0);
        assert_eq!(expected, y.data);
    }

    /// Exp 関数のテスト。
    #[test]
    fn test_exp_2() {
        let x = Variable::new(Array::from_elem(IxDyn(&[]), 2.0));
        let mut f = Exp { input: None };
        let y = f.call(&x);
        let e = std::f64::consts::E;
        let expected = Array::from_elem(IxDyn(&[]), e.powf(2.0));

        assert_eq!(expected, y.data);
    }
}
