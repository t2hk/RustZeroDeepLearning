//! ステップ7 バックプロパゲーションの自動化
//!
//! 実行方法
//! ```
//! cargo run --bin [RSファイル]
//! ```

use ndarray::{Array, IxDyn};

/// Variable 構造体
/// * data (Array<f64, IxDyn>): 変数
/// * grad (Option<Array<f64, IxDyn>): 変数に対応した微分した値。逆伝播によって実際に微分が計算されたときに値を設定する。
/// * creator (Option<>)
struct Variable<'a, F> {
    data: Array<f64, IxDyn>,
    grad: Option<Array<f64, IxDyn>>,
    creator: Option<&'a F>,
}

impl<'a, F> Variable<'a, F> {
    /// Variable のコンストラクタ。
    ///
    /// # Arguments
    /// * data - 変数
    /// * grad - data を微分した値
    /// * creator - 生成元の関数
    fn new(data: Array<f64, IxDyn>) -> Variable<'a, F> {
        Variable {
            data: data,
            grad: None,
            creator: None,
        }
    }

    /// この変数を生成した関数を設定する。
    ///
    /// # Arguments
    /// * creator - 生成元の関数
    fn set_creator(&mut self, creator: &'a F) {
        self.creator = Some(creator);
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
    fn call(&mut self, input: &'a Variable<'a, Self>) -> Variable<'a, Self>
    where
        Self: Sized,
    {
        let x = &input.data;
        let y = self.forward(x);

        let mut output = Variable::new(y);
        self.keep_input(input);
        output.set_creator(self); // 出力変数に生成元の関数を設定する。

        self.keep_output(&output);

        output
    }

    fn keep_input(&mut self, input: &'a Variable<'a, Self>);
    fn keep_output(&mut self, output: &'a Variable<'a, Self>);

    /// 通常の計算を行う順伝播。継承して実装すること。
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn>;

    /// 微分の計算を行う逆伝播。継承して実装すること。
    fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn>;
}

/// 二乗する。
struct Square<'a> {
    input: Option<&'a Variable<'a, Square<'a>>>,
    output: Option<&'a Variable<'a, Square<'a>>>,
}
impl<'a> Function<'a> for Square<'a> {
    /// 順伝播
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        x.mapv(|x| x.powi(2))
    }

    fn keep_input(&mut self, input: &'a Variable<'a, Square<'a>>) {
        self.input = Some(input);
    }

    fn keep_output(&mut self, output: &'a Variable<'a, Square<'a>>) {
        self.output = Some(output);
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
    input: Option<&'a Variable<'a, Exp<'a>>>,
    output: Option<&'a Variable<'a, Exp<'a>>>,
}
impl<'a> Function<'a> for Exp<'a> {
    // Exp (y=e^x) の順伝播
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let e = std::f64::consts::E;
        x.mapv(|x| e.powf(x))
    }

    fn keep_input(&mut self, input: &'a Variable<'a, Exp<'a>>) {
        self.input = Some(input);
    }

    fn keep_output(&mut self, output: &'a Variable<'a, Exp<'a>>) {
        self.output = Some(output);
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
    todo!();
    // // 順伝播
    // // x -> Square -> a -> Exp -> b -> Square -> y
    // let mut a_square = Square { input: None };
    // let mut b_exp = Exp { input: None };
    // let mut c_square = Square { input: None };

    // let mut x = Variable::new(Array::from_elem(IxDyn(&[]), 0.5));
    // println!("x: {}", x.data);
    // let mut a = a_square.call(&x);
    // println!("a: {}", a.data);
    // let mut b = b_exp.call(&a);
    // println!("b: {}", b.data);
    // let mut y = c_square.call(&b);
    // println!("y: {}", y.data);

    // // 逆伝播
    // //x.grad <- a.backword <- a.grad <- b.backward <- b.grad <- c.backward <- y.grad(=1.0)
    // y.grad = Some(Array::from_elem(IxDyn(&[]), 1.0));
    // b.grad = Some(c_square.backward(&y.grad.unwrap()));
    // a.grad = Some(b_exp.backward(&b.grad.unwrap()));
    // x.grad = Some(a_square.backward(&a.grad.unwrap()));

    // println!("x.grad: {}", x.grad.unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_todo() {
        todo!();
    }
}
