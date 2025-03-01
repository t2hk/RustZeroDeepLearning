//! ステップ7~10までの実装
//!   * ステップ7  バックプロパゲーションの自動化
//!   * ステップ8  再帰からループへ
//!   * ステップ9  関数をより便利に
//!   * ステップ10 テストを行う
//!
//! 書籍では関数とそのの出力結果をリンクさせ、出力から関数へのリンクを辿って逆伝播を自動化している (リンクノード)。
//! Rust で同様の仕組みを実現するのが難しく、関数をベクタで保持して繰り返し処理する方法で実現してみた。
//! 課題:
//!   * Rust でリンクノードを実現する方法
//!   * 参照で解決できそうなところも clone を使ってしまっている

use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// Variable 構造体
/// * data (Array<f64, IxDyn>): 変数
/// * grad (Option<Array<f64, IxDyn>): 変数に対応した微分した値。逆伝播によって実際に微分が計算されたときに値を設定する。
/// * creator (Option<Weak<RefCell<dyn Function>>>): 生成元の関数
#[derive(Debug, Clone)]
struct Variable {
    data: Array<f64, IxDyn>,
    grad: Option<Array<f64, IxDyn>>,
}

impl Variable {
    /// Variable のコンストラクタ。
    ///
    /// # Arguments
    /// * data - 変数
    fn new(data: Array<f64, IxDyn>) -> Variable {
        Variable { data, grad: None }
    }
}

/// 関数の入出力値を持つ構造体
/// * input (Option<Rc<RefCell<Variable>>>): 入力値
/// * output (Option<Rc<RefCell<Variable>>>): 出力値
#[derive(Debug, Clone)]
struct FunctionParameters {
    input: Option<Rc<RefCell<Variable>>>,
    output: Option<Rc<RefCell<Variable>>>,
}
impl FunctionParameters {
    /// コンストラクタ
    /// 入力値、出力値を設定する。
    ///
    /// Arguments
    /// * input (Rc<RefCell<Variable>>): 入力値
    /// * output (Rc<RefCell<Variable>>): 出力値
    fn new(input: Rc<RefCell<Variable>>, output: Rc<RefCell<Variable>>) -> FunctionParameters {
        FunctionParameters {
            // input: Some(input.clone()),
            // output: Some(output.clone()),
            input: Some(input),
            output: Some(output),
        }
    }

    /// 逆伝播で算出した微分値を設定する。
    ///
    /// Arguments
    /// * grad (Array<f64, IxDyn>): 微分値
    fn set_input_grad(&mut self, grad: Array<f64, IxDyn>) {
        self.input.as_mut().unwrap().borrow_mut().grad = Some(grad);
    }

    /// 関数の入力値を取得する。
    ///
    /// Return
    /// * Option<Rc<RefCell<Variable>>>: 入力値
    fn get_input(&self) -> Option<Rc<RefCell<Variable>>> {
        if let Some(input) = &self.input {
            Some(Rc::clone(input))
        } else {
            None
        }
    }

    /// 関数の出力値を取得する。
    ///
    /// Return
    /// * Option<Rc<RefCell<Variable>>>: 出力値
    fn get_output(&self) -> Option<Rc<RefCell<Variable>>> {
        if let Some(output) = &self.output {
            Some(Rc::clone(output))
        } else {
            None
        }
    }
}

trait Function: std::fmt::Debug {
    fn set_parameters(&mut self, input: Rc<RefCell<Variable>>, output: Rc<RefCell<Variable>>);
    fn get_parameters(&self) -> Option<Rc<RefCell<FunctionParameters>>>;

    /// 順伝播
    /// 通常の計算を行う順伝播。継承して実装すること。
    ///
    /// Arguments
    /// * x (Array<f64, IxDyn>): 入力値
    ///
    /// Returns
    /// * Array<f64, IxDyn>: 出力値
    // fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn>;
    fn forward(&self, x: Array<f64, IxDyn>) -> Array<f64, IxDyn>;

    /// 微分の計算を行う逆伝播。
    /// 継承して実装すること。
    ///
    /// Arguments
    /// * gy (Array<f64, IxDyn>): 出力値に対する微分値
    ///
    /// Returns
    /// * Array<f64, IxDyn>: 入力値に対する微分値
    fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn>;

    fn call(&mut self, input: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
        // 関数を実行する。
        let y = Rc::new(RefCell::new(Variable::new(
            self.forward(Rc::clone(&input).borrow().data.clone()),
        )));

        // 関数の入力、出力を設定する。
        self.set_parameters(input, y.clone());

        y
    }
}

#[derive(Debug, Clone)]
struct Square {
    parameters: Option<Rc<RefCell<FunctionParameters>>>,
}

impl Function for Square {
    // 関数の入出力値のセッター
    fn set_parameters(&mut self, input: Rc<RefCell<Variable>>, output: Rc<RefCell<Variable>>) {
        self.parameters = Some(Rc::new(RefCell::new(FunctionParameters::new(
            input, output,
        ))));
    }
    fn get_parameters(&self) -> Option<Rc<RefCell<FunctionParameters>>> {
        if let Some(params) = &self.parameters {
            Some(Rc::clone(params))
        } else {
            None
        }
    }

    /// 順伝播
    fn forward(&self, x: Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let result = x.mapv(|x| x.powi(2));
        dbg!(&result);
        result
    }

    /// 逆伝播
    /// y=x^2 の微分であるため、dy/dx=2x である。
    fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        if let Some(input) = &self.parameters.as_ref().unwrap().borrow().get_input() {
            let x = input.borrow().data.clone();
            let gx = 2.0 * x * gy;
            gx
        } else {
            panic!("Error: Forward propagation must be performed in advance.");
        }
    }
}

/// Exp 関数
#[derive(Debug, Clone)]
struct Exp {
    parameters: Option<Rc<RefCell<FunctionParameters>>>,
}
impl Function for Exp {
    // 関数の入出力値のセッター
    fn set_parameters(&mut self, input: Rc<RefCell<Variable>>, output: Rc<RefCell<Variable>>) {
        self.parameters = Some(Rc::new(RefCell::new(FunctionParameters::new(
            input, output,
        ))));
    }
    fn get_parameters(&self) -> Option<Rc<RefCell<FunctionParameters>>> {
        if let Some(params) = &self.parameters {
            Some(Rc::clone(params))
        } else {
            None
        }
    }

    // Exp (y=e^x) の順伝播
    fn forward(&self, x: Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let e = std::f64::consts::E;
        x.mapv(|x| e.powf(x))
    }

    /// 逆伝播
    /// dy/dx=e^x である。
    fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        if let Some(input) = &self.parameters.as_ref().unwrap().borrow().get_input() {
            let x = input.borrow().data.clone();
            let e = std::f64::consts::E;
            let gx = x.mapv(|x| e.powf(x)) * gy;
            gx
        } else {
            panic!("Error: Forward propagation must be performed in advance.");
        }
    }
}

/// 合成する関数を保持する構造体
/// 順伝播と逆伝播をループで実装する。
/// ステップ7 や ステップ8 は、順伝播の結果の Variable の creator を使って逆伝播を実現しているが
/// Rust で同じ構成を実現するのは難しく、トレイトオブジェクトを vec で保持して繰り返し処理するメソッドで実現してみた。
#[derive(Debug)]
struct Functions {
    functions: Vec<Box<dyn Function>>,
    result: Option<Rc<RefCell<Variable>>>,
}
impl Functions {
    /// 合成する関数を追加する。
    ///
    /// Arguments
    /// * function (Box<dyn Function>) : 合成する関数 Function のトレイトオブジェクト
    fn push(&mut self, function: Box<dyn Function>) {
        self.functions.push(function);
    }

    /// 合成関数の順伝播
    ///
    /// Arguments
    /// * input (Rc<RefCell<Variable>>): 入力値
    fn foward(&mut self, input: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
        let mut output = Rc::clone(&input); //input.clone();
        for function in self.functions.iter_mut() {
            output = function.call(output);
        }

        self.result = Some(output.clone());
        output
    }

    /// 合成関数の逆伝播
    /// 事前に順伝播を実行している必要がある。
    fn backward(&mut self) {
        if let Some(output) = self.result.as_mut() {
            let mut grad = Array::from_elem(IxDyn(&[]), 1.0);
            output.borrow_mut().grad = Some(grad.clone());
            for function in self.functions.iter_mut().rev() {
                grad = function.backward(&grad);

                function
                    .get_parameters()
                    .unwrap()
                    .borrow_mut()
                    .set_input_grad(grad.clone());
            }
        } else {
            println!("Error Forward propagation has not beenForward propagation has not been executed. Please execute forward propagation first.");
        }
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
fn numerical_diff<F: Function>(mut f: F, x: Variable, eps: f64) -> Array<f64, IxDyn> {
    let eps_array = Array::from_elem(IxDyn(&[]), eps);

    let x0 = Rc::new(RefCell::new(Variable::new(&x.data - &eps_array)));
    let x1 = Rc::new(RefCell::new(Variable::new(&x.data + &eps_array)));
    let y0 = f.call(x0);
    let y1 = f.call(x1);

    let result = (y1.borrow().clone().data - y0.borrow().clone().data) / (eps * 2.0);

    result
}

fn main() {
    let x = Rc::new(RefCell::new(Variable::new(Array::from_elem(
        IxDyn(&[]),
        0.5,
    ))));

    let a_square = Box::new(Square { parameters: None });
    let b_exp = Box::new(Exp { parameters: None });
    let c_square = Box::new(Square { parameters: None });

    let mut functions: Functions = Functions {
        functions: vec![],
        result: None,
    };
    functions.push(a_square);
    functions.push(b_exp);
    functions.push(c_square);

    // 順伝播
    let output = functions.foward(Rc::clone(&x));

    // 逆伝播
    functions.backward();

    dbg!(functions);
    dbg!(output.clone());
    dbg!(x.clone());

    /*
    let a = a_square.call(x.clone());
    dbg!(a.clone());
    let b = b_exp.call(a.clone());
    dbg!(b.clone());
    let y = c_square.call(b.clone());
    dbg!(y.clone());

    y.borrow_mut().grad = Some(Array::from_elem(IxDyn(&[]), 1.0));

    b.borrow_mut().grad = Some(c_square.backward(y.borrow().grad.as_ref().unwrap()));

    a.borrow_mut().grad = Some(b_exp.backward(b.borrow().grad.as_ref().unwrap()));
    x.borrow_mut().grad = Some(a_square.backward(a.borrow().grad.as_ref().unwrap()));

    dbg!(a_square);
    dbg!(b_exp);
    dbg!(c_square);
     */
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::prelude::*;

    /// 二乗の順伝播テスト
    #[test]
    fn test_square() {
        let x = Rc::new(RefCell::new(Variable::new(Array::from_elem(
            IxDyn(&[]),
            2.0,
        ))));
        let mut square = Box::new(Square { parameters: None });
        let y = square.call(x);
        let expected = Array::from_elem(IxDyn(&[]), 4.0);
        assert_eq!(expected, y.borrow().data);
    }

    /// 二乗の逆伝播テスト
    #[test]
    fn test_backward() {
        let x = Rc::new(RefCell::new(Variable::new(Array::from_elem(
            IxDyn(&[]),
            3.0,
        ))));
        let expected = Array::from_elem(IxDyn(&[]), 6.0);

        let square = Box::new(Square { parameters: None });
        let mut functions: Functions = Functions {
            functions: vec![],
            result: None,
        };
        functions.push(square);

        functions.foward(x.clone());
        functions.backward();

        dbg!(expected.clone());
        dbg!(x.clone());

        assert_eq!(expected, x.borrow().grad.clone().unwrap());
    }

    /// 勾配確認によるテスト
    #[test]
    fn test_gradient_check() {
        let mut rng = rand::rng();
        let randnum = rng.random::<f64>();

        let square = Square { parameters: None };
        let x = Variable::new(Array::from_elem(IxDyn(&[]), randnum));

        let mut functions: Functions = Functions {
            functions: vec![],
            result: None,
        };
        functions.push(Box::new(square.clone()));

        functions.foward(Rc::new(RefCell::new(x.clone())));
        functions.backward();
        let expected_grad = functions.functions[0]
            .get_parameters()
            .unwrap()
            .borrow()
            .input
            .as_ref()
            .unwrap()
            .borrow()
            .grad
            .clone();

        let num_grad = numerical_diff(square, x, 1e-4);

        assert_abs_diff_eq!(expected_grad.unwrap(), num_grad.clone(), epsilon = 1e-4);
    }
}
