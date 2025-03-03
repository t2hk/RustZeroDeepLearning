//! ステップ12 可変長の引数(改善編)
//!
//! 順伝播について、可変長の引数に対応し、加算関数を追加した。
//! 逆伝播については一時的にコメントアウトしている (ステップ13 で対応予定)。
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
            grad: None,
        }
    }
}

/// CreateVariable トレイトの f64 用の実装
impl CreateVariable for f64 {
    fn create_variable(&self) -> Variable {
        Variable {
            data: Array::from_elem(IxDyn(&[]), *self),
            grad: None,
        }
    }
}

impl Variable {
    /// Variable のコンストラクタ。
    ///
    /// # Arguments
    /// * data - 変数    
    fn new<T: CreateVariable>(data: T) -> Variable {
        CreateVariable::create_variable(&data)
    }
}

/// 関数の入出力値を持つ構造体
/// * inputs (Option<Vec<Rc<RefCell<Variable>>>>): 入力値
/// * outputs (Option<Vec<Rc<RefCell<Variable>>>>): 出力値
#[derive(Debug, Clone)]
struct FunctionParameters {
    inputs: Option<Vec<Rc<RefCell<Variable>>>>,
    outputs: Option<Vec<Rc<RefCell<Variable>>>>,
}
impl FunctionParameters {
    /// コンストラクタ
    /// 入力値、出力値を設定する。
    ///
    /// Arguments
    /// * inputs (Vec<Rc<RefCell<Variable>>>): 入力値
    /// * outputs (Vec<Rc<RefCell<Variable>>>): 出力値
    fn new(
        inputs: Vec<Rc<RefCell<Variable>>>,
        outputs: Vec<Rc<RefCell<Variable>>>,
    ) -> FunctionParameters {
        FunctionParameters {
            inputs: Some(inputs),
            outputs: Some(outputs),
        }
    }

    fn add_input(&mut self, input: Rc<RefCell<Variable>>) {
        if let Some(inputs) = &mut self.inputs {
            inputs.push(input);
        } else {
            self.inputs = Some(vec![input]);
        }
    }

    fn add_output(&mut self, output: Rc<RefCell<Variable>>) {
        if let Some(outputs) = &mut self.outputs {
            outputs.push(output);
        } else {
            self.outputs = Some(vec![output]);
        }
    }

    //
    // ステップ10 可変長の引数(順伝播編) の対応のため、一時的にコメントアウト
    //
    /// 逆伝播で算出した微分値を設定する。
    ///
    /// Arguments
    /// * grad (Array<f64, IxDyn>): 微分値
    // fn set_input_grad(&mut self, grad: Array<f64, IxDyn>) {
    //   self.input.as_mut().unwrap().borrow_mut().grad = Some(grad);
    // }

    /// 関数の入力値を取得する。
    ///
    /// Return
    /// * Option<Vec<Rc<RefCell<Variable>>>>: 入力値
    fn get_inputs(&self) -> Option<Vec<Rc<RefCell<Variable>>>> {
        if let Some(inputs) = &self.inputs {
            Some(inputs.clone())
        } else {
            None
        }
    }

    /// 関数の出力値を取得する。
    ///
    /// Return
    /// * Option<Vec<Rc<RefCell<Variable>>>>: 出力値
    fn get_outputs(&self) -> Option<Vec<Rc<RefCell<Variable>>>> {
        if let Some(outputs) = &self.outputs {
            Some(outputs.clone())
        } else {
            None
        }
    }
}

trait Function: std::fmt::Debug {
    fn set_parameters(
        &mut self,
        inputs: Vec<Rc<RefCell<Variable>>>,
        outputs: Vec<Rc<RefCell<Variable>>>,
    );
    fn get_parameters(&self) -> Option<Rc<RefCell<FunctionParameters>>>;

    /// 順伝播
    /// 通常の計算を行う順伝播。継承して実装すること。
    ///
    /// Arguments
    /// * xs (Vec<Array<f64, IxDyn>>): 入力値
    ///
    /// Returns
    /// * Vec<Array<f64, IxDyn>>: 出力値
    fn forward(&self, xs: Vec<Array<f64, IxDyn>>) -> Vec<Array<f64, IxDyn>>;

    //
    // ステップ10 可変長の引数(順伝播編) の対応のため、一時的にコメントアウト
    //
    /// 微分の計算を行う逆伝播。
    /// 継承して実装すること。
    ///
    /// Arguments
    /// * gy (Array<f64, IxDyn>): 出力値に対する微分値
    ///
    /// Returns
    /// * Array<f64, IxDyn>: 入力値に対する微分値
    /// fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn>;

    fn call(&mut self, inputs: Vec<Rc<RefCell<Variable>>>) -> Vec<Rc<RefCell<Variable>>> {
        // 入力値からデータを取り出す。
        let xs_data: Vec<Array<f64, IxDyn>> = inputs
            .iter()
            .map(|input| input.borrow().data.clone())
            .collect();

        // 関数を実行する。
        let ys_data = self.forward(xs_data);

        // 関数の結果を出力値とする。
        let mut outputs: Vec<Rc<RefCell<Variable>>> = Vec::new();
        for y_data in ys_data.iter() {
            let y = Rc::new(RefCell::new(Variable::new(y_data.clone())));
            outputs.push(Rc::clone(&y));
        }

        // 関数の入力、出力を設定する。
        self.set_parameters(inputs, outputs.clone());

        outputs
    }
}

#[derive(Debug, Clone)]
struct Square {
    parameters: Option<Rc<RefCell<FunctionParameters>>>,
}

impl Function for Square {
    // 関数の入出力値のセッター
    fn set_parameters(
        &mut self,
        inputs: Vec<Rc<RefCell<Variable>>>,
        outputs: Vec<Rc<RefCell<Variable>>>,
    ) {
        self.parameters = Some(Rc::new(RefCell::new(FunctionParameters::new(
            inputs, outputs,
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
    fn forward(&self, xs: Vec<Array<f64, IxDyn>>) -> Vec<Array<f64, IxDyn>> {
        let result = vec![xs[0].mapv(|x| x.powi(2))];
        //dpg!(result);
        result
    }

    //
    // ステップ10 可変長の引数(順伝播編) の対応のため、一時的にコメントアウト
    //
    // 逆伝播
    // y=x^2 の微分であるため、dy/dx=2x である。
    // fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
    //     if let Some(input) = &self.parameters.as_ref().unwrap().borrow().get_input() {
    //         let x = input.borrow().data.clone();
    //         let gx = 2.0 * x * gy;
    //         gx
    //     } else {
    //         panic!("Error: Forward propagation must be performed in advance.");
    //     }
    // }
}

/// 二乗関数
///
/// Arguments
/// * x (Rc<RefCell<Variable>>): 二乗する変数
///
/// Return
///  * Rc<RefCell<Variable>>: 二乗結果
fn square(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let mut square = Box::new(Square { parameters: None });
    square.call(vec![x])[0].clone()
}

/// Exp 関数
#[derive(Debug, Clone)]
struct Exp {
    parameters: Option<Rc<RefCell<FunctionParameters>>>,
}
impl Function for Exp {
    // 関数の入出力値のセッター
    fn set_parameters(
        &mut self,
        inputs: Vec<Rc<RefCell<Variable>>>,
        outputs: Vec<Rc<RefCell<Variable>>>,
    ) {
        self.parameters = Some(Rc::new(RefCell::new(FunctionParameters::new(
            inputs, outputs,
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
    fn forward(&self, xs: Vec<Array<f64, IxDyn>>) -> Vec<Array<f64, IxDyn>> {
        let e = std::f64::consts::E;

        let result = vec![xs[0].mapv(|x| e.powf(x))];
        //dpg!(result);
        result
    }

    //
    // ステップ10 可変長の引数(順伝播編) の対応のため、一時的にコメントアウト
    //
    // 逆伝播
    // dy/dx=e^x である。
    //   fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
    //     if let Some(input) = &self.parameters.as_ref().unwrap().borrow().get_input() {
    //         let x = input.borrow().data.clone();
    //         let e = std::f64::consts::E;
    //         let gx = x.mapv(|x| e.powf(x)) * gy;
    //         gx
    //     } else {
    //         panic!("Error: Forward propagation must be performed in advance.");
    //     }
    // }
}

/// Exp 関数
///
/// Arguments
/// * x (Rc<RefCell<Variable>>): 指数関数の底
///
/// Return
/// * Rc<RefCell<Variable>>: 指数関数の結果  
fn exp(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let mut exp = Box::new(Exp { parameters: None });
    exp.call(vec![x])[0].clone()
}

/// Add 関数
#[derive(Debug, Clone)]
struct Add {
    parameters: Option<Rc<RefCell<FunctionParameters>>>,
}
impl Function for Add {
    // 関数の入出力値のセッター
    fn set_parameters(
        &mut self,
        inputs: Vec<Rc<RefCell<Variable>>>,
        outputs: Vec<Rc<RefCell<Variable>>>,
    ) {
        self.parameters = Some(Rc::new(RefCell::new(FunctionParameters::new(
            inputs, outputs,
        ))));
    }
    fn get_parameters(&self) -> Option<Rc<RefCell<FunctionParameters>>> {
        if let Some(params) = &self.parameters {
            Some(Rc::clone(params))
        } else {
            None
        }
    }

    // Add (加算) の順伝播
    fn forward(&self, xs: Vec<Array<f64, IxDyn>>) -> Vec<Array<f64, IxDyn>> {
        let result = vec![&xs[0] + &xs[1]];
        //dpg!(result);
        result
    }
}

/// 加算関数
///
/// Arguments
/// * x0 (Rc<RefCell<Variable>>): 加算する変数
/// * x1 (Rc<RefCell<Variable>>): 加算する変数
///
/// Return
/// * Rc<RefCell<Variable>>: 加算結果
fn add(x0: Rc<RefCell<Variable>>, x1: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let mut add = Box::new(Add { parameters: None });
    // 加算の順伝播
    add.call(vec![x0, x1])[0].clone()
}

/// 合成する関数を保持する構造体
/// 順伝播と逆伝播をループで実装する。
/// ステップ7 や ステップ8 は、順伝播の結果の Variable の creator を使って逆伝播を実現しているが
/// Rust で同じ構成を実現するのは難しく、トレイトオブジェクトを vec で保持して繰り返し処理するメソッドで実現してみた。
#[derive(Debug)]
struct Functions {
    functions: Vec<Box<dyn Function>>,
    result: Option<Vec<Rc<RefCell<Variable>>>>,
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
    /// * inputs (Vec<Rc<RefCell<Variable>>>): 入力値
    /// Return
    /// * Vec<Rc<RefCell<Variable>>: 出力値
    fn foward(&mut self, inputs: Vec<Rc<RefCell<Variable>>>) -> Vec<Rc<RefCell<Variable>>> {
        let mut outputs = inputs.clone();
        for function in self.functions.iter_mut() {
            outputs = function.call(outputs);
        }

        self.result = Some(outputs.clone());
        outputs

        // let mut output = Rc::clone(&input);
        // for function in self.functions.iter_mut() {
        //     output = function.call(output);
        // }

        // self.result = Some(output.clone());
        // output
    }

    //
    // ステップ10 可変長の引数(順伝播編) の対応のため、一時的にコメントアウト
    //
    // 合成関数の逆伝播
    // 事前に順伝播を実行している必要がある。
    // fn backward(&mut self) {
    //   if let Some(output) = self.result.as_mut() {
    //       let mut grad = Array::from_elem(IxDyn(&[]), 1.0);
    //       output.borrow_mut().grad = Some(grad.clone());
    //       for function in self.functions.iter_mut().rev() {
    //           grad = function.backward(&grad);

    //           function
    //               .get_parameters()
    //               .unwrap()
    //               .borrow_mut()
    //               .set_input_grad(grad.clone());
    //       }
    //   } else {
    //       println!("Error Forward propagation has not beenForward propagation has not been executed. Please execute forward propagation first.");
    //   }
    // }
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
    let y0 = f.call(vec![x0]);
    let y1 = f.call(vec![x1]);

    let result = (y1[0].borrow().clone().data - y0[0].borrow().clone().data) / (eps * 2.0);

    result
}

fn main() {
    let x = Rc::new(RefCell::new(Variable::new(0.5)));

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
    let outputs = functions.foward(vec![Rc::clone(&x)]);

    // 逆伝播
    // functions.backward();

    dbg!(functions);
    dbg!(outputs.clone());
    dbg!(x.clone());

    // 加算の実行
    let x1 = Rc::new(RefCell::new(Variable::new(1.0)));
    let x2 = Rc::new(RefCell::new(Variable::new(2.0)));

    let add = Box::new(Add { parameters: None });
    let mut functions: Functions = Functions {
        functions: vec![],
        result: None,
    };
    functions.push(add);
    // 加算の順伝播
    let outputs = functions.foward(vec![x1, x2]);
    dbg!(outputs.clone());
}

#[cfg(test)]
mod tests {
    use super::*;
    // use approx::assert_abs_diff_eq;
    use rand::prelude::*;

    /// FunctionParameters のテスト
    #[test]
    fn test_function_params() {
        let inputs = vec![Rc::new(RefCell::new(Variable::new(2.0)))];
        let outputs = vec![Rc::new(RefCell::new(Variable::new(3.0)))];

        // dbg!(inputs.clone());
        // dbg!(inputs.clone());

        let funcparams = FunctionParameters::new(inputs.clone(), outputs);
        let inputs_from_funcparams = funcparams.get_inputs();

        let update_value = Array::from_elem(IxDyn(&[]), 99.0);
        inputs[0].borrow_mut().data = update_value.clone(); //Array::from_elem(IxDyn(&[]), 99.0);

        assert_eq!(
            update_value,
            inputs_from_funcparams.unwrap()[0].borrow().data
        );
    }

    /// Function と FunctionParameters のテスト
    #[test]
    fn test_function_and_function_params() {
        let inputs = vec![Rc::new(RefCell::new(Variable::new(2.0)))];
        let outputs = vec![Rc::new(RefCell::new(Variable::new(3.0)))];

        let expected = Array::from_elem(IxDyn(&[]), 99.0);

        let mut square = Box::new(Square { parameters: None });
        square.set_parameters(inputs.clone(), outputs.clone());

        inputs[0].borrow_mut().data = expected.clone();
        inputs[0].borrow_mut().grad = Some(expected.clone());

        let data_from_func = square
            .get_parameters()
            .unwrap()
            .borrow()
            .get_inputs()
            .unwrap()[0]
            .borrow()
            .clone()
            .data;
        let grad_from_func = square
            .get_parameters()
            .unwrap()
            .borrow()
            .get_inputs()
            .unwrap()[0]
            .borrow()
            .clone()
            .grad
            .unwrap();

        assert_eq!(expected, data_from_func);
        assert_eq!(expected, grad_from_func);
    }

    /// 二乗の順伝播テスト
    #[test]
    fn test_square() {
        let x = Rc::new(RefCell::new(Variable::new(2.0)));
        let mut square: Box<Square> = Box::new(Square { parameters: None });
        let y = square.call(vec![x]);
        let expected = Array::from_elem(IxDyn(&[]), 4.0);
        assert_eq!(expected, y[0].borrow().data);
    }

    /// Exp 関数のテスト。
    #[test]
    fn test_exp() {
        let x = Rc::new(RefCell::new(Variable::new(2.0)));
        let mut exp: Box<Exp> = Box::new(Exp { parameters: None });

        let ys = exp.call(vec![x]);
        let e = std::f64::consts::E;
        let expected = Array::from_elem(IxDyn(&[]), e.powf(2.0));

        assert_eq!(expected, ys[0].borrow().data);
    }

    /// Add 関数のテスト。
    #[test]
    fn test_add() {
        let mut rng = rand::rng();

        let rand_x1 = rng.random::<f64>();
        let rand_x2 = rng.random::<f64>();

        let x1 = Rc::new(RefCell::new(Variable::new(rand_x1)));
        let x2 = Rc::new(RefCell::new(Variable::new(rand_x2)));

        let expected = Array::from_elem(IxDyn(&[]), rand_x1 + rand_x2);

        let add = Box::new(Add { parameters: None });
        let mut functions: Functions = Functions {
            functions: vec![],
            result: None,
        };
        functions.push(add);

        // 加算の順伝播
        let outputs = functions.foward(vec![x1, x2]);
        assert_eq!(expected, outputs[0].borrow().data);
    }

    /// 加算関数のテスト。
    #[test]
    fn test_add_simple() {
        let mut rng = rand::rng();

        let rand_x1 = rng.random::<f64>();
        let rand_x2 = rng.random::<f64>();
        let expected = Array::from_elem(IxDyn(&[]), rand_x1 + rand_x2);

        let x1 = Rc::new(RefCell::new(Variable::new(rand_x1)));
        let x2 = Rc::new(RefCell::new(Variable::new(rand_x2)));
        let result = add(x1, x2);
        assert_eq!(expected, result.borrow().data);
    }

    /// 二乗関数のテスト。
    #[test]
    fn test_square_simple() {
        let mut rng = rand::rng();

        let rand_x = rng.random::<f64>();
        let expected = Array::from_elem(IxDyn(&[]), rand_x * rand_x);

        let x = Rc::new(RefCell::new(Variable::new(rand_x)));
        let result = square(x);
        assert_eq!(expected, result.borrow().data);
    }

    /// Exp 関数のテスト。
    #[test]
    fn test_exp_simple() {
        let mut rng = rand::rng();

        let rand_x = rng.random::<f64>();

        let e = std::f64::consts::E;
        let expected = Array::from_elem(IxDyn(&[]), e.powf(rand_x));

        let x = Rc::new(RefCell::new(Variable::new(rand_x)));
        let result = exp(x);
        assert_eq!(expected, result.borrow().data);
    }

    /// 二乗と Exp の合成関数のテスト。
    #[test]
    fn add_and_square() {
        let mut rng = rand::rng();

        let rand_x1 = rng.random::<f64>();
        let rand_x2 = rng.random::<f64>();

        let x1 = Rc::new(RefCell::new(Variable::new(rand_x1)));
        let x2 = Rc::new(RefCell::new(Variable::new(rand_x2)));

        let expected = Array::from_elem(IxDyn(&[]), (rand_x1 + rand_x2) * (rand_x1 + rand_x2));

        let add = Box::new(Add { parameters: None });
        let square = Box::new(Square { parameters: None });

        let mut funcs: Functions = Functions {
            functions: vec![],
            result: None,
        };
        funcs.push(add);
        funcs.push(square);

        let outputs = funcs.foward(vec![x1, x2]);
        dbg!(funcs);
        assert_eq!(expected, outputs[0].borrow().data);
    }

    //
    // ステップ10 可変長の引数(順伝播編) の対応のため、一時的にコメントアウト
    //
    // 二乗の逆伝播テスト
    // #[test]
    // fn test_backward() {
    //     let x = Rc::new(RefCell::new(Variable::new(Array::from_elem(
    //         IxDyn(&[]),
    //         3.0,
    //     ))));
    //     let expected = Array::from_elem(IxDyn(&[]), 6.0);

    //     let square = Box::new(Square { parameters: None });
    //     let mut functions: Functions = Functions {
    //         functions: vec![],
    //         result: None,
    //     };
    //     functions.push(square);

    //     functions.foward(x.clone());
    //     functions.backward();

    //     dbg!(expected.clone());
    //     dbg!(x.clone());

    //     assert_eq!(expected, x.borrow().grad.clone().unwrap());
    // }

    //
    // ステップ10 可変長の引数(順伝播編) の対応のため、一時的にコメントアウト
    //
    // 勾配確認によるテスト
    // #[test]
    // fn test_gradient_check() {
    //     let mut rng = rand::rng();
    //     let randnum = rng.random::<f64>();

    //     let square = Square { parameters: None };
    //     let x = Variable::new(Array::from_elem(IxDyn(&[]), randnum));

    //     let mut functions: Functions = Functions {
    //         functions: vec![],
    //         result: None,
    //     };
    //     functions.push(Box::new(square.clone()));

    //     functions.foward(Rc::new(RefCell::new(x.clone())));
    //     functions.backward();
    //     let expected_grad = functions.functions[0]
    //         .get_parameters()
    //         .unwrap()
    //         .borrow()
    //         .input
    //         .as_ref()
    //         .unwrap()
    //         .borrow()
    //         .grad
    //         .clone();

    //     let num_grad = numerical_diff(square, x, 1e-4);

    //     assert_abs_diff_eq!(expected_grad.unwrap(), num_grad.clone(), epsilon = 1e-4);
    // }
}
