use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// Variable 構造体
/// * data (Array<f64, IxDyn>): 変数
/// * grad (Option<Array<f64, IxDyn>): 変数に対応した微分した値。逆伝播によって実際に微分が計算されたときに値を設定する。
/// * creator (Option<Weak<RefCell<dyn Function>>>): 生成元の関数
#[derive(Debug)]
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
            input: Some(input.clone()),
            output: Some(output.clone()),
        }
    }

    /// 逆伝播で算出した微分値を設定する。
    ///
    /// Arguments
    /// * grad (Array<f64, IxDyn>): 微分値
    fn set_input_grad(&mut self, grad: Array<f64, IxDyn>) {
        self.input.as_ref().unwrap().borrow_mut().grad = Some(grad);
    }
}

trait Function: std::fmt::Debug {
    fn set_parameters(&mut self, input: Rc<RefCell<Variable>>, output: Rc<RefCell<Variable>>);
    fn get_parameters(&mut self) -> Option<&mut FunctionParameters>;

    /// 順伝播
    /// 通常の計算を行う順伝播。継承して実装すること。
    ///
    /// Arguments
    /// * x (Array<f64, IxDyn>): 入力値
    ///
    /// Returns
    /// * Array<f64, IxDyn>: 出力値
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn>;

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
            self.forward(&input.borrow().data),
        )));

        // 関数の入力、出力を設定する。
        self.set_parameters(input, y.clone());

        y
    }
}

#[derive(Debug)]
struct Square {
    parameters: Option<FunctionParameters>,
}

impl Function for Square {
    fn set_parameters(&mut self, input: Rc<RefCell<Variable>>, output: Rc<RefCell<Variable>>) {
        self.parameters = Some(FunctionParameters::new(input, output));
    }
    fn get_parameters(&mut self) -> Option<&mut FunctionParameters> {
        self.parameters.as_mut()
    }

    /// 順伝播
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let result = x.mapv(|x| x.powi(2));
        dbg!(&result);
        result
    }

    /// 逆伝播
    /// y=x^2 の微分であるため、dy/dx=2x である。
    fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let x = &self
            .parameters
            .clone()
            .unwrap()
            .input
            .unwrap()
            .borrow()
            .data
            .clone();
        let gx = 2.0 * x * gy;
        gx
    }
}

/// Exp 関数
#[derive(Debug, Clone)]
struct Exp {
    parameters: Option<FunctionParameters>,
}
impl Function for Exp {
    // 関数の入出力値のセッター
    fn set_parameters(&mut self, input: Rc<RefCell<Variable>>, output: Rc<RefCell<Variable>>) {
        self.parameters = Some(FunctionParameters::new(input, output));
    }
    fn get_parameters(&mut self) -> Option<&mut FunctionParameters> {
        self.parameters.as_mut()
    }

    // Exp (y=e^x) の順伝播
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let e = std::f64::consts::E;
        x.mapv(|x| e.powf(x))
    }

    /// 逆伝播
    /// dy/dx=e^x である。
    fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let x = &self
            .parameters
            .clone()
            .unwrap()
            .input
            .unwrap()
            .borrow()
            .data
            .clone();
        let e = std::f64::consts::E;
        let gx = x.mapv(|x| e.powf(x)) * gy;
        gx
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
        let mut output = input.clone();
        for function in self.functions.iter_mut() {
            output = function.call(output);
        }

        self.result = Some(output.clone());
        output
    }

    fn backward(&mut self) {
        if let Some(output) = self.result.clone() {
            let mut grad = Array::from_elem(IxDyn(&[]), 1.0);
            output.borrow_mut().grad = Some(grad.clone());
            for function in self.functions.iter_mut().rev() {
                // function.backward(output.borrow().grad.as_ref().unwrap());
                grad = function.backward(&grad);
                // output.borrow_mut().grad = Some(grad.clone());
                function
                    .get_parameters()
                    .unwrap()
                    .set_input_grad(grad.clone());
            }
        } else {
            println!("Error Forward propagation has not beenForward propagation has not been executed. Please execute forward propagation first.");
        }
    }
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
    let output = functions.foward(x);

    // 逆伝播
    functions.backward();

    dbg!(functions);
    dbg!(output.clone());

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
