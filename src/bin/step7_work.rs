use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// Variable 構造体
/// * data (Array<f64, IxDyn>): 変数
/// * grad (Option<Array<f64, IxDyn>): 変数に対応した微分した値。逆伝播によって実際に微分が計算されたときに値を設定する。
/// * creator (Option<Weak<RefCell<dyn Function>>>): 生成元の関数
#[derive(Debug, Clone)]
struct Variable {
    data: Array<f64, IxDyn>,
    grad: Option<Array<f64, IxDyn>>,
    creator: Option<Weak<RefCell<dyn Function>>>,
}

impl Variable {
    /// Variable のコンストラクタ。
    ///
    /// # Arguments
    /// * data - 変数
    fn new(data: Array<f64, IxDyn>) -> Variable {
        Variable {
            data,
            grad: None,
            creator: None,
        }
    }

    /// この変数を生成した関数を設定する。
    ///
    /// # Arguments
    /// * creator - 生成元の関数
    fn set_creator(&mut self, creator: Rc<RefCell<dyn Function>>) {
        self.creator = Some(Rc::downgrade(&creator));
    }
}

/// Function トレイト
/// Variable を入力し、処理を実行して結果を Variable で返却する。
trait Function {
    /// 入力値、出力値を設定する。
    ///
    /// Arguments
    ///  * input (Rc<RefCell<Variable>>): 入力値
    /// * output (Rc<RefCell<Variable>>): 出力値
    fn set_parameters(&mut self, input: Rc<RefCell<Variable>>, output: Rc<RefCell<Variable>>);

    /// 入力値、出力値を取得する。
    /// 入力値、出力値が存在しない場合は None を返却する。
    ///
    /// Returns
    /// * Option<Parameters>: 入力値、出力値
    fn get_parameters(&self) -> Option<&FunctionParameters>;

    /// 順伝播
    /// 通常の計算を行う順伝播。継承して実装すること。
    ///
    /// Arguments
    /// * x (Array<f64, IxDyn>): 入力値
    ///
    /// Returns
    /// * Array<f64, IxDyn>: 出力値
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn>;

    /// デバッグ用の出力
    fn print(&self) {
        println!(
            "Function::print \n input: {:?} \n output: {:?}",
            &self.get_parameters().unwrap().get_input().clone(),
            &self.get_parameters().unwrap().get_output().clone(),
        );
    }
}

/// Parameters 構造体
/// 関数に対する入力値、出力値を保持する。
///
/// Arguments
/// * input (Option<Weak<RefCell<Variable>>>): 入力値
/// * output (Option<Weak<RefCell<Variable>>>): 出力値
#[derive(Debug, Clone)]
struct FunctionParameters {
    input: Option<Weak<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
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
            input: Some(Rc::downgrade(&input)),
            output: Some(Rc::downgrade(&output)),
        }
    }

    /// 入力値を取得する。
    /// 入力値が存在しない場合は None を返却する。
    ///
    /// Returns
    /// * Option<Variable>: 入力値
    fn get_input(&self) -> Option<Variable> {
        println!("call get_input");

        match &self.input {
            Some(input) => {
                if let Some(strong_input) = input.upgrade() {
                    println!("get_input input data: {:?}", strong_input.borrow().data);
                } else {
                    println!("get_input unknow input.");
                }
                let result = input.upgrade().unwrap().borrow().clone();
                Some(result)
            }
            None => {
                println!("self.input is None.");
                None
            }
        }
    }

    /// 出力値を取得する。
    /// 出力値が存在しない場合は None を返却する。
    ///
    /// Returns
    /// * Option<Variable>: 出力値
    fn get_output(&self) -> Option<Variable> {
        println!("call get_output");
        match &self.output {
            Some(output) => {
                if let Some(strong_output) = output.upgrade() {
                    // println!("get_output output data: {:?}", strong_output.borrow().data);
                    Some(strong_output.borrow().clone())
                } else {
                    println!("get_output unknow output.");
                    None
                }
            }
            None => {
                println!("self.output is None.");
                None
            }
        }
    }
}

/// 二乗する。
#[derive(Debug, Clone)]
struct Square {
    parameters: Option<FunctionParameters>,
}

impl Function for Square {
    fn set_parameters(&mut self, input: Rc<RefCell<Variable>>, output: Rc<RefCell<Variable>>) {
        self.parameters = Some(FunctionParameters::new(input.clone(), output.clone()));
    }
    fn get_parameters(&self) -> Option<&FunctionParameters> {
        match &self.parameters {
            Some(parameters) => Some(&parameters),
            None => None,
        }
    }

    /// 順伝播
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let result = x.mapv(|x| x.powi(2));
        dbg!(&result);
        result
    }
}

fn main() {
    let mut square = Square { parameters: None };

    let mut x = Rc::new(RefCell::new(Variable::new(Array::from_elem(
        IxDyn(&[]),
        0.5,
    ))));

    let mut y = Rc::new(RefCell::new(Variable::new(
        square.forward(&x.borrow().data.clone()),
    )));

    dbg!(&x);
    dbg!(&y);
    square.set_parameters(x.clone(), y.clone());

    let rc_square = Rc::new(RefCell::new(square.clone()));
    x.borrow_mut().set_creator(rc_square);

    square.print();
}
