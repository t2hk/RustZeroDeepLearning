use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use rand::random_iter;
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
    fn new<T: CreateVariable>(data: T) -> Variable {
        CreateVariable::create_variable(&data)
    }
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

/// Function トレイト
trait Function: std::fmt::Debug {
    /// 順伝播
    /// 通常の計算を行う順伝播。継承して実装すること。
    ///
    /// Arguments
    /// * xs (Vec<Array<f64, IxDyn>>): 入力値
    ///
    /// Returns
    /// * Vec<Array<f64, IxDyn>>: 出力値
    fn forward(&self, xs: Vec<Array<f64, IxDyn>>) -> Vec<Array<f64, IxDyn>>;

    /// 微分の計算を行う逆伝播。
    /// 継承して実装すること。
    ///
    /// Arguments
    /// * gys (Vec<Array<f64, IxDyn>>): 出力値に対する微分値
    ///
    /// Returns
    /// * Vec<Array<f64, IxDyn>>: 入力値に対する微分値
    fn backward(
        &self,
        inputs: Vec<Rc<RefCell<Variable>>>,
        gys: Vec<Array<f64, IxDyn>>,
    ) -> Vec<Array<f64, IxDyn>>;
}

#[derive(Debug, Clone)]
struct FunctionExecutor {
    inputs: Option<Vec<Rc<RefCell<Variable>>>>,
    outputs: Option<Vec<Rc<RefCell<Variable>>>>,
    creator: Rc<RefCell<dyn Function>>,
}
impl FunctionExecutor {
    fn new(creator: Rc<RefCell<dyn Function>>) -> FunctionExecutor {
        FunctionExecutor {
            inputs: None,
            outputs: None,
            creator: creator,
        }
    }
    fn forward(&mut self, inputs: Vec<Rc<RefCell<Variable>>>) {
        // 入力値からデータを取り出す。
        let xs_data: Vec<Array<f64, IxDyn>> = inputs
            .iter()
            .map(|input| input.borrow().data.clone())
            .collect();

        // 関数を実行する。
        let ys_data = self.creator.borrow().forward(xs_data);

        // 関数の結果を出力値とする。
        let mut outputs: Vec<Rc<RefCell<Variable>>> = Vec::new();
        for y_data in ys_data.iter() {
            let y = Rc::new(RefCell::new(Variable::new(y_data.clone())));
            outputs.push(Rc::clone(&y));
        }

        self.inputs = Some(inputs);
        self.outputs = Some(outputs);
    }
    fn backward(&mut self) {
        // 逆伝播の最初の関数の微分値として 1.0 を設定する。
        let grad_one = Array::from_elem(IxDyn(&[]), 1.0);

        let mut gys = vec![];
        self.outputs.as_mut().unwrap().iter().for_each(|output| {
            if let Some(gy) = output.borrow_mut().grad.clone() {
                gys.push(gy.clone());
            } else {
                output.borrow_mut().grad = Some(grad_one.clone());
                gys.push(grad_one.clone());
            }
        });
        let gxs = self
            .creator
            .borrow_mut()
            .backward(self.inputs.clone().unwrap(), gys);

        for (i, input) in self.inputs.clone().unwrap().iter().enumerate() {
            input.borrow_mut().grad = Some(gxs[i].clone());
        }
    }
}

#[derive(Debug, Clone)]
struct Square;

impl Function for Square {
    /// 順伝播
    fn forward(&self, xs: Vec<Array<f64, IxDyn>>) -> Vec<Array<f64, IxDyn>> {
        let result = vec![xs[0].mapv(|x| x.powi(2))];
        //dpg!(result);
        result
    }

    /// 逆伝播
    /// y=x^2 の微分であるため、dy/dx=2x である。
    fn backward(
        &self,
        inputs: Vec<Rc<RefCell<Variable>>>,
        gys: Vec<Array<f64, IxDyn>>,
    ) -> Vec<Array<f64, IxDyn>> {
        let x = inputs[0].borrow().data.clone();
        let gxs = vec![2.0 * x * gys[0].clone()];
        gxs
    }
}

fn main() {
    let x: Rc<RefCell<Variable>> = Rc::new(RefCell::new(Variable::new(0.5)));

    let square = Square;
    let mut square_exe = FunctionExecutor::new(Rc::new(RefCell::new(square)));

    square_exe.forward(vec![x]);
    square_exe.backward();

    dbg!(square_exe);
}
