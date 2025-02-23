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
    /// * input (Rc<RefCell<Variable>>): 入力値
    /// * output (Rc<RefCell<Variable>>): 出力値
    fn set_parameters(&mut self, input: Rc<RefCell<Variable>>, output: Rc<RefCell<Variable>>);

    /// 入力値、出力値を取得する。
    /// 入力値、出力値が存在しない場合は None を返却する。
    ///
    /// Returns
    /// * Option<&FunctionParameters>: 入力値、出力値
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

    /// 微分の計算を行う逆伝播。
    /// 継承して実装すること。
    ///
    /// Arguments
    /// * gy (Array<f64, IxDyn>): 出力値に対する微分値
    ///
    /// Returns
    /// * Array<f64, IxDyn>: 入力値に対する微分値
    fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn>;

    /// 関数を実行する。
    /// 関数の実装は Function を継承して行う。
    ///
    /// # Arguments
    /// * input (Variable) - 変数
    ///
    /// # Return
    /// * Variable - 処理結果
    fn call(
        &mut self,
        input: &Variable,
        self_rc: Rc<RefCell<dyn Function>>,
    ) -> Rc<RefCell<Variable>> {
        // 関数を実行する。
        let x = Rc::new(RefCell::new(input.clone()));
        let y = Rc::new(RefCell::new(Variable::new(self.forward(&x.borrow().data))));

        // 関数の出力結果に生成元の関数を設定する。
        y.borrow_mut().set_creator(Rc::clone(&self_rc));
        // 関数の入力、出力を設定する。
        self.set_parameters(x.clone(), y.clone());

        y.clone()
    }

    /// デバッグ用の出力
    fn print(&self) {
        println!(
            "Function::print \n input: {:?} \n output: {:?}",
            self.get_parameters().unwrap().get_input().clone(),
            self.get_parameters().unwrap().get_output().clone(),
        );
    }
}

/// FunctionParameters 構造体
/// 関数に対する入力値、出力値を保持する。
///
/// Arguments
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
            input: Some(input.clone()),
            output: Some(output.clone()),
        }
    }

    /// 入力値を取得する。
    /// 入力値が存在しない場合は None を返却する。
    ///
    /// Returns
    /// * Option<Rc<RefCell<Variable>>>: 入力値
    fn get_input(&self) -> Option<Rc<RefCell<Variable>>> {
        println!("call get_input");

        match &self.input {
            Some(input) => Some(input.clone()),
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
    /// * Option<Rc<RefCell<Variable>>>: 出力値
    fn get_output(&self) -> Option<Rc<RefCell<Variable>>> {
        println!("call get_output");
        match &self.output {
            Some(output) => Some(output.clone()),
            None => {
                println!("self.input is None.");
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
    // 関数の入出力値のセッター
    fn set_parameters(&mut self, input: Rc<RefCell<Variable>>, output: Rc<RefCell<Variable>>) {
        self.parameters = Some(FunctionParameters::new(input.clone(), output.clone()));
    }
    // 関数の入出力値のゲッター
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

    /// 逆伝播
    /// y=x^2 の微分であるため、dy/dx=2x である。
    fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let x = self
            .get_parameters()
            .unwrap()
            .get_input()
            .as_ref()
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
        self.parameters = Some(FunctionParameters::new(input.clone(), output.clone()));
    }
    // 関数の入出力値のゲッター
    fn get_parameters(&self) -> Option<&FunctionParameters> {
        match &self.parameters {
            Some(parameters) => Some(&parameters),
            None => None,
        }
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
            .get_parameters()
            .unwrap()
            .get_input()
            .as_ref()
            .unwrap()
            .borrow()
            .data
            .clone();
        let e = std::f64::consts::E;
        let gx = x.mapv(|x| e.powf(x)) * gy;
        gx
    }
}

fn main() {
    let mut a_square = Square { parameters: None };
    let mut b_exp = Exp { parameters: None };
    let mut c_square = Square { parameters: None };

    let x = Rc::new(RefCell::new(Variable::new(Array::from_elem(
        IxDyn(&[]),
        0.5,
    ))));

    let a = a_square.call(&x.borrow(), Rc::new(RefCell::new(a_square.clone())));
    let b = b_exp.call(&a.borrow(), Rc::new(RefCell::new(b_exp.clone())));
    let y = c_square.call(&b.borrow(), Rc::new(RefCell::new(c_square.clone())));

    y.borrow_mut().grad = Some(Array::from_elem(IxDyn(&[]), 1.0));
    b.borrow_mut().grad = Some(c_square.backward(y.borrow().grad.as_ref().unwrap()));
    a.borrow_mut().grad = Some(b_exp.backward(b.borrow().grad.as_ref().unwrap()));
    x.borrow_mut().grad = Some(a_square.backward(a.borrow().grad.as_ref().unwrap()));

    dbg!(&y);
    dbg!(&x.borrow().grad);

    println!("y: {:?}, x.grad: {:?}", y.borrow().data, x.borrow().grad);
    c_square.print();
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_square_forward() {
        let square = Square { parameters: None };
        let x = array![2.0].into_dyn();
        let y = square.forward(&x);
        assert_eq!(y, array![4.0].into_dyn());
    }

    #[test]
    fn test_square_backward() {
        let mut square = Square { parameters: None };
        let x = Rc::new(RefCell::new(Variable::new(array![2.0].into_dyn())));
        let y = Rc::new(RefCell::new(Variable::new(array![4.0].into_dyn())));
        square.set_parameters(x.clone(), y.clone());

        let gy = array![1.0].into_dyn();
        let gx = square.backward(&gy);
        assert_eq!(gx, array![4.0].into_dyn());
    }

    #[test]
    fn test_exp_forward() {
        let exp = Exp { parameters: None };
        let x = array![1.0].into_dyn();
        let y = exp.forward(&x);
        assert_eq!(y, array![std::f64::consts::E].into_dyn());
    }

    #[test]
    fn test_exp_backward() {
        let mut exp = Exp { parameters: None };
        let x = Rc::new(RefCell::new(Variable::new(array![1.0].into_dyn())));
        let y = Rc::new(RefCell::new(Variable::new(
            array![std::f64::consts::E].into_dyn(),
        )));
        exp.set_parameters(x.clone(), y.clone());

        let gy = array![1.0].into_dyn();
        let gx = exp.backward(&gy);
        assert_eq!(gx, array![std::f64::consts::E].into_dyn());
    }

    #[test]
    fn test_chain() {
        let mut a_square = Square { parameters: None };
        let mut b_exp = Exp { parameters: None };
        let mut c_square = Square { parameters: None };

        let x = Rc::new(RefCell::new(Variable::new(Array::from_elem(
            IxDyn(&[]),
            0.5,
        ))));

        let a = a_square.call(&x.borrow(), Rc::new(RefCell::new(a_square.clone())));
        let b = b_exp.call(&a.borrow(), Rc::new(RefCell::new(b_exp.clone())));
        let y = c_square.call(&b.borrow(), Rc::new(RefCell::new(c_square.clone())));

        y.borrow_mut().grad = Some(Array::from_elem(IxDyn(&[]), 1.0));
        b.borrow_mut().grad = Some(c_square.backward(y.borrow().grad.as_ref().unwrap()));
        a.borrow_mut().grad = Some(b_exp.backward(b.borrow().grad.as_ref().unwrap()));
        x.borrow_mut().grad = Some(a_square.backward(a.borrow().grad.as_ref().unwrap()));

        assert_eq!(
            y.borrow().data,
            Array::from_elem(IxDyn(&[]), 1.648721270700128)
        );
        assert_eq!(
            x.borrow().grad,
            Some(Array::from_elem(IxDyn(&[]), 3.297442541400256))
        );
    }
}
