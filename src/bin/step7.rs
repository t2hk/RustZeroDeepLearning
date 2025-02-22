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

    fn backward(self) {
        match self.creator {
            Some(creator) => {
                let x = creator.upgrade().unwrap().borrow().get_input();
                println!("valiable backword x: {:?}", x);
                // match x {
                //     Some(x) => {
                //         &x.grad = Some(
                //             creator
                //                 .upgrade()
                //                 .unwrap()
                //                 .borrow()
                //                 .backward(self.grad.as_ref().unwrap()),
                //         );
                //         x.backward();
                //     }
                //     None => {}
                // }
            }
            None => {
                println!("variable backward: None");
            }
        }
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
    fn call(&self, input: &Variable, self_rc: Rc<RefCell<dyn Function>>) -> Variable {
        let x = &input.data;
        let y = self.forward(x);

        let mut output = Variable::new(y);
        output.set_creator(Rc::clone(&self_rc));

        self_rc
            .borrow_mut()
            .keep_input_output(input.clone(), output.clone());

        output
    }

    fn keep_input_output(&mut self, input: Variable, output: Variable);

    /// 通常の計算を行う順伝播。継承して実装すること。
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn>;

    /// 微分の計算を行う逆伝播。継承して実装すること。
    fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn>;

    fn get_input(&self) -> Option<Variable>;
}

/// 二乗する。
#[derive(Debug, Clone)]
struct Square {
    input: Option<Weak<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
}

impl Function for Square {
    /// 順伝播
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        x.mapv(|x| x.powi(2))
    }

    fn keep_input_output(&mut self, input: Variable, output: Variable) {
        self.input = Some(Rc::downgrade(&Rc::new(RefCell::new(input))));
        self.output = Some(Rc::downgrade(&Rc::new(RefCell::new(output))));
    }

    fn get_input(&self) -> Option<Variable> {
        match &self.input {
            Some(input) => {
                let input = input.upgrade().unwrap().borrow().clone();
                Some(input)
            }
            None => None,
        }
    }

    /// 逆伝播
    /// y=x^2 の微分であるため、dy/dx=2x である。
    fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let x = self
            .input
            .as_ref()
            .unwrap()
            .upgrade()
            .unwrap()
            .borrow()
            .data
            .clone();
        let gx = 2.0 * x * gy;
        gx
    }
}

/// Exp 関数
#[derive(Clone)]
struct Exp {
    input: Option<Weak<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
}

impl Function for Exp {
    // Exp (y=e^x) の順伝播
    fn forward(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let e = std::f64::consts::E;
        x.mapv(|x| e.powf(x))
    }

    fn keep_input_output(&mut self, input: Variable, output: Variable) {
        self.input = Some(Rc::downgrade(&Rc::new(RefCell::new(input))));
        self.output = Some(Rc::downgrade(&Rc::new(RefCell::new(output))));
    }

    fn get_input(&self) -> Option<Variable> {
        match &self.input {
            Some(input) => {
                let input = input.upgrade().unwrap().borrow().clone();
                Some(input)
            }
            None => None,
        }
    }
    /// 逆伝播
    /// dy/dx=e^x である。
    fn backward(&self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let x = &self
            .input
            .as_ref()
            .unwrap()
            .upgrade()
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
    // 順伝播
    // x -> Square -> a -> Exp -> b -> Square -> y
    let a_square = Square {
        input: None,
        output: None,
    };
    let b_exp = Exp {
        input: None,
        output: None,
    };
    let c_square = Square {
        input: None,
        output: None,
    };

    let mut x = Variable::new(Array::from_elem(IxDyn(&[]), 0.5));
    println!("x: {}", x.data);
    let rc_a_square = Rc::new(RefCell::new(a_square.clone()));
    let mut a = a_square.call(&x, rc_a_square);
    println!("a: {}", a.data);
    let rc_b_exp = Rc::new(RefCell::new(b_exp.clone()));
    let mut b = b_exp.call(&a, rc_b_exp);
    println!("b: {}", b.data);
    let rc_c_square = Rc::new(RefCell::new(c_square.clone()));
    let mut y = c_square.call(&b, rc_c_square);
    println!("y: {}", y.data);

    x.clone().backward();
    a.clone().backward();
    b.clone().backward();
    y.clone().backward();

    // 逆伝播
    // x.grad <- a.backward <- a.grad <- b.backward <- b.grad <- c.backward <- y.grad(=1.0)
    y.grad = Some(Array::from_elem(IxDyn(&[]), 1.0));
    println!("y.grad: {}", y.grad.clone().unwrap());
    b.grad = Some(c_square.backward(&y.grad.unwrap()));
    println!("b.grad: {}", b.grad.clone().unwrap());
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
        let f = Square {
            input: None,
            output: None,
        };
        let rc_f = Rc::new(RefCell::new(f.clone()));
        let y = f.call(&x, rc_f);
        let expected = Array::from_elem(IxDyn(&[]), 0.0);
        assert_eq!(expected, y.data);
    }

    /// 1の二乗のテスト。
    #[test]
    fn test_one_square() {
        let x = Variable::new(Array::from_elem(IxDyn(&[]), 1.0));
        let f = Square {
            input: None,
            output: None,
        };
        let rc_f = Rc::new(RefCell::new(f.clone()));
        let y = f.call(&x, rc_f);
        let expected = Array::from_elem(IxDyn(&[]), 1.0);
        assert_eq!(expected, y.data);
    }

    /// 10の二乗のテスト。
    #[test]
    fn test_ten_square() {
        let x = Variable::new(Array::from_elem(IxDyn(&[]), 10.0));
        let f = Square {
            input: None,
            output: None,
        };
        let rc_f = Rc::new(RefCell::new(f.clone()));
        let y = f.call(&x, rc_f);
        let expected = Array::from_elem(IxDyn(&[]), 100.0);
        assert_eq!(expected, y.data);
    }

    /// 負の値の二乗のテスト。
    #[test]
    fn test_negative_square() {
        let x = Variable::new(Array::from_elem(IxDyn(&[]), -5.0));
        let f = Square {
            input: None,
            output: None,
        };
        let rc_f = Rc::new(RefCell::new(f.clone()));
        let y = f.call(&x, rc_f);
        let expected = Array::from_elem(IxDyn(&[]), 25.0);
        assert_eq!(expected, y.data);
    }

    /// Exp 関数のテスト。
    #[test]
    fn test_exp_2() {
        let x = Variable::new(Array::from_elem(IxDyn(&[]), 2.0));
        let f = Exp {
            input: None,
            output: None,
        };
        let rc_f = Rc::new(RefCell::new(f.clone()));
        let y = f.call(&x, rc_f);
        let e = std::f64::consts::E;
        let expected = Array::from_elem(IxDyn(&[]), e.powf(2.0));

        assert_eq!(expected, y.data);
    }
}
