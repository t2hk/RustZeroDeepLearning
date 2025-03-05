use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use rand::random_iter;
use std::cell::RefCell;
use std::rc::Rc;

/// Variable 構造体
/// * data (Array<f64, IxDyn>): 変数
/// * grad (Option<Array<f64, IxDyn>): 変数に対応した微分した値。逆伝播によって実際に微分が計算されたときに値を設定する。
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

        let mut gys: Vec<Array<f64, IxDyn>> = vec![];
        self.outputs
            .as_mut()
            .unwrap()
            .iter_mut()
            .for_each(|output| {
                if output.borrow().grad.is_none() {
                    output.borrow_mut().grad = Some(grad_one.clone());
                }
                gys.push(output.borrow().grad.clone().unwrap());
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

/// Add 関数
#[derive(Debug, Clone)]
struct Add;
impl Function for Add {
    // Add (加算) の順伝播
    fn forward(&self, xs: Vec<Array<f64, IxDyn>>) -> Vec<Array<f64, IxDyn>> {
        let result = vec![&xs[0] + &xs[1]];
        //dpg!(result);
        result
    }

    /// 逆伝播
    /// y=x0+x1 の微分であるため、dy/dx0=1, dy/dx1=1 である。
    fn backward(
        &self,
        inputs: Vec<Rc<RefCell<Variable>>>,
        gys: Vec<Array<f64, IxDyn>>,
    ) -> Vec<Array<f64, IxDyn>> {
        vec![gys[0].clone(), gys[0].clone()]
    }
}

fn main() {
    let x1: Rc<RefCell<Variable>> = Rc::new(RefCell::new(Variable::new(2.0)));
    let x2: Rc<RefCell<Variable>> = Rc::new(RefCell::new(Variable::new(3.0)));

    let square = Square;
    let mut square_exe = FunctionExecutor::new(Rc::new(RefCell::new(square)));

    square_exe.forward(vec![x1.clone()]);
    square_exe.backward();

    dbg!(square_exe);
    dbg!(x1.borrow());
    dbg!(x2.borrow());

    let add = Add;
    let mut add_exe = FunctionExecutor::new(Rc::new(RefCell::new(add)));
    add_exe.forward(vec![x1.clone(), x2.clone()]);
    add_exe.backward();
    dbg!(add_exe);
    dbg!(x1.borrow());
    dbg!(x2.borrow());
}

#[cfg(test)]
mod tests {
    use super::*;
    // use approx::assert_abs_diff_eq;
    use rand::prelude::*;

    /// 二乗のテスト
    #[test]
    fn test_square() {
        // 2乗する値をランダムに生成する。
        let mut rng = rand::rng();
        let rand_x = rng.random::<f64>();
        let x = Rc::new(RefCell::new(Variable::new(rand_x)));

        // 2乗した結果の期待値を計算する。
        let expected_output_data = Array::from_elem(IxDyn(&[]), rand_x * rand_x);

        // 2乗の順伝播、逆伝播を実行する。
        let square = Square;
        let mut square_exe = FunctionExecutor::new(Rc::new(RefCell::new(square)));

        square_exe.forward(vec![x.clone()]);
        square_exe.backward();

        // 順伝播と逆伝播の処理結果を取得する。
        let input_result = square_exe.clone().inputs.unwrap().get(0).unwrap().clone();
        let output_result = square_exe.clone().outputs.unwrap().get(0).unwrap().clone();

        let input_data = input_result.borrow().data.clone();
        let input_grad = input_result.borrow().grad.clone().unwrap();
        let output_data = output_result.borrow().data.clone();
        let output_grad = output_result.borrow().grad.clone().unwrap();

        dbg!(square_exe.clone());
        dbg!(input_result.clone());
        dbg!(output_result.clone());

        assert_eq!(Array::from_elem(IxDyn(&[]), rand_x.clone()), input_data);
        assert_eq!(expected_output_data.clone(), output_data.clone());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), output_grad.clone());
        assert_eq!(
            Array::from_elem(IxDyn(&[]), 2.0 * 1.0 * rand_x.clone()),
            input_grad.clone()
        );
    }

    /// 加算のテスト
    #[test]
    fn test_add() {
        // 加算値をランダムに生成する。
        let mut rng = rand::rng();
        let rand_x1 = rng.random::<f64>();
        let rand_x2 = rng.random::<f64>();
        let x1 = Rc::new(RefCell::new(Variable::new(rand_x1)));
        let x2 = Rc::new(RefCell::new(Variable::new(rand_x2)));

        // 加算した結果の期待値を計算する。
        let expected_output_data = Array::from_elem(IxDyn(&[]), rand_x1 + rand_x2);

        // 順伝播、逆伝播を実行する。
        let add = Add;
        let mut add_exe = FunctionExecutor::new(Rc::new(RefCell::new(add)));

        add_exe.forward(vec![x1.clone(), x2.clone()]);
        add_exe.backward();

        // 順伝播と逆伝播の処理結果を取得する。
        let input1_result = add_exe.clone().inputs.unwrap().get(0).unwrap().clone();

        let input2_result = add_exe.clone().inputs.unwrap().get(1).unwrap().clone();
        let output_result = add_exe.clone().outputs.unwrap().get(0).unwrap().clone();

        let input1_data = input1_result.borrow().data.clone();
        let input2_data = input2_result.borrow().data.clone();
        let input1_grad = input1_result.borrow().grad.clone().unwrap();
        let input2_grad = input2_result.borrow().grad.clone().unwrap();
        let output_data = output_result.borrow().data.clone();
        let output_grad = output_result.borrow().grad.clone().unwrap();

        dbg!(add_exe.clone());
        dbg!(input1_result.clone());
        dbg!(input2_result.clone());
        dbg!(output_result.clone());

        assert_eq!(Array::from_elem(IxDyn(&[]), rand_x1.clone()), input1_data);
        assert_eq!(Array::from_elem(IxDyn(&[]), rand_x2.clone()), input2_data);

        assert_eq!(expected_output_data.clone(), output_data.clone());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), output_grad.clone());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), input1_grad.clone());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), input2_grad.clone());
    }
}
