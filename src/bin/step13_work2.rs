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
    //creator: Option<Rc<RefCell<dyn Function>>>,
    creator: Option<Rc<RefCell<FunctionExecutor>>>,
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
            creator: None,
        }
    }
}

/// CreateVariable トレイトの f64 用の実装
impl CreateVariable for f64 {
    fn create_variable(&self) -> Variable {
        Variable {
            data: Array::from_elem(IxDyn(&[]), *self),
            grad: None,
            creator: None,
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
    fn forward(&mut self, inputs: Vec<Rc<RefCell<Variable>>>) -> Vec<Rc<RefCell<Variable>>> {
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
            let mut val = Variable::new(y_data.clone());
            // val.creator = Some(self.creator.clone());
            val.creator = Some(Rc::new(RefCell::new(self.clone())));
            let y = Rc::new(RefCell::new(val));

            outputs.push(Rc::clone(&y));
        }

        self.inputs = Some(inputs);
        self.outputs = Some(outputs.clone());
        for output in outputs.clone().iter_mut() {
            output.borrow_mut().creator = Some(Rc::new(RefCell::new(self.clone())));
        }
        self.outputs.clone().unwrap()
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

    fn extract_creators(inputs: Vec<Rc<RefCell<Variable>>>) -> Vec<FunctionExecutor> {
        let mut creators = vec![];
        let mut local_inputs = inputs.clone();

        loop {
            let mut local_creators = vec![];
            println!("local inputs len: {:?}", local_inputs.len());
            local_inputs.iter().for_each(|input| {
                println!("local inputs iter");

                if let Some(creator) = input.borrow().clone().creator {
                    // let mut creator = input.borrow().clone().creator.unwrap().borrow().clone();
                    creators.push(creator.borrow().clone());
                    local_creators.push(creator.clone());
                }
            });

            if local_creators.is_empty() {
                println!("local creators is empty.break.");
                break;
            }

            local_inputs.clear();
            println!("clear local inputs");

            local_creators.iter_mut().for_each(|creator| {
                println!("local creators iter");
                creator
                    .borrow()
                    .inputs
                    .clone()
                    .unwrap()
                    .iter()
                    .for_each(|input| {
                        println!("creator inputs iter in local creators iter");
                        local_inputs.push(input.clone());
                    });
            });

            println!("check break");
            if local_inputs.is_empty() {
                break;
            }
        }
        creators
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
        let output_creator = output_result.borrow().creator.clone().unwrap();

        dbg!(output_creator.borrow().clone().creator);
        //dbg!(output_creator);

        //dbg!(add_exe.clone());
        dbg!(input1_result.clone());
        dbg!(input2_result.clone());
        //dbg!(output_result.clone());

        assert_eq!(Array::from_elem(IxDyn(&[]), rand_x1.clone()), input1_data);
        assert_eq!(Array::from_elem(IxDyn(&[]), rand_x2.clone()), input2_data);

        assert_eq!(expected_output_data.clone(), output_data.clone());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), output_grad.clone());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), input1_grad.clone());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), input2_grad.clone());
    }

    /// 2乗と加算のテスト
    /// (x1 + x2)^2 の順伝播と逆伝播をテストする。
    #[test]
    fn test_add_square_1() {
        // テスト用の入力値
        let x1_arr = Array::from_elem(IxDyn(&[]), 2.0);
        let x2_arr = Array::from_elem(IxDyn(&[]), 3.0);
        let x1 = Rc::new(RefCell::new(Variable::new(x1_arr.clone())));
        let x2 = Rc::new(RefCell::new(Variable::new(x2_arr.clone())));

        let expected = Array::from_elem(IxDyn(&[]), 25.0);

        // 関数を用意する。
        let mut sq_exe = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
        let mut add_exe = FunctionExecutor::new(Rc::new(RefCell::new(Add)));

        // 順伝播を実行する。
        let results = sq_exe.forward(add_exe.forward(vec![x1.clone(), x2.clone()]));

        // 順伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていないことも確認する。
        dbg!(x1.clone());
        dbg!(x2.clone());
        assert_eq!(x1_arr.clone(), x1.borrow().data.clone());
        assert_eq!(x2_arr.clone(), x2.borrow().data.clone());
        assert_eq!(None, x1.borrow().grad.clone());
        assert_eq!(None, x2.borrow().grad.clone());
        assert_eq!(
            expected.clone(),
            //results.clone().get(0).unwrap().borrow().clone().data
            results.get(0).unwrap().borrow().data.clone()
        );

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let creators = FunctionExecutor::extract_creators(results.clone());
        //dbg!(creators);
        creators.clone().iter_mut().for_each(|creator| {
            creator.backward();
        });

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        dbg!(x1.clone());
        dbg!(x2.clone());

        let expected_grad = Array::from_elem(IxDyn(&[]), 10.0);
        assert_eq!(x1_arr.clone(), x1.borrow().data.clone());
        assert_eq!(x2_arr.clone(), x2.borrow().data.clone());
        assert_eq!(expected_grad, x1.borrow().grad.clone().unwrap());
        assert_eq!(expected_grad, x2.borrow().grad.clone().unwrap());
        assert_eq!(
            expected.clone(),
            results.clone().get(0).unwrap().borrow().clone().data
        );
    }

    /// 2乗と加算のテスト
    /// x1^2 + x2^2 の順伝播と逆伝播をテストする。
    #[test]
    fn test_add_square_2() {
        // テスト用の入力値
        let x1_arr = Array::from_elem(IxDyn(&[]), 2.0);
        let x2_arr = Array::from_elem(IxDyn(&[]), 3.0);
        let x1 = Rc::new(RefCell::new(Variable::new(x1_arr.clone())));
        let x2 = Rc::new(RefCell::new(Variable::new(x2_arr.clone())));

        let expected = Array::from_elem(IxDyn(&[]), 13.0);

        // 関数を用意する。
        let mut sq_exe_1 = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
        let mut sq_exe_2 = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
        let mut add_exe = FunctionExecutor::new(Rc::new(RefCell::new(Add)));

        // 順伝播の実行
        let results = add_exe.forward(vec![
            sq_exe_1.forward(vec![x1.clone()]).get(0).unwrap().clone(),
            sq_exe_2.forward(vec![x2.clone()]).get(0).unwrap().clone(),
        ]);

        // 順伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていないことも確認する。
        dbg!(x1.clone());
        dbg!(x2.clone());
        assert_eq!(x1_arr.clone(), x1.borrow().data.clone());
        assert_eq!(x2_arr.clone(), x2.borrow().data.clone());
        assert_eq!(None, x1.borrow().grad.clone());
        assert_eq!(None, x2.borrow().grad.clone());
        assert_eq!(
            expected.clone(),
            //results.clone().get(0).unwrap().borrow().clone().data
            results.get(0).unwrap().borrow().data.clone()
        );

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let creators = FunctionExecutor::extract_creators(results.clone());
        //dbg!(creators);
        creators.clone().iter_mut().for_each(|creator| {
            creator.backward();
        });

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        dbg!(x1.clone());
        dbg!(x2.clone());

        let expected_x1_grad = Array::from_elem(IxDyn(&[]), 4.0);
        let expected_x2_grad = Array::from_elem(IxDyn(&[]), 6.0);
        assert_eq!(x1_arr.clone(), x1.borrow().data.clone());
        assert_eq!(x2_arr.clone(), x2.borrow().data.clone());
        assert_eq!(expected_x1_grad, x1.borrow().grad.clone().unwrap());
        assert_eq!(expected_x2_grad, x2.borrow().grad.clone().unwrap());
        assert_eq!(
            expected.clone(),
            results.clone().get(0).unwrap().borrow().clone().data
        );
    }

    // #[test]
    // fn test_add_square() {
    //     // 計算する値をランダムに生成する。
    //     let mut rng = rand::rng();
    //     let rand_x1 = rng.random::<f64>();
    //     let rand_x2 = rng.random::<f64>();
    //     let x1 = Rc::new(RefCell::new(Variable::new(rand_x1)));
    //     let x2 = Rc::new(RefCell::new(Variable::new(rand_x2)));

    //     // 加算した結果の期待値を計算する。
    //     let expected_output_data = Array::from_elem(IxDyn(&[]), rand_x1 + rand_x2);

    //     let mut sq1_exe = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
    //     let mut sq2_exe = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
    //     let mut add_exe = FunctionExecutor::new(Rc::new(RefCell::new(Add)));

    //     add_exe.forward(vec![
    //         sq1_exe.forward(vec![x1.clone()]),
    //         sq2_exe.forward(vec![x2.clone()]),
    //     ]);

    //     add_exe.forward(vec![x1.clone(), x2.clone()]);
    //     add_exe.backward();
    // }
}
