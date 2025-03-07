//! ステップ15 複雑な計算グラフ(理論編)
//! ステップ16 複雑な計算グラフ(実装編)

use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use rand::random_iter;
use std::cell::RefCell;
use std::rc::Rc;

/// Variable 構造体
/// * data (Array<f64, IxDyn>): 変数
/// * grad (Option<Array<f64, IxDyn>): 変数に対応した微分した値。逆伝播によって実際に微分が計算されたときに値を設定する。
/// * creator (Option<Rc<RefCell<FunctionExecutor>>>): この変数を生成した関数
/// * generation (i64): 計算グラフ上の世代
#[derive(Debug, Clone)]
struct Variable {
    data: Array<f64, IxDyn>,
    grad: Option<Array<f64, IxDyn>>,
    creator: Option<Rc<RefCell<FunctionExecutor>>>,
    generation: i64,
}

impl Variable {
    /// Variable のコンストラクタ。
    ///
    /// # Arguments
    /// * data - 変数    
    fn new<T: CreateVariable>(data: T) -> Variable {
        CreateVariable::create_variable(&data)
    }

    /// この変数を生成した関数を設定する。
    ///
    /// Arguments
    /// * creator (Rc<RefCell<FunctionExecutor>>): 関数のラッパー
    fn set_creator(&mut self, creator: Rc<RefCell<FunctionExecutor>>) {
        self.creator = Some(creator.clone());
        self.generation = creator.borrow().generation + 1;
    }

    /// 微分をリセットする。
    fn clear_grad(&mut self) {
        self.grad = None;
    }

    /// 変数の盛大を取得する。
    ///
    /// Return
    /// i64: 世代
    fn get_generation(&self) -> i64 {
        self.generation
    }

    /// 生成した関数の世代を取得する。
    ///
    /// Return
    /// i64: 生成した関数の世代
    fn get_creator_generation(&self) -> i64 {
        self.creator.clone().unwrap().borrow().generation
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
            generation: 0,
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
            generation: 0,
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
    /// * inputs (Vec<Rc<RefCell<Variable>>>): 順伝播の入力値
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

/// 関数の実行用ラッパー
/// 関数の入出力値と関数のトレイトオブジェクトを保持し、順伝播、逆伝播を呼び出す。
#[derive(Debug, Clone)]
struct FunctionExecutor {
    inputs: Option<Vec<Rc<RefCell<Variable>>>>, // 関数の入力値
    outputs: Option<Vec<Rc<RefCell<Variable>>>>, //関数の出力値
    creator: Rc<RefCell<dyn Function>>,         // 関数のトレイトオブジェクト
    generation: i64,                            // 関数の世代
}
impl FunctionExecutor {
    /// コンストラクタ
    ///
    /// Arguments
    /// * creator (Rc<RefCell<dyn Function>>): 関数のトレイトオブジェクト
    ///
    /// Return
    /// * FunctionExecutor: 関数のラッパー
    fn new(creator: Rc<RefCell<dyn Function>>) -> FunctionExecutor {
        FunctionExecutor {
            inputs: None,
            outputs: None,
            creator: creator,
            generation: 0,
        }
    }

    /// 順伝播
    ///
    /// Arguments
    /// * inputs (Vec<Rc<RefCell<Variable>>>): 関数の入力値
    ///
    /// Return
    /// * Vec<Rc<RefCell<Variable>>>: 関数の実行結果
    fn forward(&mut self, inputs: Vec<Rc<RefCell<Variable>>>) -> Vec<Rc<RefCell<Variable>>> {
        // 入力値からデータを取り出す。
        let xs_data: Vec<Array<f64, IxDyn>> = inputs
            .iter()
            .map(|input| input.borrow().data.clone())
            .collect();
        self.generation = inputs
            .iter()
            .map(|input| input.borrow().generation.clone())
            .max()
            .unwrap();

        // 関数を実行する。
        let ys_data = self.creator.borrow().forward(xs_data);

        // 関数の結果を出力値とする。
        let mut outputs: Vec<Rc<RefCell<Variable>>> = Vec::new();
        for y_data in ys_data.iter() {
            let mut val = Variable::new(y_data.clone());
            // val.creator = Some(self.creator.clone());
            // val.creator = Some(Rc::new(RefCell::new(self.clone())));
            val.set_creator(Rc::new(RefCell::new(self.clone())));
            let y = Rc::new(RefCell::new(val));

            outputs.push(Rc::clone(&y));
        }

        // 入出力を自身に設定する。
        self.inputs = Some(inputs);
        self.outputs = Some(outputs.clone());
        for output in outputs.clone().iter_mut() {
            //output.borrow_mut().creator = Some(Rc::new(RefCell::new(self.clone())));
            output
                .borrow_mut()
                .set_creator(Rc::new(RefCell::new(self.clone())));
        }
        self.outputs.clone().unwrap()
    }

    /// 逆伝播
    /// 自身で保持している出力値を使って逆伝播を実行する。
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

        // 逆伝播を実行する。
        let gxs = self
            .creator
            .borrow_mut()
            .backward(self.inputs.clone().unwrap(), gys);

        // 逆伝播の結果を入力値に設定する。
        // 入力値にすでに逆伝播による微分値が設定されている場合、加算する。
        for (i, input) in self.inputs.clone().unwrap().iter().enumerate() {
            if input.borrow_mut().grad.is_none() {
                input.borrow_mut().grad = Some(gxs[i].clone());
            } else {
                let input_grad = input.borrow().grad.clone().unwrap();
                input.borrow_mut().grad = Some(input_grad + gxs[i].clone());
            }
        }
    }

    fn get_generation(&self) -> i64 {
        self.generation
    }

    /// 逆伝播のために計算グラフ上の関数を取得する。
    ///
    /// Arguments
    /// * outputs (Vec<Rc<RefCell<Variable>>>): 計算グラフの順伝播の出力値
    fn extract_creators(outputs: Vec<Rc<RefCell<Variable>>>) -> Vec<FunctionExecutor> {
        let mut creators = vec![]; // 計算グラフの creator を保持する。
        let mut local_variables = outputs.clone(); // 1 つの creator の入力値を保持する。

        // 計算グラフ上の creator を取得する。
        // creator の入力値を取得し、さらにその入力値の creator を取得することを繰り返す。
        // 取得した creator は creators ベクタに保存し、最終結果として返す。
        // 1 つの creator の入力値は local_variables ベクタに保存し、次のループ時にそれぞれ creator を取得する。
        loop {
            // 変数の creator を探す。
            let mut local_creators = vec![];
            local_variables.iter().for_each(|variable| {
                if let Some(creator) = variable.borrow().clone().creator {
                    creators.push(creator.borrow().clone());
                    local_creators.push(creator.clone());
                }
            });

            // creator が1つも見つからない場合、計算グラフの最初の入力値と判断して終了する。
            if local_creators.is_empty() {
                break;
            }

            // 見つけた creator の入力値を探し、local_variables ベクタに保存して次のループに備える。
            local_variables.clear();
            local_creators.iter_mut().for_each(|creator| {
                creator
                    .borrow()
                    .inputs
                    .clone()
                    .unwrap()
                    .iter()
                    .for_each(|input| {
                        local_variables.push(input.clone());
                    });
            });
        }
        creators
    }
}

/// 二乗関数
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

/// 二乗関数
///
/// Arguments
/// * input (Rc<RefCell<Variable>>): 加算する変数
///
/// Return
/// * Rc<RefCell<Variable>>: 二乗の結果
fn square(input: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let mut square = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
    // 二乗の順伝播
    square.forward(vec![input]).get(0).unwrap().clone()
}

/// 加算関数
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
        _inputs: Vec<Rc<RefCell<Variable>>>,
        gys: Vec<Array<f64, IxDyn>>,
    ) -> Vec<Array<f64, IxDyn>> {
        vec![gys[0].clone(), gys[0].clone()]
    }
}

/// 加算関数
///
/// Arguments
/// * x1 (Rc<RefCell<Variable>>): 加算する変数
/// * x2 (Rc<RefCell<Variable>>): 加算する変数
///
/// Return
/// * Rc<RefCell<Variable>>: 加算結果
fn add(x1: Rc<RefCell<Variable>>, x2: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let mut add = FunctionExecutor::new(Rc::new(RefCell::new(Add)));
    // 加算の順伝播
    add.forward(vec![x1.clone(), x2.clone()])
        .get(0)
        .unwrap()
        .clone()
}

/// Exp 関数
#[derive(Debug, Clone)]
struct Exp;
impl Function for Exp {
    // Exp (y=e^x) の順伝播
    fn forward(&self, xs: Vec<Array<f64, IxDyn>>) -> Vec<Array<f64, IxDyn>> {
        let e = std::f64::consts::E;

        let result = vec![xs[0].mapv(|x| e.powf(x))];
        result
    }

    /// 逆伝播
    /// dy/dx=e^x である。
    fn backward(
        &self,
        inputs: Vec<Rc<RefCell<Variable>>>,
        gys: Vec<Array<f64, IxDyn>>,
    ) -> Vec<Array<f64, IxDyn>> {
        let x = inputs[0].borrow().data.clone();
        let e = std::f64::consts::E;
        let gxs = vec![x.mapv(|x| e.powf(x)) * gys[0].clone()];
        gxs
    }
}

/// Exp 関数
///
/// Arguments
/// * input (Rc<RefCell<Variable>>): 入力値
///
/// Return
/// * Rc<RefCell<Variable>>: 結果
fn exp(input: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let mut exp = FunctionExecutor::new(Rc::new(RefCell::new(Exp)));
    // EXP の順伝播
    exp.forward(vec![input.clone()]).get(0).unwrap().clone()
}

fn main() {
    let x1: Rc<RefCell<Variable>> = Rc::new(RefCell::new(Variable::new(2.0)));
    let x2: Rc<RefCell<Variable>> = Rc::new(RefCell::new(Variable::new(3.0)));

    let square = Square;
    let mut square_exe = FunctionExecutor::new(Rc::new(RefCell::new(square)));

    square_exe.forward(vec![x1.clone()]);
    square_exe.backward();

    // dbg!(square_exe);
    // dbg!(x1.borrow());
    // dbg!(x2.borrow());

    let add = Add;
    let mut add_exe = FunctionExecutor::new(Rc::new(RefCell::new(add)));
    add_exe.forward(vec![x1.clone(), x2.clone()]);
    add_exe.backward();
    // dbg!(add_exe);
    // dbg!(x1.borrow());
    // dbg!(x2.borrow());
}

#[cfg(test)]
mod tests {
    use super::*;
    // use approx::assert_abs_diff_eq;
    use rand::prelude::*;

    // 世代に関するテスト。
    // x1 -> x1^2 -> a -> a^2 -> b -> b+c -> d -> d+x2 -> y
    //               -> a^2 -> c /          x2
    #[test]
    fn test_generations() {
        let x1 = Rc::new(RefCell::new(Variable::new(2.0)));
        let x2 = Rc::new(RefCell::new(Variable::new(3.0)));
        let a = square(x1.clone());
        let b = square(a.clone());
        let c = square(a.clone());
        let d = add(b.clone(), c.clone());
        let y = add(d.clone(), x2.clone());

        // 順伝播の結果
        assert_eq!(Array::from_elem(IxDyn(&[]), 35.0), y.borrow().data.clone());
        // 各変数の世代のテスト
        assert_eq!(0, x1.borrow().get_generation());
        assert_eq!(0, x2.borrow().get_generation());
        assert_eq!(1, a.borrow().get_generation());
        assert_eq!(2, b.borrow().get_generation());
        assert_eq!(2, c.borrow().get_generation());
        assert_eq!(3, d.borrow().get_generation());
        assert_eq!(4, y.borrow().get_generation());

        // 各関数の世代のテスト
        assert_eq!(0, a.borrow().get_creator_generation());
        assert_eq!(1, b.borrow().get_creator_generation());
        assert_eq!(1, c.borrow().get_creator_generation());
        assert_eq!(2, d.borrow().get_creator_generation());
        assert_eq!(3, y.borrow().get_creator_generation());
    }

    /// ステップ14に向けた事前確認用のテスト。
    #[test]
    fn test_add_same_input() {
        // 加算値をランダムに生成する。
        let x = Rc::new(RefCell::new(Variable::new(1.0)));

        // 加算した結果の期待値を計算する。
        let expected_output_data = Array::from_elem(IxDyn(&[]), 2.0);

        // 順伝播、逆伝播を実行する。
        let add = Add;
        let mut add_exe = FunctionExecutor::new(Rc::new(RefCell::new(add)));

        add_exe.forward(vec![x.clone(), x.clone()]);
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

        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), input1_data);
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), input2_data);

        assert_eq!(expected_output_data.clone(), output_data.clone());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), output_grad.clone());
        // 入力値の微分結果が 1 になってしまうが、2が正しい。
        assert_ne!(Array::from_elem(IxDyn(&[]), 1.0), input1_grad.clone());
        assert_ne!(Array::from_elem(IxDyn(&[]), 1.0), input2_grad.clone());
        assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), input1_grad.clone());
        assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), input2_grad.clone());
        dbg!(x.clone());
    }

    /// ステップ14 同一の値を３回加算した場合のテスト。
    #[test]
    fn test_add_same_input_3times() {
        // 加算値をランダムに生成する。
        let x = Rc::new(RefCell::new(Variable::new(2.0)));

        // 加算した結果の期待値を計算する。
        let expected_output_data = Array::from_elem(IxDyn(&[]), 6.0);

        let result = add(add(x.clone(), x.clone()), x.clone());

        // 順伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていないことも確認する。
        dbg!(x.clone());
        assert_eq!(
            expected_output_data.clone(),
            //results.clone().get(0).unwrap().borrow().clone().data
            // results.get(0).unwrap().borrow().data.clone()
            result.borrow().data.clone()
        );

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let creators = FunctionExecutor::extract_creators(vec![result.clone()]);
        //dbg!(creators);
        creators.clone().iter_mut().for_each(|creator| {
            creator.backward();
        });

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        dbg!(x.clone());

        let expected_grad = Array::from_elem(IxDyn(&[]), 3.0);
        assert_eq!(expected_grad, x.borrow().grad.clone().unwrap());
        assert_eq!(expected_output_data.clone(), result.borrow().clone().data);
    }

    /// ステップ14 微分のクリアに関するテスト
    #[test]
    fn test_clear_grad() {
        // 加算値を生成する。
        let x = Rc::new(RefCell::new(Variable::new(2.0)));

        // 加算した結果の期待値を計算する。
        let expected_output_data = Array::from_elem(IxDyn(&[]), 4.0);

        let result = add(x.clone(), x.clone());

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let creators = FunctionExecutor::extract_creators(vec![result.clone()]);
        //dbg!(creators);
        creators.clone().iter_mut().for_each(|creator| {
            creator.backward();
        });

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        dbg!(x.clone());

        let expected_grad = Array::from_elem(IxDyn(&[]), 2.0);
        assert_eq!(expected_grad, x.borrow().grad.clone().unwrap());
        assert_eq!(expected_output_data.clone(), result.borrow().clone().data);

        ////////////////////////////////
        // 微分をクリアせずにもう一度計算する。
        ////////////////////////////////
        let result2 = add(x.clone(), x.clone());

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let creators2 = FunctionExecutor::extract_creators(vec![result2.clone()]);
        //dbg!(creators);
        creators2.clone().iter_mut().for_each(|creator| {
            creator.backward();
        });

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        dbg!(x.clone());

        // 1回目の微分と２回目の微分を加算した ４ になってしまうことを確認する。
        let expected_grad2 = Array::from_elem(IxDyn(&[]), 4.0);
        assert_eq!(expected_grad2, x.borrow().grad.clone().unwrap());
        assert_eq!(expected_output_data.clone(), result2.borrow().clone().data);

        ////////////////////////////////
        // 微分をクリアしてもう一度計算する。
        ////////////////////////////////
        x.borrow_mut().clear_grad();
        let result3 = add(x.clone(), x.clone());

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let creators3 = FunctionExecutor::extract_creators(vec![result3.clone()]);
        //dbg!(creators);
        creators3.clone().iter_mut().for_each(|creator| {
            creator.backward();
        });

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        dbg!(x.clone());
        // 微分をクリアしたことで正しい結果となることを確認する。
        let expected_grad3 = Array::from_elem(IxDyn(&[]), 2.0);
        assert_eq!(expected_grad3, x.borrow().grad.clone().unwrap());
        assert_eq!(expected_output_data.clone(), result2.borrow().clone().data);
    }

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

        //dbg!(square_exe.clone());
        // dbg!(input_result.clone());
        // dbg!(output_result.clone());

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

    /// Exp 関数のテスト。
    #[test]
    fn test_exp() {
        let x = Rc::new(RefCell::new(Variable::new(2.0)));

        let e = std::f64::consts::E;
        let expected = Array::from_elem(IxDyn(&[]), e.powf(2.0));
        dbg!(expected.clone());

        // 順伝播、逆伝播を実行する。
        let exp = Exp;
        let mut exp_exe = FunctionExecutor::new(Rc::new(RefCell::new(exp)));
        exp_exe.forward(vec![x.clone()]);
        exp_exe.backward();

        // 順伝播と逆伝播の処理結果を取得する。
        let input_result = exp_exe.clone().inputs.unwrap().get(0).unwrap().clone();
        let output_result = exp_exe.clone().outputs.unwrap().get(0).unwrap().clone();

        let input_data = input_result.borrow().data.clone();
        let input_grad = input_result.borrow().grad.clone().unwrap();
        let output_data = output_result.borrow().data.clone();
        let output_grad = output_result.borrow().grad.clone().unwrap();
        let output_creator = output_result.borrow().creator.clone().unwrap();

        dbg!(output_creator.borrow().clone().creator);
        dbg!(input_result.clone());

        assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), input_data);
        assert_eq!(expected.clone(), output_data.clone());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), output_grad.clone());
        assert_eq!(expected.clone(), input_grad.clone());
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
        // let mut sq_exe = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
        // let mut add_exe = FunctionExecutor::new(Rc::new(RefCell::new(Add)));

        // 順伝播を実行する。
        // let results = sq_exe.forward(add_exe.forward(vec![x1.clone(), x2.clone()]));
        let result = square(add(x1.clone(), x2.clone()));

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
            // results.get(0).unwrap().borrow().data.clone()
            result.borrow().data.clone()
        );

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let creators = FunctionExecutor::extract_creators(vec![result.clone()]);
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
        assert_eq!(expected.clone(), result.borrow().clone().data);
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
        // let mut sq_exe_1 = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
        // let mut sq_exe_2 = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
        // let mut add_exe = FunctionExecutor::new(Rc::new(RefCell::new(Add)));

        // 順伝播の実行
        // let results = add_exe.forward(vec![
        //     sq_exe_1.forward(vec![x1.clone()]).get(0).unwrap().clone(),
        //     sq_exe_2.forward(vec![x2.clone()]).get(0).unwrap().clone(),
        // ]);
        let result = add(square(x1.clone()), square(x2.clone()));

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
            result.borrow().data.clone()
        );

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let creators = FunctionExecutor::extract_creators(vec![result.clone()]);
        //dbg!(creators);
        creators.clone().iter_mut().for_each(|creator| {
            dbg!(creator.generation.clone());
            creator.backward();
        });

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        dbg!(x1.clone());
        dbg!(x2.clone());
        dbg!(result.clone().borrow().generation);

        let expected_x1_grad = Array::from_elem(IxDyn(&[]), 4.0);
        let expected_x2_grad = Array::from_elem(IxDyn(&[]), 6.0);
        assert_eq!(x1_arr.clone(), x1.borrow().data.clone());
        assert_eq!(x2_arr.clone(), x2.borrow().data.clone());
        assert_eq!(expected_x1_grad, x1.borrow().grad.clone().unwrap());
        assert_eq!(expected_x2_grad, x2.borrow().grad.clone().unwrap());
        assert_eq!(expected.clone(), result.borrow().clone().data);
    }

    /// 2乗と加算のテスト
    /// x1^2 + x2^2 の順伝播と逆伝播をテストする。
    #[test]
    fn test_square_exp_square() {
        // テスト用の入力値
        let x_arr = Array::from_elem(IxDyn(&[]), 0.5);
        let x = Rc::new(RefCell::new(Variable::new(x_arr.clone())));

        let e = std::f64::consts::E;
        let expected = Array::from_elem(IxDyn(&[]), e.powf(0.5 * 0.5) * e.powf(0.5 * 0.5));
        dbg!(expected.clone());

        // 関数を用意する。
        // let mut sq_exe_1 = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
        // let mut exp_exe = FunctionExecutor::new(Rc::new(RefCell::new(Exp)));
        // let mut sq_exe_2 = FunctionExecutor::new(Rc::new(RefCell::new(Square)));

        // 順伝播の実行
        // let results = sq_exe_2.forward(exp_exe.forward(sq_exe_1.forward(vec![x.clone()])));
        let result = square(exp(square(x.clone())));

        // 順伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていないことも確認する。
        dbg!(x.clone());
        assert_eq!(x_arr.clone(), x.borrow().data.clone());
        assert_eq!(None, x.borrow().grad.clone());
        assert_eq!(expected.clone(), result.borrow().data.clone());

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let creators = FunctionExecutor::extract_creators(vec![result.clone()]);
        //dbg!(creators);
        creators.clone().iter_mut().for_each(|creator| {
            creator.backward();
        });

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        dbg!(x.clone());

        // 逆伝播の正解は書籍の値を使用。
        let expected_x_grad = Array::from_elem(IxDyn(&[]), 3.297442541400256);

        assert_eq!(x_arr.clone(), x.borrow().data.clone());
        assert_eq!(expected_x_grad, x.borrow().grad.clone().unwrap());
        assert_eq!(expected.clone(), result.borrow().clone().data);
    }
}
