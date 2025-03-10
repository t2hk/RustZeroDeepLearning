//! ステップ17 メモリ管理と循環参照

use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use rand::random_iter;
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

/// Variable 構造体
/// * data (Array<f64, IxDyn>): 変数
/// * grad (Option<Array<f64, IxDyn>): 変数に対応した微分した値。逆伝播によって実際に微分が計算されたときに値を設定する。
/// * creator (Option<Rc<RefCell<FunctionExecutor>>>): この変数を生成した関数
/// * generation (i32): 計算グラフ上の世代
#[derive(Debug, Clone)]
struct Variable {
    data: Array<f64, IxDyn>,
    grad: Option<Array<f64, IxDyn>>,
    creator: Option<Rc<RefCell<FunctionExecutor>>>,
    generation: i32,
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
        self.creator = Some(Rc::clone(&creator));
        self.generation = creator.borrow().generation + 1;
    }

    /// 微分をリセットする。
    fn clear_grad(&mut self) {
        self.grad = None;
    }

    /// 変数の盛大を取得する。
    ///
    /// Return
    /// i32: 世代
    fn get_generation(&self) -> i32 {
        self.generation
    }

    /// 生成した関数の世代を取得する。
    ///
    /// Return
    /// i32: 生成した関数の世代
    fn get_creator_generation(&self) -> i32 {
        self.creator.clone().unwrap().borrow().generation
    }

    /// 値を取得する。
    ///
    /// Return
    /// * Array<f64, IxDyn>: 値
    fn get_data(&self) -> Array<f64, IxDyn> {
        self.data.clone()
    }

    /// 微分値を取得する。逆伝播を実行した場合のみ値が返る。
    ///
    /// Return
    /// * Array<f64, IxDyn>: 微分値
    fn get_grad(&self) -> Array<f64, IxDyn> {
        self.grad.clone().unwrap()
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
    inputs: Vec<Rc<RefCell<Variable>>>,    // 関数の入力値
    outputs: Vec<Weak<RefCell<Variable>>>, //関数の出力値
    creator: Rc<RefCell<dyn Function>>,    // 関数のトレイトオブジェクト
    generation: i32,                       // 関数の世代
}

/// 関数ラッパーの比較
/// オブジェクトのポインターが一致する場合、同一と判定する。
impl PartialEq for FunctionExecutor {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::addr_eq(self, other)
    }
}
impl Eq for FunctionExecutor {}

/// 関数ラッパーの優先度に基づいた大小比較。
impl PartialOrd for FunctionExecutor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FunctionExecutor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.get_generation().cmp(&other.get_generation())
    }
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
            inputs: vec![],
            outputs: vec![],
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
            let val = Variable::new(y_data.clone());
            let y = Rc::new(RefCell::new(val));

            outputs.push(Rc::clone(&y));
        }

        // 入出力を自身に設定する。
        self.inputs = inputs;
        self.outputs = outputs.iter().map(|output| Rc::downgrade(output)).collect();
        for output in outputs.clone().iter_mut() {
            output
                .borrow_mut()
                .set_creator(Rc::new(RefCell::new(self.clone())));
        }
        self.outputs
            .iter()
            .map(|output| output.upgrade().unwrap())
            .collect()
    }

    /// 逆伝播
    /// 自身で保持している出力値を使って逆伝播を実行する。
    fn backward(&mut self) {
        // 逆伝播の最初の関数の微分値として 1.0 を設定する。
        let grad_one = Array::from_elem(IxDyn(&[]), 1.0);
        let mut gys: Vec<Array<f64, IxDyn>> = vec![];
        dbg!(&self.outputs.get(0));
        self.outputs
            .iter_mut()
            .map(|output| output.upgrade().unwrap())
            .for_each(|output| {
                if output.borrow().grad.is_none() {
                    output.borrow_mut().grad = Some(grad_one.clone());
                }
                gys.push(output.borrow().grad.clone().unwrap());
            });

        // 逆伝播を実行する。
        let gxs = self.creator.borrow_mut().backward(self.inputs.clone(), gys);

        // 逆伝播の結果を入力値に設定する。
        // 入力値にすでに逆伝播による微分値が設定されている場合、加算する。
        for (i, input) in self.inputs.iter().enumerate() {
            if input.borrow_mut().grad.is_none() {
                input.borrow_mut().grad = Some(gxs[i].clone());
            } else {
                let input_grad = input.borrow().grad.clone().unwrap();
                input.borrow_mut().grad = Some(input_grad + gxs[i].clone());
            }
        }
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }

    /// 逆伝播のために計算グラフ上の関数を取得する。
    ///
    /// Arguments
    /// * outputs (Vec<Rc<RefCell<Variable>>>): 計算グラフの順伝播の出力値
    fn extract_creators(
        outputs: Vec<Rc<RefCell<Variable>>>,
    ) -> BinaryHeap<(i32, Rc<RefCell<FunctionExecutor>>)> {
        let mut creators = BinaryHeap::new();
        let mut creators_map: HashMap<String, &str> = HashMap::new();
        let mut local_variables: Vec<Rc<RefCell<Variable>>> = outputs.clone(); // 1 つの creator の入力値を保持する。

        // 計算グラフ上の creator を取得する。
        // creator の入力値を取得し、さらにその入力値の creator を取得することを繰り返す。
        // 取得した creator は creators ベクタに保存し、最終結果として返す。
        // 1 つの creator の入力値は local_variables ベクタに保存し、次のループ時にそれぞれ creator を取得する。
        loop {
            // 変数の creator を探す。
            let mut local_creators = vec![];
            local_variables.iter().for_each(|variable| {
                // すでに発見している creator は対象としないように、ハッシュマップで重複を排除する。重複の判断はポインタを使う。
                if let Some(creator) = variable.borrow().clone().creator {
                    if !creators_map.contains_key(&format!("{:p}", creator.as_ptr())) {
                        creators.push((creator.borrow().get_generation(), Rc::clone(&creator)));
                        creators_map.insert(format!("{:p}", creator.as_ptr()), "");
                        local_creators.push(Rc::clone(&creator));
                    }
                }
            });

            // creator が1つも見つからない場合、計算グラフの最初の入力値と判断して終了する。
            if local_creators.is_empty() {
                break;
            }

            // 見つけた creator の入力値を探し、local_variables ベクタに保存して次のループに備える。
            local_variables.clear();
            local_creators.iter_mut().for_each(|creator| {
                creator.borrow().inputs.iter().for_each(|input| {
                    local_variables.push(Rc::clone(input));
                });
            });
        }

        println!("heap len: {:?}", creators.len());
        for x in creators.iter() {
            println!("heap {:?},  {:?}", x.0, x.1.borrow().creator.borrow());
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

fn main() {}

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
        let a = square(Rc::clone(&x1));
        let b = square(Rc::clone(&a));
        let c = square(Rc::clone(&a));
        let d = add(Rc::clone(&b), Rc::clone(&c));
        let y = add(Rc::clone(&d), Rc::clone(&x2));

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

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let creators = FunctionExecutor::extract_creators(vec![y.clone()]);

        dbg!(&d);
        dbg!(&creators);

        // 実行した関数の数をチェックする。
        assert_eq!(5, creators.len());

        // 逆伝播を実行する。
        for (gen, creator) in creators.iter() {
            creator.borrow_mut().backward();
        }

        // 逆伝播の結果の確認
        assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), x1.borrow().get_data());
        assert_eq!(Array::from_elem(IxDyn(&[]), 64.0), x1.borrow().get_grad());
        assert_eq!(Array::from_elem(IxDyn(&[]), 3.0), x2.borrow().get_data());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), x2.borrow().get_grad());
        assert_eq!(Array::from_elem(IxDyn(&[]), 4.0), a.borrow().get_data());
        assert_eq!(Array::from_elem(IxDyn(&[]), 16.0), a.borrow().get_grad());
        assert_eq!(Array::from_elem(IxDyn(&[]), 16.0), b.borrow().get_data());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), b.borrow().get_grad());
        assert_eq!(Array::from_elem(IxDyn(&[]), 16.0), c.borrow().get_data());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), c.borrow().get_grad());
        assert_eq!(Array::from_elem(IxDyn(&[]), 32.0), d.borrow().get_data());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), d.borrow().get_grad());
        assert_eq!(Array::from_elem(IxDyn(&[]), 35.0), y.borrow().get_data());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), y.borrow().get_grad());
    }

    /// ステップ14に向けた事前確認用のテスト。
    #[test]
    fn test_add_same_input() {
        // 加算値をランダムに生成する。
        let x = Rc::new(RefCell::new(Variable::new(1.0)));

        // 加算した結果の期待値を計算する。
        let expected_output_data = Array::from_elem(IxDyn(&[]), 2.0);

        // 順伝播、逆伝播を実行する。
        let result = add(Rc::clone(&x), Rc::clone(&x));

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let creators = FunctionExecutor::extract_creators(vec![result.clone()]);

        dbg!(&creators);

        // 実行した関数の数をチェックする。
        assert_eq!(1, creators.len());

        // 逆伝播を実行する。
        for (gen, creator) in creators.iter() {
            creator.borrow_mut().backward();
        }

        // 足し算の結果
        assert_eq!(expected_output_data, result.borrow().data);
        // 逆伝播の結果
        assert_eq!(expected_output_data, x.borrow().get_grad());

        let input1_result = Rc::clone(
            &result
                .borrow()
                .creator
                .clone()
                .unwrap()
                .borrow()
                .inputs
                .get(0)
                .unwrap(),
        );
        let input2_result = Rc::clone(
            &result
                .borrow()
                .creator
                .clone()
                .unwrap()
                .borrow()
                .inputs
                .get(1)
                .unwrap(),
        );

        let output_result = Rc::clone(
            &result
                .borrow()
                .creator
                .clone()
                .unwrap()
                .borrow()
                .outputs
                .get(0)
                .unwrap()
                .upgrade()
                .unwrap(),
        );
        let input1_data = input1_result.borrow().get_data();
        let input2_data = input2_result.borrow().get_data();
        let input1_grad = input1_result.borrow().get_grad();
        let input2_grad = input2_result.borrow().get_grad();
        let output_data = output_result.borrow().get_data();
        let output_grad = output_result.borrow().get_grad();
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), input1_data);
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), input2_data);

        assert_eq!(expected_output_data.clone(), output_data.clone());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), output_grad.clone());
        // 入力値の微分結果が 1 になってしまうが、2が正しい。
        assert_ne!(Array::from_elem(IxDyn(&[]), 1.0), input1_grad.clone());
        assert_ne!(Array::from_elem(IxDyn(&[]), 1.0), input2_grad.clone());
        assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), input1_grad.clone());
        assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), input2_grad.clone());
        dbg!(&output_result);
    }

    /// ステップ14 同一の値を３回加算した場合のテスト。
    #[test]
    fn test_add_same_input_3times() {
        // 加算値をランダムに生成する。
        let x = Rc::new(RefCell::new(Variable::new(2.0)));

        // 加算した結果の期待値を計算する。
        let expected_output_data = Array::from_elem(IxDyn(&[]), 6.0);

        let result = add(add(Rc::clone(&x), Rc::clone(&x)), Rc::clone(&x));

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
        for (gen, creator) in creators.iter() {
            creator.borrow_mut().backward();
        }

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
        for (gen, creator) in creators.iter() {
            creator.borrow_mut().backward();
        }

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        dbg!(x.clone());

        let expected_grad = Array::from_elem(IxDyn(&[]), 2.0);
        assert_eq!(expected_grad, x.borrow().grad.clone().unwrap());
        assert_eq!(expected_output_data.clone(), result.borrow().clone().data);

        ////////////////////////////////
        // 微分をクリアせずにもう一度計算する。
        ////////////////////////////////
        let result2 = add(Rc::clone(&x), Rc::clone(&x));

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let creators2 = FunctionExecutor::extract_creators(vec![Rc::clone(&result2)]);
        //dbg!(creators);
        for (gen, creator) in creators.iter() {
            creator.borrow_mut().backward();
        }

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
        let result3 = add(Rc::clone(&x), Rc::clone(&x));

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let creators3 = FunctionExecutor::extract_creators(vec![Rc::clone(&result3)]);
        //dbg!(creators);
        for (gen, creator) in creators.iter() {
            creator.borrow_mut().backward();
        }

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
        let expected_grad_val = rand_x * 2.0 * 1.0;
        let expected_output_grad = Array::from_elem(IxDyn(&[]), expected_grad_val);

        // 順伝播、逆伝播を実行する。
        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let result = square(Rc::clone(&x));
        let creators = FunctionExecutor::extract_creators(vec![Rc::clone(&result)]);

        dbg!(&creators);

        // 実行した関数の数をチェックする。
        assert_eq!(1, creators.len());

        // 逆伝播を実行する。
        for (gen, creator) in creators.iter() {
            creator.borrow_mut().backward();
        }

        // 二乗の結果
        assert_eq!(expected_output_data, result.borrow().data);
        // 逆伝播の結果
        assert_eq!(expected_output_grad, x.borrow().get_grad());
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
        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        let result = add(Rc::clone(&x1), Rc::clone(&x2));
        let creators = FunctionExecutor::extract_creators(vec![Rc::clone(&result)]);

        dbg!(&creators);

        // 実行した関数の数をチェックする。
        assert_eq!(1, creators.len());

        // 逆伝播を実行する。
        for (gen, creator) in creators.iter() {
            creator.borrow_mut().backward();
        }

        // 足し算の結果
        assert_eq!(expected_output_data, result.borrow().data);
        // 逆伝播の結果
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), x1.borrow().get_grad());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), x2.borrow().get_grad());
    }

    /// Exp 関数のテスト。
    #[test]
    fn test_exp() {
        let x = Rc::new(RefCell::new(Variable::new(2.0)));

        let e = std::f64::consts::E;
        let expected_output_data = Array::from_elem(IxDyn(&[]), e.powf(2.0));
        dbg!(expected_output_data.clone());

        // 順伝播、逆伝播を実行する。
        let result = exp(Rc::clone(&x));
        let creators = FunctionExecutor::extract_creators(vec![Rc::clone(&result)]);

        dbg!(&creators);

        // 実行した関数の数をチェックする。
        assert_eq!(1, creators.len());

        // 逆伝播を実行する。
        for (gen, creator) in creators.iter() {
            creator.borrow_mut().backward();
        }

        // exp 結果
        assert_eq!(expected_output_data, result.borrow().data);
        // 逆伝播の結果 exp^x の微分は exp^x
        assert_eq!(expected_output_data, x.borrow().get_grad());

        assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), x.borrow().get_data());
        assert_eq!(
            Array::from_elem(IxDyn(&[]), 1.0),
            result.borrow().get_grad()
        );
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
        for (gen, creator) in creators.iter() {
            creator.borrow_mut().backward();
        }

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
        for (gen, creator) in creators.iter() {
            creator.borrow_mut().backward();
        }

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
        for (gen, creator) in creators.iter() {
            creator.borrow_mut().backward();
        }

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
