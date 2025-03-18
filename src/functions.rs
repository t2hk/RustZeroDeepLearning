use crate::settings::*;
use crate::variable::*;

use ndarray::{array, Array, ArrayD, IntoDimension, IxDyn};
use num_traits::{Num, NumCast};
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

/// Function トレイト
pub trait Function<V>: std::fmt::Debug
where
    V: MathOps,
{
    /// 順伝播
    /// 通常の計算を行う順伝播。継承して実装すること。
    ///
    /// Arguments
    /// * xs (Vec<Array<f64, IxDyn>>): 入力値
    ///
    /// Returns
    /// * Vec<Array<f64, IxDyn>>: 出力値
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>>;

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
        inputs: Vec<Rc<RefCell<Variable<V>>>>,
        gys: Vec<Array<V, IxDyn>>,
    ) -> Vec<Array<V, IxDyn>>;
}

/// 関数の実行用ラッパー
/// 関数の入出力値と関数のトレイトオブジェクトを保持し、順伝播、逆伝播を呼び出す。
#[derive(Debug, Clone)]
pub struct FunctionExecutor<V: MathOps> {
    inputs: Vec<Rc<RefCell<Variable<V>>>>,    // 関数の入力値
    outputs: Vec<Weak<RefCell<Variable<V>>>>, //関数の出力値
    creator: Rc<RefCell<dyn Function<V>>>,    // 関数のトレイトオブジェクト
    generation: i32,                          // 関数の世代
}

/// 関数ラッパーの比較
/// オブジェクトのポインターが一致する場合、同一と判定する。
impl<V: MathOps> PartialEq for FunctionExecutor<V> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::addr_eq(self, other)
    }
}
impl<V: MathOps> Eq for FunctionExecutor<V> {}

/// 関数ラッパーの優先度に基づいた大小比較。
impl<V: MathOps> PartialOrd for FunctionExecutor<V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<V: MathOps> Ord for FunctionExecutor<V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.get_generation().cmp(&other.get_generation())
    }
}

impl<V: MathOps> FunctionExecutor<V> {
    /// コンストラクタ
    ///
    /// Arguments
    /// * creator (Rc<RefCell<dyn Function>>): 関数のトレイトオブジェクト
    ///
    /// Return
    /// * FunctionExecutor: 関数のラッパー
    pub fn new(creator: Rc<RefCell<dyn Function<V>>>) -> FunctionExecutor<V> {
        FunctionExecutor {
            inputs: vec![],
            outputs: vec![],
            creator: creator,
            generation: 0,
        }
    }

    /// 世代を取得する。
    ///
    /// Return
    /// * i32: 世代
    pub fn get_generation(&self) -> i32 {
        self.generation
    }

    /// 順伝播
    ///
    /// Arguments
    /// * inputs (Vec<Rc<RefCell<Variable>>>): 関数の入力値
    ///
    /// Return
    /// * Vec<Rc<RefCell<Variable>>>: 関数の実行結果
    pub fn forward(
        &mut self,
        inputs: Vec<Rc<RefCell<Variable<V>>>>,
    ) -> Vec<Rc<RefCell<Variable<V>>>> {
        // 入力値からデータを取り出す。
        let xs_data: Vec<Array<V, IxDyn>> = inputs
            .iter()
            .map(|input| input.borrow().get_data().clone())
            .collect();

        // 逆伝播を有効にする場合、世代を設定する。
        if Setting::is_enable_backprop() {
            self.generation = inputs
                .iter()
                .map(|input| input.borrow().get_generation())
                .max()
                .unwrap_or(0);
        }

        // 関数を実行する。
        let ys_data = self.creator.borrow().forward(xs_data);

        // 関数の結果を出力値とする。
        let mut outputs: Vec<Rc<RefCell<Variable<V>>>> = ys_data
            .into_iter()
            .map(|y_data| {
                let val = Variable::new(y_data);
                Rc::new(RefCell::new(val))
            })
            .collect();

        // 入出力を自身に設定する。
        self.inputs = inputs;
        self.outputs = outputs.iter().map(|output| Rc::downgrade(output)).collect();
        for output in &outputs {
            output
                .borrow_mut()
                .set_creator(Rc::new(RefCell::new(self.clone())));
        }
        outputs
    }

    /// 逆伝播
    /// 自身で保持している出力値を使って逆伝播を実行する。
    pub fn backward(&self) {
        // 逆伝播の最初の関数の微分値として 1 を設定する。
        let grad_one = Array::from_elem(IxDyn(&[]), V::one());
        let mut gys: Vec<Array<V, IxDyn>> = vec![];
        self.outputs
            .iter()
            .map(|output| output.upgrade().unwrap())
            .for_each(|output| {
                // if output.borrow().grad.is_none() {
                if output.borrow().get_grad().is_none() {
                    output.borrow_mut().set_grad(grad_one.clone());
                }
                gys.push(output.borrow().get_grad().clone().unwrap());
            });

        // 逆伝播を実行する。
        let gxs = self.creator.borrow_mut().backward(self.inputs.clone(), gys);

        // 逆伝播の結果を入力値に設定する。
        // 入力値にすでに逆伝播による微分値が設定されている場合、加算する。
        for (i, input) in self.inputs.iter().enumerate() {
            if input.borrow_mut().get_grad().is_none() {
                input.borrow_mut().set_grad(gxs[i].clone());
            } else {
                let input_grad = input.borrow().get_grad().clone().unwrap();
                input.borrow_mut().set_grad(input_grad + gxs[i].clone());
            }
        }

        // 微分値を保持しない場合、中間変数の微分値を削除する。
        if !Setting::is_enable_retain_grad() {
            self.outputs
                .iter()
                .map(|output| output.upgrade().unwrap())
                .for_each(|output| {
                    output.borrow_mut().clear_grad();
                });
        }
    }

    pub fn get_generation(&self) -> i32 {
        self.generation
    }

    /// 逆伝播のために計算グラフ上の関数を取得する。
    ///
    /// Arguments
    /// * outputs (Vec<Rc<RefCell<Variable>>>): 計算グラフの順伝播の出力値
    pub fn extract_creators(
        outputs: Vec<Rc<RefCell<Variable<V>>>>,
    ) -> BinaryHeap<(i32, Rc<RefCell<FunctionExecutor<V>>>)> {
        let mut creators = BinaryHeap::new();
        let mut creators_map: HashMap<String, &str> = HashMap::new();
        let mut local_variables: Vec<Rc<RefCell<Variable<V>>>> = outputs.clone(); // 1 つの creator の入力値を保持する。

        // 計算グラフ上の creator を取得する。
        // creator の入力値を取得し、さらにその入力値の creator を取得することを繰り返す。
        // 取得した creator は creators ベクタに保存し、最終結果として返す。
        // 1 つの creator の入力値は local_variables ベクタに保存し、次のループ時にそれぞれ creator を取得する。
        loop {
            // 変数の creator を探す。
            let mut local_creators = vec![];
            local_variables.iter().for_each(|variable| {
                // すでに発見している creator は対象としないように、ハッシュマップで重複を排除する。重複の判断はポインタを使う。
                if let Some(creator) = variable.borrow().get_creator() {
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

    /// 順伝播の結果から逆伝播を一括で実行する。
    ///
    /// Arguments:
    /// * outputs (Vec<Rc<RefCell<Variable>>>): 順伝播の結果
    fn backward_all(outputs: Vec<Rc<RefCell<Variable<V>>>>) {
        let creators = FunctionExecutor::extract_creators(outputs);
        // 逆伝播を実行する。
        for (_gen, creator) in creators.iter() {
            creator.borrow_mut().backward();
        }
    }
}
