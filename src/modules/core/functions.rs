// ライブラリを一括でインポート
use crate::modules::*;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

/// Function トレイト
pub trait Function<V>: std::fmt::Debug
where
    V: MathOps,
{
    fn get_name(&self) -> String;

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
    /// * inputs (Vec<Rc<RefCell<RawData>>>): 順伝播の入力値
    /// * gys (Vec<Variable<V>>): 出力値に対する微分値
    ///
    /// Returns
    /// * Vec<Variable<V>>: 入力値に対する微分値
    fn backward(
        &self,
        inputs: Vec<Variable<V>>,
        outputs: Vec<Weak<RefCell<RawData<V>>>>,
        gys: Vec<Variable<V>>,
    ) -> Vec<Variable<V>>;
}

/// 関数の実行用ラッパー
/// 関数の入出力値と関数のトレイトオブジェクトを保持し、順伝播、逆伝播を呼び出す。
#[derive(Debug, Clone)]
pub struct FunctionExecutor<V: MathOps> {
    inputs: Vec<Variable<V>>,                // 関数の入力値
    outputs: Vec<Weak<RefCell<RawData<V>>>>, //関数の出力値
    creator: Rc<RefCell<dyn Function<V>>>,   // 関数のトレイトオブジェクト
    generation: i32,                         // 関数の世代
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

/// 関数ラッパーの世代に基づいた大小比較。
impl<V: MathOps> Ord for FunctionExecutor<V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.get_generation().cmp(&other.get_generation())
    }
}

/// 関数ラッパーの実装
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

    pub fn detail(&self) -> String {
        let inputs_detail: Vec<String> = self
            .inputs
            .iter()
            .map(|input| format!("{:?}", input.get_data()))
            .collect();

        let outputs_detail: Vec<String> = self
            .outputs
            .iter()
            .map(|output| {
                if let Some(output) = output.upgrade() {
                    format!("{:?}", output.borrow().get_data())
                } else {
                    format!("None")
                }
            })
            .collect();

        let detail = format!(
            "creator: {}, generation: {}, inputs: {:?}, outputs: {:?}",
            self.creator.borrow().get_name(),
            self.generation,
            inputs_detail,
            outputs_detail
        );
        detail.to_string()
    }

    /// 入力値を取得する。
    ///
    /// Return
    /// * Vec<Variable<V>>: 関数に対する入力値のベクタ
    pub fn get_inputs(&self) -> Vec<Variable<V>> {
        self.inputs.clone()
    }

    /// 出力値を取得する。
    ///
    /// Return
    /// * Vec<Weak<RefCell<Variable<V>>>>: 関数の出力値のベクタ
    pub fn get_outputs(&self) -> Vec<Weak<RefCell<RawData<V>>>> {
        self.outputs.clone()
    }

    /// 世代を取得する。
    ///
    /// Return
    /// * i32: 世代
    pub fn get_generation(&self) -> i32 {
        self.generation
    }

    /// 関数のオブジェクトを取得する。
    ///
    /// Return
    /// * Rc<RefCell<dyn Function<V>>>
    pub fn get_creator(&self) -> Rc<RefCell<dyn Function<V>>> {
        self.creator.clone()
    }

    /// 順伝播
    ///
    /// Arguments
    /// * inputs (Vec<Variable<V>>): 関数の入力値
    ///
    /// Return
    /// * Vec<Variable<V>>: 関数の実行結果
    pub fn forward(&mut self, inputs: Vec<Variable<V>>) -> Vec<Variable<V>> {
        info!("[forward {}]", &self.creator.borrow().get_name());

        // 入力値からデータを取り出す。
        let xs_data: Vec<Array<V, IxDyn>> =
            inputs.iter().map(|variable| variable.get_data()).collect();

        // 逆伝播を有効にする場合、世代を設定する。
        if Setting::is_enable_backprop() {
            self.generation = inputs
                .iter()
                .map(|variable| variable.get_generation())
                .max()
                .unwrap_or(0);
        }

        // 関数を実行する。
        let ys_data = self.creator.borrow().forward(xs_data);

        // 関数の結果を出力値とする。
        let outputs: Vec<Variable<V>> = ys_data
            .into_iter()
            .map(|y_data| {
                let val = RawData::new(y_data);
                Variable::new(val)
            })
            .collect();

        // 入出力を自身に設定する。
        self.inputs = inputs.into_iter().map(|input| input).collect();
        self.outputs = outputs
            .iter()
            .map(|output| Rc::downgrade(&output.raw()))
            .collect();

        for output in &outputs {
            output.set_creator(Rc::new(RefCell::new(self.clone())));
        }

        outputs
    }

    /// 逆伝播
    /// 自身で保持している出力値を使って逆伝播を実行する。
    pub fn backward(&self) {
        info!(
            "[backward: {:?} gen:{}]",
            &self.creator.borrow().get_name(),
            &self.generation
        );

        // 逆伝播の最初の関数の微分値として 1 を設定する。
        let mut gys: Vec<Variable<V>> = vec![];

        self.outputs
            .iter()
            .map(|output| output.upgrade().unwrap())
            .for_each(|output| {
                if output.borrow().get_grad().is_none() {
                    let grad_one = Variable::new(RawData::new(Array::ones(
                        output.borrow().get_data().shape(),
                    )));
                    output.borrow_mut().set_grad(grad_one);
                }

                gys.push(output.borrow().get_grad().clone().unwrap());
            });

        // if Setting::is_enable_backprop() {
        // 逆伝播を実行する。
        let gxs = self
            .creator
            .borrow()
            .backward(self.inputs.clone(), self.outputs.clone(), gys);

        // 逆伝播の結果を入力値に設定する。
        // 入力値にすでに逆伝播による微分値が設定されている場合、加算する。
        for (i, input) in self.inputs.iter().enumerate() {
            let mut new_grad = gxs[i].clone();

            if input.get_grad().is_some() {
                let input_grad = input.get_grad().clone().unwrap();
                new_grad = &input_grad + &new_grad;
            }
            input.set_grad(new_grad);
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
        //}
    }

    /// 逆伝播のために計算グラフ上の関数を取得する。
    ///
    /// Arguments
    /// * outputs (Vec<Rc<RefCell<RawData>>>): 計算グラフの順伝播の出力値
    pub fn extract_creators(
        outputs: Vec<Variable<V>>,
    ) -> BinaryHeap<(i32, Rc<RefCell<FunctionExecutor<V>>>)> {
        let mut creators = BinaryHeap::new();
        let mut creators_map: HashMap<String, &str> = HashMap::new();
        let mut local_variables: Vec<Variable<V>> = outputs.clone(); // 1 つの creator の入力値を保持する。

        // 計算グラフ上の creator を取得する。
        // creator の入力値を取得し、さらにその入力値の creator を取得することを繰り返す。
        // 取得した creator は creators ベクタに保存し、最終結果として返す。
        // 1 つの creator の入力値は local_variables ベクタに保存し、次のループ時にそれぞれ creator を取得する。
        loop {
            // 変数の creator を探す。
            let mut local_creators = vec![];
            local_variables.iter().for_each(|variable| {
                // すでに発見している creator は対象としないように、ハッシュマップで重複を排除する。重複の判断はポインタを使う。
                if let Some(creator) = variable.get_creator() {
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
                    local_variables.push(input.clone());
                });
            });
        }
        creators
    }
}
