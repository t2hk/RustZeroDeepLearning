//! ステップ20 演算子のオーバーロード(1)

use core::fmt::Debug;
use ndarray::{array, Array, ArrayD, IntoDimension, IxDyn};
use num_traits::{Num, NumCast};
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

pub trait MathOps: Num + NumCast + Copy {}
impl<V> MathOps for V where V: Num + NumCast + Copy + Clone {}

thread_local!(
  static SETTING: Rc<RefCell<Setting>> = {
      Rc::new(RefCell::new(Setting { enable_backprop: true, retain_grad: false }))
  }
);

struct Setting {
    enable_backprop: bool,
    retain_grad: bool,
}
impl Setting {
    /// 逆伝播を有効にする。
    fn set_backprop_enabled() {
        SETTING.with(|setting| {
            let mut s = setting.borrow_mut();
            s.enable_backprop = true;
        });
    }

    /// 逆伝播を行わない。
    fn set_backprop_disabled() {
        SETTING.with(|setting| {
            let mut s = setting.borrow_mut();
            s.enable_backprop = false;
        });
    }

    /// 逆伝播が有効かどうか。
    ///
    /// Return
    /// * bool: 逆伝播が有効な場合は true を返す。
    /// デフォルトは true である。
    fn is_enable_backprop() -> bool {
        SETTING.with(|setting| setting.borrow().enable_backprop)
    }

    /// メモリ削減のため、逆伝播の中間変数について微分値を保持する。
    /// 初期値は false (微分値を保持しない)
    fn set_retain_grad_enabled() {
        SETTING.with(|setting| {
            let mut s = setting.borrow_mut();
            s.retain_grad = true;
        });
    }

    /// 中間変数の微分を保持しない(デフォルト)。
    fn set_retain_grad_disabled() {
        SETTING.with(|setting| {
            let mut s = setting.borrow_mut();
            s.retain_grad = false;
        });
    }

    /// 中間変数の微分を保持するかどうか。
    ///
    /// Return
    /// * bool: 中間変数の微分を保持する場合は true を返す。
    /// デフォルトは false である。
    fn is_enable_retain_grad() -> bool {
        SETTING.with(|setting| setting.borrow().retain_grad)
    }
}

/// Variable 構造体
/// * data (Array<f64, IxDyn>): 変数
/// * name (Option<String>): 変数の名前
/// * grad (Option<Array<f64, IxDyn>): 変数に対応した微分した値。逆伝播によって実際に微分が計算されたときに値を設定する。
/// * creator (Option<Rc<RefCell<FunctionExecutor>>>): この変数を生成した関数
/// * generation (i32): 計算グラフ上の世代
#[derive(Debug, Clone)]
struct Variable<V: MathOps> {
    data: Array<V, IxDyn>,
    name: Option<String>,
    grad: Option<Array<V, IxDyn>>,
    creator: Option<Rc<RefCell<FunctionExecutor<V>>>>,
    generation: i32,
}

impl<V: MathOps> Variable<V> {
    /// Variable のコンストラクタ。
    ///
    /// # Arguments
    /// * data - 変数    
    fn new<T: CreateVariable<V>>(data: T) -> Variable<V> {
        CreateVariable::create_variable(&data)
    }

    /// Variable を次元と値から生成する。
    /// 以下のように使用する。
    ///   let dim = vec![2, 2, 2];
    ///   let values = vec![1., 2., 3., 4., 5., 6., 7., 8.];
    ///   let variable = Variable::new(dim, values);
    ///
    /// Arguments
    /// * shape (Vec<i32>): 次元
    /// * values (Vec<f64>): 変数
    ///
    /// Returns
    /// * Result<Self, ShapeError>
    fn from_shape_vec<Sh>(shape: Sh, values: Vec<V>) -> Self
    where
        Sh: IntoDimension<Dim = IxDyn>,
    {
        let dim = shape.into_dimension();
        let array = ArrayD::from_shape_vec(dim, values).expect("Shape error while creating array");
        Self {
            data: array,
            name: None,
            grad: None,
            creator: None,
            generation: 0,
        }
    }

    /// この変数を生成した関数を設定する。
    ///
    /// Arguments
    /// * creator (Rc<RefCell<FunctionExecutor>>): 関数のラッパー
    fn set_creator(&mut self, creator: Rc<RefCell<FunctionExecutor<V>>>) {
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
    fn get_data(&self) -> Array<V, IxDyn> {
        self.data.clone()
    }

    /// 変数の名前を取得する。
    ///
    /// Return
    /// * String: 名前
    fn get_name(&self) -> Option<String> {
        self.name.clone()
    }

    /// 微分値を取得する。逆伝播を実行した場合のみ値が返る。
    ///
    /// Return
    /// * Array<f64, IxDyn>: 微分値
    fn get_grad(&self) -> Array<V, IxDyn> {
        self.grad.clone().unwrap()
    }

    /// 要素の数
    fn get_size(&self) -> usize {
        self.data.len()
    }

    /// 次元ごとの要素数
    fn get_shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// 次元数
    fn get_ndim(&self) -> usize {
        self.data.ndim()
    }

    /// 型
    fn get_dtype(&self) -> String {
        format!("{}", std::any::type_name::<V>())
    }

    /// この変数を出力結果とした場合の逆伝播を行う。
    fn backward(&self) {
        let creators =
            FunctionExecutor::extract_creators(vec![Rc::new(RefCell::new(self.clone()))]);
        for (_gen, creator) in creators.iter() {
            creator.borrow().backward();
        }
    }
}

/// Variable 構造体を生成するためのトレイト
/// * create_variable: Variable 構造体を生成する
trait CreateVariable<V: MathOps> {
    fn create_variable(&self) -> Variable<V>;
}

/// CreateVariable トレイトの Array<f64, IxDyn> 用の実装
impl<V: MathOps> CreateVariable<V> for Array<V, IxDyn> {
    fn create_variable(&self) -> Variable<V> {
        Variable {
            data: self.clone(),
            name: None,
            grad: None,
            creator: None,
            generation: 0,
        }
    }
}

/// CreateVariable トレイトの 数値用の実装
impl<V: MathOps> CreateVariable<V> for V {
    fn create_variable(&self) -> Variable<V> {
        Variable {
            data: Array::from_elem(IxDyn(&[]), *self),
            name: None,
            grad: None,
            creator: None,
            generation: 0,
        }
    }
}

/// Function トレイト
trait Function<V>: std::fmt::Debug
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
struct FunctionExecutor<V: MathOps> {
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
    fn new(creator: Rc<RefCell<dyn Function<V>>>) -> FunctionExecutor<V> {
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
    fn forward(&mut self, inputs: Vec<Rc<RefCell<Variable<V>>>>) -> Vec<Rc<RefCell<Variable<V>>>> {
        // 入力値からデータを取り出す。
        let xs_data: Vec<Array<V, IxDyn>> = inputs
            .iter()
            .map(|input| input.borrow().data.clone())
            .collect();

        // 逆伝播を有効にする場合、世代を設定する。
        if Setting::is_enable_backprop() {
            self.generation = inputs
                .iter()
                .map(|input| input.borrow().generation)
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
    fn backward(&self) {
        // 逆伝播の最初の関数の微分値として 1 を設定する。
        let grad_one = Array::from_elem(IxDyn(&[]), V::one());
        let mut gys: Vec<Array<V, IxDyn>> = vec![];
        self.outputs
            .iter()
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

        // 微分値を保持しない場合、中間変数の微分値を削除する。
        if !Setting::is_enable_retain_grad() {
            self.outputs
                .iter()
                .map(|output| output.upgrade().unwrap())
                .for_each(|output| {
                    output.borrow_mut().grad = None;
                });
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

/// 二乗関数
#[derive(Debug, Clone)]
struct Square;
impl<V: MathOps> Function<V> for Square {
    /// 順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        //let result = vec![xs[0].mapv(|x| x.pow(V::from(2).unwrap()))];
        let result = vec![xs[0].mapv(|x| x * x)];

        //dpg!(result);

        result
    }

    /// 逆伝播
    /// y=x^2 の微分であるため、dy/dx=2x である。
    fn backward(
        &self,
        inputs: Vec<Rc<RefCell<Variable<V>>>>,
        gys: Vec<Array<V, IxDyn>>,
    ) -> Vec<Array<V, IxDyn>> {
        let x = inputs[0].borrow().data.clone();
        // let gxs = vec![V::from(2).unwrap() * &x * &gys[0].clone()];
        let x_gys = &gys[0].clone() * &x;
        let gxs = vec![x_gys.mapv(|x| x * V::from(2).unwrap())];
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
fn square<V: MathOps>(input: Rc<RefCell<Variable<V>>>) -> Rc<RefCell<Variable<V>>> {
    let mut square = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
    // 二乗の順伝播
    square.forward(vec![input]).get(0).unwrap().clone()
}

/// 加算関数
#[derive(Debug, Clone)]
struct Add;
impl<V: MathOps> Function<V> for Add {
    // Add (加算) の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        let result = vec![&xs[0] + &xs[1]];
        result
    }

    /// 逆伝播
    /// y=x0+x1 の微分であるため、dy/dx0=1, dy/dx1=1 である。
    fn backward(
        &self,
        _inputs: Vec<Rc<RefCell<Variable<V>>>>,
        gys: Vec<Array<V, IxDyn>>,
    ) -> Vec<Array<V, IxDyn>> {
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
fn add<V: MathOps>(
    x1: Rc<RefCell<Variable<V>>>,
    x2: Rc<RefCell<Variable<V>>>,
) -> Rc<RefCell<Variable<V>>> {
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
impl<V: MathOps> Function<V> for Exp {
    // Exp (y=e^x) の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        let e = std::f64::consts::E;
        // let result = vec![xs[0].mapv(|x| e.powf(x.to_f64().unwrap()))];

        // let result = vec![xs[0].mapv(|x| V::from(x.to_f64().unwrap().exp()).unwrap())];
        let result = vec![xs[0].mapv(|x| V::from(e.powf(x.to_f64().unwrap())).unwrap())];

        result
    }

    /// 逆伝播
    /// dy/dx=e^x である。
    fn backward(
        &self,
        inputs: Vec<Rc<RefCell<Variable<V>>>>,
        gys: Vec<Array<V, IxDyn>>,
    ) -> Vec<Array<V, IxDyn>> {
        let e = std::f64::consts::E;
        let x = inputs[0].borrow().data.clone();
        let gys_val = gys[0].clone();
        let x_exp = vec![x.mapv(|x| V::from(e.powf(x.to_f64().unwrap())).unwrap())];
        let gxs = x_exp.iter().map(|x_exp| x_exp * &gys_val).collect();
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
fn exp<V: MathOps>(input: Rc<RefCell<Variable<V>>>) -> Rc<RefCell<Variable<V>>> {
    let mut exp = FunctionExecutor::new(Rc::new(RefCell::new(Exp)));
    // EXP の順伝播
    exp.forward(vec![input.clone()]).get(0).unwrap().clone()
}

fn type_of<T>(_: T) -> String {
    let a = std::any::type_name::<T>();
    return a.to_string();
}

/// 乗算関数
#[derive(Debug, Clone)]
struct Mul;
impl<V: MathOps> Function<V> for Mul {
    // Mul (乗算) の順伝播
    fn forward(&self, xs: Vec<Array<V, IxDyn>>) -> Vec<Array<V, IxDyn>> {
        let result = vec![&xs[0] * &xs[1]];
        result
    }

    /// 逆伝播
    /// y=x1 * x2 の微分であるため、dy/dx1=x2 * gy, dy/dx2= x1 * gy である。
    fn backward(
        &self,
        inputs: Vec<Rc<RefCell<Variable<V>>>>,
        gys: Vec<Array<V, IxDyn>>,
    ) -> Vec<Array<V, IxDyn>> {
        let x1 = inputs[0].borrow().data.clone();
        let x2 = inputs[1].borrow().data.clone();
        let gx_x1 = &gys[0].clone() * &x2;
        let gx_x2 = &gys[0].clone() * &x1;

        let gxs = vec![gx_x1, gx_x2];
        gxs
    }
}

/// 乗算関数
///
/// Arguments
/// * x1 (Rc<RefCell<Variable>>): 乗算する変数
/// * x2 (Rc<RefCell<Variable>>): 乗算する変数
///
/// Return
/// * Rc<RefCell<Variable>>: 乗算結果
fn mul<V: MathOps>(
    x1: Rc<RefCell<Variable<V>>>,
    x2: Rc<RefCell<Variable<V>>>,
) -> Rc<RefCell<Variable<V>>> {
    let mut mul = FunctionExecutor::new(Rc::new(RefCell::new(Mul)));
    // 乗算の順伝播
    mul.forward(vec![x1.clone(), x2.clone()])
        .get(0)
        .unwrap()
        .clone()
}

fn main() {
    let x1 = Rc::new(RefCell::new(Variable::new(1.0)));
    let x2 = Rc::new(RefCell::new(Variable::new(1.0)));

    let dtypetest = Variable::new(10i32);
    println!("dtype: {:?}", dtypetest.get_dtype());

    let result = add(
        Rc::clone(&x1),
        Rc::clone(&add(Rc::clone(&x1), Rc::clone(&x2))),
    );
    // let creators = FunctionExecutor::extract_creators(vec![Rc::clone(&result)]);
    // // 逆伝播を実行する。
    // for (gen, creator) in creators.iter() {
    //     creator.borrow_mut().backward();
    //}
    FunctionExecutor::backward_all(vec![Rc::clone(&result)]);
    println!(
        "result grad: {:?}, x1 grad: {:?}, x2 grad: {:?}",
        &result.borrow().grad,
        &x1.borrow().grad,
        &x2.borrow().grad
    );

    let arr_vals = vec![
        10., 11., 12., 13., 14., 15., 16., 20., 21., 22., 23., 24., 25., 26., 30., 31., 32., 33.,
        34., 35., 36., 40., 41., 42., 43., 44., 45., 46., 50., 51., 52., 53., 54., 55., 56.,
    ];
    let arr_dim = vec![5, 7];
    let arr_var = Variable::from_shape_vec(arr_dim, arr_vals);
    dbg!(&arr_var.data);
    dbg!(&arr_var.data.ndim());
    dbg!(&arr_var.data.view());

    let arr = array!(
        [12, 12, 12, 12, 12, 12, 12,],
        [12, 12, 12, 12, 12, 12, 12,],
        [12, 12, 12, 12, 12, 12, 12,],
        [12, 12, 12, 12, 12, 12, 12,],
        [2, 2, 2, 2, 2, 2, 2,],
    );
    dbg!(&arr);
    dbg!(type_of(&arr));
    println!("arr shape: {:?})", arr.shape());
    println!("arr len: {:?})", arr.len());
    println!("arr view: {:?})", arr.view());
    println!("arr ndim: {:?})", arr.ndim());

    let shape = vec![2, 2, 2];
    let values = vec![1., 2., 3., 4., 5., 6., 7., 8.];
    let arr2 = Variable::from_shape_vec(shape, values);
    dbg!(&arr2.data);
    println!("arr2 shape: {:?})", arr2.data.shape());
    println!("arr2 len: {:?})", arr2.data.len());
    println!("arr2 ndim: {:?})", arr2.data.ndim());

    let arr_2x3x4 = ndarray::array![
        [[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]],
        [
            [13., 14., 15., 16.],
            [17., 18., 19., 20.],
            [21., 22., 23., 24.]
        ]
    ];
    let arr_4x2x3 = ndarray::array![
        [[1., 2., 3.], [4., 5., 6.]],
        [[7., 8., 9.], [10., 11., 12.]],
        [[13., 14., 15.], [16., 17., 18.]],
        [[19., 20., 21.], [22., 23., 24.]]
    ];

    dbg!(&arr_2x3x4);
    dbg!(&arr_4x2x3);
}

#[cfg(test)]
mod tests {
    use super::*;
    // use approx::assert_abs_diff_eq;
    use rand::prelude::*;

<<<<<<< HEAD
=======
    /// Variable に実装した逆伝播メソッドのテスト。
    #[test]
    fn test_add_mul2() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        // 順伝播
        let a = Rc::new(RefCell::new(Variable::new(3.0f32)));
        let b = Rc::new(RefCell::new(Variable::new(2.0f32)));
        let c = Rc::new(RefCell::new(Variable::new(1.0f32)));
        let expected = Variable::new(7f32);

        let result = add(mul(Rc::clone(&a), Rc::clone(&b)), Rc::clone(&c));
        assert_eq!(expected.get_data(), result.borrow().get_data());

        // 逆伝播
        result.as_ref().clone().borrow().backward();

        println!(
            "result grad: {:?}, a grad: {:?}, b grad: {:?}, c grad: {:?}",
            &result.borrow().grad,
            &a.borrow().grad,
            &b.borrow().grad,
            &c.borrow().grad,
        );

        assert_eq!(
            Array::from_elem(IxDyn(&[]), 1.0),
            result.borrow().get_grad()
        );
        assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), a.borrow().get_grad());
        assert_eq!(Array::from_elem(IxDyn(&[]), 3.0), b.borrow().get_grad());
    }

    /// 複数の演算を結合した場合のオーバーロードのテスト
    /// Weak 参照が解放されてしまい、微分値を保持できず失敗する。
    /// 対応を検討中。
    // #[test]
    // fn test_overload_add_mul() {
    //     // 逆伝播を実行する。微分値を保持する。
    //     Setting::set_retain_grad_enabled();

    //     // バックプロパゲーションを行う。
    //     Setting::set_backprop_enabled();

    //     // 順伝播
    //     let a = Variable::new(3.0f32);
    //     let b = Variable::new(2.0f32);
    //     let c = Variable::new(1.0f32);
    //     let expected = Variable::new(7f32);

    //     //let result = Rc::new(RefCell::new(a.clone() * b.clone() + c.clone()));
    //     let result = Rc::new(RefCell::new(a.clone() * b.clone() + c.clone()));

    //     //assert_eq!(expected.get_data(), result.clone().get_data());

    //     // 逆伝播
    //     // FunctionExecutor::backward_all(vec![Rc::clone(&result)]);
    //     // println!(
    //     //     "result grad: {:?}, a grad: {:?}, b grad: {:?}, c grad: {:?}",
    //     //     &result.borrow().grad,
    //     //     &a.grad,
    //     //     &b.grad,
    //     //     &c.grad,
    //     // );
    //     result.as_ref().clone().borrow().backward();

    //     assert_eq!(
    //         Array::from_elem(IxDyn(&[]), 1.0),
    //         &result.borrow().get_grad()
    //     );
    //     assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), a.get_grad());
    //     assert_eq!(Array::from_elem(IxDyn(&[]), 3.0), b.get_grad());
    // }

    /// 乗算のオーバーロードのテスト
    #[test]
    fn test_mul_overload() {
        let a = Variable::new(3.0f32);
        let b = Variable::new(2.0f32);
        let result = a * b;
        let expected = Variable::new(6.0f32);
        dbg!(&result);
        assert_eq!(expected.get_data(), result.get_data());
    }

    /// 加算のオーバーロードのテスト
    #[test]
    fn test_add_overload() {
        let a = Variable::new(3.0f32);
        let b = Variable::new(2.0f32);
        let result = a + b;
        let expected = Variable::new(5.0f32);
        dbg!(&result);
        assert_eq!(expected.get_data(), result.get_data());
    }

>>>>>>> 078d94f (ちょっとしたリファクタリング)
    #[test]
    fn test_add_mul() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        // 順伝播
        let a = Rc::new(RefCell::new(Variable::new(3.0f32)));
        let b = Rc::new(RefCell::new(Variable::new(2.0f32)));
        let c = Rc::new(RefCell::new(Variable::new(1.0f32)));
        let expected = Variable::new(7f32);

        let result = add(mul(Rc::clone(&a), Rc::clone(&b)), Rc::clone(&c));
        assert_eq!(expected.get_data(), result.borrow().get_data());

        // 逆伝播
        //FunctionExecutor::backward_all(vec![Rc::clone(&result)]);
        result.as_ref().clone().borrow().backward();

        println!(
            "result grad: {:?}, a grad: {:?}, b grad: {:?}, c grad: {:?}",
            &result.borrow().grad,
            &a.borrow().grad,
            &b.borrow().grad,
            &c.borrow().grad,
        );

        assert_eq!(
            Array::from_elem(IxDyn(&[]), 1.0),
            result.borrow().get_grad()
        );
        assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), a.borrow().get_grad());
        assert_eq!(Array::from_elem(IxDyn(&[]), 3.0), b.borrow().get_grad());
    }

    #[test]
    /// 乗算のテスト(f32)
    fn test_mul_2() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        // 順伝播
        let x1 = Rc::new(RefCell::new(Variable::new(5.0f32)));
        let x2 = Rc::new(RefCell::new(Variable::new(10.0f32)));
        let expected = Variable::new(50.0f32);

        let result = mul(Rc::clone(&x1), Rc::clone(&x2));
        assert_eq!(expected.get_data(), result.borrow().get_data());

        // 逆伝播
        //FunctionExecutor::backward_all(vec![Rc::clone(&result)]);
        result.as_ref().clone().borrow().backward();

        println!(
            "result grad: {:?}, x1 grad: {:?}, x2 grad: {:?}",
            &result.borrow().grad,
            &x1.borrow().grad,
            &x2.borrow().grad
        );

        assert_eq!(
            Array::from_elem(IxDyn(&[]), 1.0),
            result.borrow().get_grad()
        );
        assert_eq!(Array::from_elem(IxDyn(&[]), 10.0), x1.borrow().get_grad());
        assert_eq!(Array::from_elem(IxDyn(&[]), 5.0), x2.borrow().get_grad());
    }

    #[test]
    /// 乗算のテスト(i32)
    fn test_mul_1() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // バックプロパゲーションを行う。
        Setting::set_backprop_enabled();

        // 順伝播
        let x1 = Rc::new(RefCell::new(Variable::new(5i32)));
        let x2 = Rc::new(RefCell::new(Variable::new(10i32)));
        let expected = Variable::new(50);

        let result = mul(Rc::clone(&x1), Rc::clone(&x2));
        assert_eq!(expected.get_data(), result.borrow().get_data());

        // 逆伝播
        //FunctionExecutor::backward_all(vec![Rc::clone(&result)]);
        result.as_ref().clone().borrow().backward();

        println!(
            "result grad: {:?}, x1 grad: {:?}, x2 grad: {:?}",
            &result.borrow().grad,
            &x1.borrow().grad,
            &x2.borrow().grad
        );

        assert_eq!(Array::from_elem(IxDyn(&[]), 1), result.borrow().get_grad());
        assert_eq!(Array::from_elem(IxDyn(&[]), 10), x1.borrow().get_grad());
        assert_eq!(Array::from_elem(IxDyn(&[]), 5), x2.borrow().get_grad());
    }

    #[test]
    /// 変数の型名に関するテスト。
    fn test_get_dtype() {
        let var_i8 = Variable::new(10i8);
        let var_i16 = Variable::new(10i16);
        let var_i32 = Variable::new(10i32);
        let var_i64 = Variable::new(10i64);
        let var_f32 = Variable::new(10.0f32);
        let var_f64 = Variable::new(10.0f64);
        let var_u8 = Variable::new(10u8);
        let var_u16 = Variable::new(10u16);
        let var_u32 = Variable::new(10u32);
        let var_u64 = Variable::new(10u64);
        assert_eq!("i8", var_i8.get_dtype());
        assert_eq!("i16", var_i16.get_dtype());
        assert_eq!("i32", var_i32.get_dtype());
        assert_eq!("i64", var_i64.get_dtype());
        assert_eq!("f32", var_f32.get_dtype());
        assert_eq!("f64", var_f64.get_dtype());
        assert_eq!("u8", var_u8.get_dtype());
        assert_eq!("u16", var_u16.get_dtype());
        assert_eq!("u32", var_u32.get_dtype());
        assert_eq!("u64", var_u64.get_dtype());
    }

    /// 中間変数の微分結果を保持し無い場合のテスト
    #[test]
    fn test_retain_grad_disabled_u32() {
        let x1 = Rc::new(RefCell::new(Variable::new(2u32)));
        let x2 = Rc::new(RefCell::new(Variable::new(3u32)));
        let a = square(Rc::clone(&x1));
        let b = square(Rc::clone(&a));
        let c = square(Rc::clone(&a));
        let d = add(Rc::clone(&b), Rc::clone(&c));
        let y = add(Rc::clone(&d), Rc::clone(&x2));

        // 順伝播の結果
        assert_eq!(Array::from_elem(IxDyn(&[]), 35), y.borrow().data.clone());
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

        // dbg!(&d);
        // dbg!(&creators);

        // 実行した関数の数をチェックする。
        assert_eq!(5, creators.len());

        // 逆伝播を実行する。微分値を保持しない。
        Setting::set_retain_grad_disabled();
        for (_gen, creator) in creators.iter() {
            creator.borrow_mut().backward();
        }

        // 逆伝播の結果の確認
        // 途中結果の変数には微分値が設定されていないことを確認する。
        assert_eq!(Array::from_elem(IxDyn(&[]), 2), x1.borrow().get_data());
        assert_eq!(Array::from_elem(IxDyn(&[]), 64), x1.borrow().get_grad());
        assert_eq!(Array::from_elem(IxDyn(&[]), 3), x2.borrow().get_data());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1), x2.borrow().get_grad());
        assert_eq!(Array::from_elem(IxDyn(&[]), 4), a.borrow().get_data());
        assert!(a.borrow().grad.is_none());
        assert_eq!(Array::from_elem(IxDyn(&[]), 16), b.borrow().get_data());
        assert!(b.borrow().grad.is_none());
        assert_eq!(Array::from_elem(IxDyn(&[]), 16), c.borrow().get_data());
        assert!(c.borrow().grad.is_none());
        assert_eq!(Array::from_elem(IxDyn(&[]), 32), d.borrow().get_data());
        assert!(d.borrow().grad.is_none());
        assert_eq!(Array::from_elem(IxDyn(&[]), 35), y.borrow().get_data());
        assert!(y.borrow().grad.is_none());
    }

    /// 変数の size, shape, ndim のテスト
    #[test]
    fn test_variable_params() {
        let var0 = Variable::new(1.0);
        assert_eq!(1, var0.get_size());
        let a: [usize; 0] = [];
        assert_eq!(&a, var0.get_shape());
        assert_eq!(0, var0.get_ndim());

        let var1 = Variable::from_shape_vec(vec![1], vec![1.0]);
        assert_eq!(1, var1.get_size());
        assert_eq!([1], var1.get_shape());
        assert_eq!(1, var1.get_ndim());

        let sh2x2 = vec![2, 2];
        let val2x2 = vec![1., 2., 3., 4.];
        let var2x2 = Variable::from_shape_vec(sh2x2, val2x2);

        assert_eq!(4, var2x2.get_size());
        assert_eq!([2, 2], var2x2.get_shape());
        assert_eq!(2, var2x2.get_ndim());
        dbg!(&var2x2.get_shape());

        let sh10x20x30x40x50 = vec![10, 20, 30, 40, 50];
        let val10x20x30x40x50: Vec<f64> = (1..=12000000).map(|x| x as f64).collect();

        let var10x20x30x40x50 = Variable::from_shape_vec(sh10x20x30x40x50, val10x20x30x40x50);
        assert_eq!(12000000, var10x20x30x40x50.get_size());
        assert_eq!([10, 20, 30, 40, 50], var10x20x30x40x50.get_shape());
        assert_eq!(5, var10x20x30x40x50.get_ndim());
    }

    /// 2乗と加算のテスト
    /// (x1 + x2)^2 の順伝播と逆伝播をテストする。
    #[test]
    fn test_multidim_add_square_1() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // テスト用の入力値
        let sh1 = vec![2, 2];
        let val1 = vec![1., 2., 3., 4.];
        let var1 = Variable::from_shape_vec(sh1, val1);
        dbg!(&var1);
        let sh2 = vec![2, 2];
        let val2 = vec![11., 12., 13., 14.];
        let var2 = Variable::from_shape_vec(sh2, val2);
        dbg!(&var2);

        let x1 = Rc::new(RefCell::new(var1));
        let x2 = Rc::new(RefCell::new(var2));

        // 順伝播の結果 [[12., 14.],[16., 18.]]^2 = [[144., 196.], [256., 324.]]
        let expected = Variable::from_shape_vec(vec![2, 2], vec![144., 196., 256., 324.]);
        // 逆伝播の結果 2 * [[12., 14.], [16., 18.]]
        let expected_grad = Variable::from_shape_vec(vec![2, 2], vec![24., 28., 32., 36.]);

        let result = square(add(x1.clone(), x2.clone()));

        // 順伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていないことも確認する。
        dbg!(x1.clone());
        dbg!(x2.clone());
        assert_eq!(None, x1.borrow().grad.clone());
        assert_eq!(None, x2.borrow().grad.clone());
        assert_eq!(expected.data, result.borrow().data.clone());

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        // let creators = FunctionExecutor::extract_creators(vec![result.clone()]);
        // //dbg!(creators);
        // for (_gen, creator) in creators.iter() {
        //     creator.borrow_mut().backward();
        // }
        result.as_ref().clone().borrow().backward();

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        dbg!(x1.clone());
        dbg!(x2.clone());
        assert_eq!(expected_grad.get_data(), x1.borrow().grad.clone().unwrap());
        assert_eq!(expected_grad.get_data(), x2.borrow().grad.clone().unwrap());
    }

    #[test]
    fn test_multidim_square() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

        // 入力値の準備
        let sh1 = vec![2, 2];
        let arr1 = vec![1., 2., 3., 4.];
        let var1 = Variable::from_shape_vec(sh1, arr1);
        let x1 = Rc::new(RefCell::new(var1.clone()));

        let expected_var = Variable::from_shape_vec(vec![2, 2], vec![1., 4., 9., 16.]);
        let expected_grad = Variable::from_shape_vec(vec![2, 2], vec![2., 4., 6., 8.]);

        // 順伝播、逆伝播を実行する。
        let result = square(Rc::clone(&x1));
        assert_eq!(&expected_var.get_data(), &result.borrow().get_data());

        result.as_ref().clone().borrow().backward();
        // dbg!(&result.borrow().get_grad());
        // dbg!(&x1.borrow().get_grad());
        assert_eq!(&expected_grad.get_data(), &x1.borrow().get_grad());
    }

    #[test]
    fn test_multidim_add() {
        let sh1 = vec![2, 2];
        let val1 = vec![1., 2., 3., 4.];
        let var1 = Variable::from_shape_vec(sh1, val1);
        dbg!(&var1);
        let sh2 = vec![2, 2];
        let val2 = vec![11., 12., 13., 14.];
        let var2 = Variable::from_shape_vec(sh2, val2);
        // dbg!(&var2);

        // 加算値をランダムに生成する。
        let x1 = Rc::new(RefCell::new(var1));
        let x2 = Rc::new(RefCell::new(var2));

        let expected_var = Variable::from_shape_vec(vec![2, 2], vec![12., 14., 16., 18.]);

        // 加算した結果の期待値を計算する。
        // let expected_output_data = Array::from_elem(IxDyn(&[]), 2.0);

        // 順伝播を実行する。
        let result = add(Rc::clone(&x1), Rc::clone(&x2));
        assert_eq!(&expected_var.get_data(), &result.borrow().get_data());

        // dbg!(&result.borrow().get_data());
    }

    #[test]
    /// 任意の形状に関するテスト。
    fn test_dyndim_array() {
        let shape = vec![2, 2, 2];
        let values = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let var = Variable::from_shape_vec(shape, values);
        // dbg!(&var);
        assert_eq!(&[2, 2, 2], var.get_data().shape());
    }

    /// 変数の名前のテスト。
    #[test]
    fn test_variable_name() {
        let mut val = Variable::new(Array::from_elem(IxDyn(&[100, 100, 100]), 1.0));

        assert_eq!(None, val.get_name());

        val.name = Some("test_val".to_string());
        assert_eq!(Some("test_val".to_string()), val.get_name());
    }

    /// バックプロパゲーションの有効・無効のテスト。
    #[test]
    fn test_disable_backprop() {
        // バックプロパゲーションを行わない場合
        Setting::set_backprop_disabled();
        let x = Rc::new(RefCell::new(Variable::new(Array::from_elem(
            IxDyn(&[100, 100, 100]),
            1.0,
        ))));

        let result = square(square(square(Rc::clone(&x))));

        // dbg!(&result.borrow().generation);
        assert_eq!(1, result.borrow().generation);

        // バックプロパゲーションを行う場合
        Setting::set_backprop_enabled();
        let x = Rc::new(RefCell::new(Variable::new(Array::from_elem(
            IxDyn(&[100, 100, 100]),
            1.0,
        ))));

        let result = square(square(square(Rc::clone(&x))));

        //dbg!(&result.borrow().generation);
        assert_eq!(3, result.borrow().generation);
    }

    /// 中間変数の微分結果を保持し無い場合のテスト
    #[test]
    fn test_retain_grad_disabled() {
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

        // dbg!(&d);
        // dbg!(&creators);

        // 実行した関数の数をチェックする。
        assert_eq!(5, creators.len());

        // 逆伝播を実行する。微分値を保持しない。
        Setting::set_retain_grad_disabled();
        // for (_gen, creator) in creators.iter() {
        //     creator.borrow_mut().backward();
        // }
        y.as_ref().clone().borrow().backward();

        // 逆伝播の結果の確認
        // 途中結果の変数には微分値が設定されていないことを確認する。
        assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), x1.borrow().get_data());
        assert_eq!(Array::from_elem(IxDyn(&[]), 64.0), x1.borrow().get_grad());
        assert_eq!(Array::from_elem(IxDyn(&[]), 3.0), x2.borrow().get_data());
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), x2.borrow().get_grad());
        assert_eq!(Array::from_elem(IxDyn(&[]), 4.0), a.borrow().get_data());
        assert!(a.borrow().grad.is_none());
        assert_eq!(Array::from_elem(IxDyn(&[]), 16.0), b.borrow().get_data());
        assert!(b.borrow().grad.is_none());
        assert_eq!(Array::from_elem(IxDyn(&[]), 16.0), c.borrow().get_data());
        assert!(c.borrow().grad.is_none());
        assert_eq!(Array::from_elem(IxDyn(&[]), 32.0), d.borrow().get_data());
        assert!(d.borrow().grad.is_none());
        assert_eq!(Array::from_elem(IxDyn(&[]), 35.0), y.borrow().get_data());
        assert!(y.borrow().grad.is_none());
    }

    /// 中間変数の微分結果を保持する場合のテスト。
    #[test]
    fn test_retain_grad_enabled() {
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

        // dbg!(&d);
        // dbg!(&creators);

        // 実行した関数の数をチェックする。
        assert_eq!(5, creators.len());

        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();
        // for (_gen, creator) in creators.iter() {
        //     creator.borrow_mut().backward();
        // }
        y.as_ref().clone().borrow().backward();

        // 逆伝播の結果の確認
        // 途中結果の変数には微分値が設定されていないことを確認する。
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

    // 世代に関するテスト。
    // x1 -> x1^2 -> a -> a^2 -> b -> b+c -> d -> d+x2 -> y
    //               -> a^2 -> c /          x2
    #[test]
    fn test_generations() {
        // 逆伝播を実行する。微分値を保持する。
        Setting::set_retain_grad_enabled();

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

        // dbg!(&d);
        // dbg!(&creators);

        // 実行した関数の数をチェックする。
        assert_eq!(5, creators.len());

        // 逆伝播を実行する。
        // for (_gen, creator) in creators.iter() {
        //     Setting::set_retain_grad_enabled();
        //     creator.borrow_mut().backward();
        // }
        y.as_ref().clone().borrow().backward();

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

        // dbg!(&creators);

        // 実行した関数の数をチェックする。
        assert_eq!(1, creators.len());

        // 逆伝播を実行する。
        // for (_gen, creator) in creators.iter() {
        //     creator.borrow_mut().backward();
        // }
        result.as_ref().clone().borrow().backward();

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
        // let output_data = output_result.borrow().get_data();
        // let output_grad = output_result.borrow().get_grad();
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), input1_data);
        assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), input2_data);

        // assert_eq!(expected_output_data.clone(), output_data.clone());
        // assert_eq!(Array::from_elem(IxDyn(&[]), 1.0), output_grad.clone());
        // 入力値の微分結果が 1 になってしまうが、2が正しい。
        assert_ne!(Array::from_elem(IxDyn(&[]), 1.0), input1_grad.clone());
        assert_ne!(Array::from_elem(IxDyn(&[]), 1.0), input2_grad.clone());
        assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), input1_grad.clone());
        assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), input2_grad.clone());
        //  dbg!(&output_result);
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
        // dbg!(x.clone());
        assert_eq!(
            expected_output_data.clone(),
            //results.clone().get(0).unwrap().borrow().clone().data
            // results.get(0).unwrap().borrow().data.clone()
            result.borrow().data.clone()
        );

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        // let creators = FunctionExecutor::extract_creators(vec![result.clone()]);
        // //dbg!(creators);
        // for (_gen, creator) in creators.iter() {
        //     creator.borrow_mut().backward();
        // }
        result.as_ref().clone().borrow().backward();

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        // dbg!(x.clone());

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
        // let creators = FunctionExecutor::extract_creators(vec![result.clone()]);
        // //dbg!(creators);
        // for (_gen, creator) in creators.iter() {
        //     creator.borrow_mut().backward();
        // }
        result.as_ref().clone().borrow().backward();

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        // dbg!(x.clone());

        let expected_grad = Array::from_elem(IxDyn(&[]), 2.0);
        assert_eq!(expected_grad, x.borrow().grad.clone().unwrap());
        assert_eq!(expected_output_data.clone(), result.borrow().clone().data);

        ////////////////////////////////
        // 微分をクリアせずにもう一度計算する。
        ////////////////////////////////
        let result2 = add(Rc::clone(&x), Rc::clone(&x));

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        // let creators2 = FunctionExecutor::extract_creators(vec![Rc::clone(&result2)]);
        // //dbg!(creators);
        // for (_gen, creator) in creators2.iter() {
        //     creator.borrow_mut().backward();
        // }
        result2.as_ref().clone().borrow().backward();

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        // dbg!(x.clone());

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
        // let creators3 = FunctionExecutor::extract_creators(vec![Rc::clone(&result3)]);
        //dbg!(creators);
        // for (_gen, creator) in creators3.iter() {
        //     creator.borrow_mut().backward();
        // }
        result3.as_ref().clone().borrow().backward();

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        // dbg!(x.clone());
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

        // dbg!(&creators);

        // 実行した関数の数をチェックする。
        assert_eq!(1, creators.len());

        // 逆伝播を実行する。
        // for (_gen, creator) in creators.iter() {
        //     creator.borrow_mut().backward();
        // }
        result.as_ref().clone().borrow().backward();

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

        // dbg!(&creators);

        // 実行した関数の数をチェックする。
        assert_eq!(1, creators.len());

        // 逆伝播を実行する。
        // for (_gen, creator) in creators.iter() {
        //     creator.borrow_mut().backward();
        // }
        result.as_ref().clone().borrow().backward();

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
        // for (_gen, creator) in creators.iter() {
        //     creator.borrow_mut().backward();
        // }
        result.as_ref().clone().borrow().backward();

        // exp 結果
        assert_eq!(expected_output_data, result.borrow().data);
        // 逆伝播の結果 exp^x の微分は exp^x
        assert_eq!(expected_output_data, x.borrow().get_grad());

        assert_eq!(Array::from_elem(IxDyn(&[]), 2.0), x.borrow().get_data());
        // assert_eq!(
        //     Array::from_elem(IxDyn(&[]), 1.0),
        //     result.borrow().get_grad()
        // );
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
        // dbg!(x1.clone());
        // dbg!(x2.clone());
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
        // let creators = FunctionExecutor::extract_creators(vec![result.clone()]);
        // //dbg!(creators);
        // for (_gen, creator) in creators.iter() {
        //     creator.borrow_mut().backward();
        // }
        result.as_ref().clone().borrow().backward();

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        // dbg!(x1.clone());
        // dbg!(x2.clone());

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
        //  dbg!(x1.clone());
        // dbg!(x2.clone());
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
        // let creators = FunctionExecutor::extract_creators(vec![result.clone()]);
        // //dbg!(creators);
        // for (_gen, creator) in creators.iter() {
        //     creator.borrow_mut().backward();
        // }
        result.as_ref().clone().borrow().backward();

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        // dbg!(x1.clone());
        // dbg!(x2.clone());
        // dbg!(result.clone().borrow().generation);

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
        // dbg!(expected.clone());

        // 関数を用意する。
        // let mut sq_exe_1 = FunctionExecutor::new(Rc::new(RefCell::new(Square)));
        // let mut exp_exe = FunctionExecutor::new(Rc::new(RefCell::new(Exp)));
        // let mut sq_exe_2 = FunctionExecutor::new(Rc::new(RefCell::new(Square)));

        // 順伝播の実行
        // let results = sq_exe_2.forward(exp_exe.forward(sq_exe_1.forward(vec![x.clone()])));
        let result = square(exp(square(x.clone())));

        // 順伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていないことも確認する。
        // dbg!(x.clone());
        assert_eq!(x_arr.clone(), x.borrow().data.clone());
        assert_eq!(None, x.borrow().grad.clone());
        assert_eq!(expected.clone(), result.borrow().data.clone());

        // 逆伝播のため、順伝播の関数の実行結果を取得し、逆伝播を実行する。
        // let creators = FunctionExecutor::extract_creators(vec![result.clone()]);
        // //dbg!(creators);
        // for (_gen, creator) in creators.iter() {
        //     creator.borrow_mut().backward();
        // }
        result.as_ref().clone().borrow().backward();

        // 逆伝播の結果を確認する。
        // 逆伝播の微分結果 grad が入力値に設定されていることも確認する。
        // dbg!(x.clone());

        // 逆伝播の正解は書籍の値を使用。
        let expected_x_grad = Array::from_elem(IxDyn(&[]), 3.297442541400256);

        assert_eq!(x_arr.clone(), x.borrow().data.clone());
        assert_eq!(expected_x_grad, x.borrow().grad.clone().unwrap());
        assert_eq!(expected.clone(), result.borrow().clone().data);
    }
}
