// ライブラリを一括でインポート
use crate::modules::*;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{
    s, stack, Array, ArrayView, ArrayViewMut, Axis, Dimension, Ix0, Ix1, IxDyn, Slice, SliceInfo,
    SliceInfoElem, Zip,
};
use plotters::chart::ChartBuilder;
use plotters::prelude::{BitMapBackend, Circle, IntoDrawingArea, PathElement};
use plotters::series::{LineSeries, PointSeries};
use plotters::style::{Color, IntoFont, BLACK, BLUE, RED, WHITE};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io::{Result, Write};
use std::ops::AddAssign;
use std::path::Path;
use std::rc::Rc;
use std::slice::Iter;

/// Variable を graphviz の DOT 言語で出力する。
///
/// Arguments:
/// * variable (Variable<V>): 出力対象の変数
/// * verbose (bool): 詳細を出力するかどうか(省略可能)
///
/// Returns:
/// * String: 変数の DOT 言語表記
#[macro_export]
macro_rules! dot_var {
    // verbose を指定しない場合は false で実行する。
    ($variable:ident) => {{
        dot_var!($variable, false)
    }};
    // verbose の指定がある場合の実行
    ($variable:ident, $verbose:literal) => {{
        let mut v_name = "".to_string();
        let temp_raw_v = $variable.raw().borrow().clone();
        if $verbose {
            if let Some(tmp_v_name) = $variable.get_name() {
                v_name = format!("{}", tmp_v_name);
            }
            let v_shape = &temp_raw_v.get_shape();
            let v_dtype = &temp_raw_v.get_dtype();
            v_name = format!("{}{:?} {}", v_name, &v_shape, &v_dtype);
        }
        let v_id = format!("{:p}", $variable.as_ref().as_ptr())
            .trim_start_matches("0x")
            .to_string();

        let result = format!(
            "{:?} [label=\"{}\", color=orange, style=filled]\n",
            v_id,
            v_name.to_string()
        );
        result
    }};
}

/// 関数の入出力関係を graphviz の DOT 言語で出力する。
///
/// Arguments:
/// * fe (Rc<RefCell<FunctionExecutor<V>>>): 出力対象の関数
///
/// Return:
/// * String: 関数の DOT 言語表記
pub fn dot_func<V: MathOps>(fe: Rc<RefCell<FunctionExecutor<V>>>) -> String {
    let inputs = fe.borrow().get_inputs();
    let outputs = fe.borrow().get_outputs();

    // この関数のポインタを ID として使用する。
    let f_id = format!("{:p}", fe.borrow().get_creator().as_ptr())
        .trim_start_matches("0x")
        .to_string();
    let f_name = &fe.borrow().get_creator().borrow().get_name();

    // 関数の情報を DOT 表記する。
    let mut txt = format!(
        "{:?} [label={:?}, color=lightblue, style=filled, shape=box]\n",
        &f_id, &f_name
    );

    // 入力値と関数の関係を DOT 表記する。
    // input のポインタを入力値の ID とする。
    for input in inputs {
        //let input_id = &input.as_ref().as_ptr();
        let input_id = format!("{:p}", input.as_ref().as_ptr())
            .trim_start_matches("0x")
            .to_string();

        let local_txt = format!("{:?} -> {:?}\n", &input_id, &f_id);
        txt = format!("{}{}", txt, local_txt);
    }

    // 出力値と関数の関係を DOT 表記する。
    // output のポインタを出力値の ID とする。
    for output in outputs {
        let output_id = format!("{:p}", output.upgrade().unwrap().as_ptr())
            .trim_start_matches("0x")
            .to_string();
        let local_txt = format!("{:?} -> {:?}\n", &f_id, &output_id);
        txt = format!("{}{}", txt, local_txt);
    }

    txt.to_string()
}

/// 計算グラフを graphviz の DOT 言語で出力する。
///
/// Arguments:
/// * variable (Variable<V>): 計算グラフの出力結果
/// * verbose (bool): 詳細を出力するかどうか(省略可能)
///
/// Returns:
/// * String: 計算グラフの DOT 言語表記
#[macro_export]
macro_rules! get_dot_graph {
    // verbose を指定しない場合は false で実行する。
    ($variable:ident) => {{
        get_dot_graph!($variable, false)
    }};
    // verbose の指定がある場合の実行
    ($variable:ident, $verbose:literal) => {{
        let mut txt = "".to_string();

        let local_dot_var_txt = $crate::dot_var!($variable, $verbose);
        txt = format!("{}{}", txt, local_dot_var_txt);

        let mut creators = FunctionExecutor::extract_creators(vec![$variable.clone()]);

        while let Some(creator) = creators.pop() {
            let local_dot_func_txt = dot_func(creator.1.clone());
            txt = format!("{}{}", txt, local_dot_func_txt);

            let inputs = creator.1.borrow().get_inputs();
            for input in inputs {
                let local_dot_var_txt = $crate::dot_var!(input, $verbose);
                txt = format!("{}{}", txt, local_dot_var_txt);
            }

            // let outputs = creator.1.get_outputs();
            // outputs.iter().for_each(|output| {
            //     let name = output.upgrade().unwrap().get_name();
            //     let ptr = output.upgrade().unwrap().as_ptr();
            // });

            // outputs.iter().for_each(|output| {
            //     let local_dot_var_txt = $crate::dot_var!(output.upgrade().unwrap(), $verbose);
            //     txt = format!("{}{}", txt, local_dot_var_txt);
            // });
        }
        format!("digraph g {{\n{}}}", txt)
    }};
}

/// 計算グラフを graphviz の DOT 言語と画像で出力する。
/// graphviz がインストール済みで、dot コマンドで実行可能であること。
/// graphviz のバージョン 2.43.0 で動作確認している。
///
/// 出力フォルダはカレントの .dots フォルダである。
/// .dots フォルダが存在しない場合は作成する。
///
/// Arguments:
/// * variable (Variable<V>): 計算グラフの出力結果
/// * to_file (String): 出力する png ファイル名
/// * verbose (bool): 詳細を出力するかどうか(省略可能)
///
/// Returns:
/// * String: 計算グラフの DOT 言語表記
#[macro_export]
macro_rules! plot_dot_graph {

    // verbose を指定しない場合は false で実行する。
    ($variable:ident, $to_file:ident) => {{
        plot_dot_graph!($variable, $to_file, false)
    }};
    // verbose の指定がある場合の実行
    ($variable:ident, $to_file:ident, $verbose:literal) => {{
        use std::fs::File;
        use std::io::Write;
        use std::process::{Stdio, Command};

        // ファイルを保存する .dots ディレクトリが存在しない場合は作成する。
        let output_dir = "./.dots";
        std::fs::create_dir_all(output_dir).unwrap();

        // graphviz の DOT ファイルを作成する。
        // ファイル名は to_file の末尾に .dot 拡張子を追記した名前とする。
        let dot_txt = $crate::get_dot_graph!($variable, $verbose);
        let tmp_dot_txt_file_path = format!("{}/{}.dot", output_dir, $to_file);
        let mut tmp_dot_txt_file = File::create(&tmp_dot_txt_file_path).unwrap();
        tmp_dot_txt_file.write_all(&dot_txt.as_bytes()).unwrap();

        // graphviz コマンドを実行し、png ファイルを作成する。
        let dot_png_file_path = format!("{}/{}", output_dir, $to_file);

        let args = [
            &tmp_dot_txt_file_path.to_string(),
            &"-T".to_string(),
            &"png".to_string(),
            &"-o".to_string(),
            &dot_png_file_path.to_string(),
        ];

        let output =Command::new("dot")
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        println!("{:?}", output);
        println!("{}", String::from_utf8_lossy(&output.stdout));
    }};
}

/// Variable の詳細を出力する。
pub fn debug_variable<V: MathOps>(x: Variable<V>, indent_num: usize) {
    let indent = format!("{}", "  ".repeat(indent_num));

    println!("{}variable", indent);
    println!("{}  name: {:?}", indent, x.get_name());
    println!("{}  data: {:?}", indent, x.get_data());
    println!("{}  generation: {:?}", indent, x.get_generation());

    match x.get_grad() {
        Some(grad) => {
            println!("{}  grad", indent);
            debug_variable(grad.clone(), indent_num + 2usize);
        }
        _ => println!("{}  grad is None", indent),
    }
    let creator = x.get_creator();
    match creator {
        Some(creator) => {
            println!(
                "{}  creator: {:?} gen: {:?}",
                indent,
                creator.borrow().get_creator().borrow().get_name(),
                creator.borrow().get_generation()
            );
            println!("{}  inputs", indent);
            let inputs = creator.borrow().get_inputs();
            for input in inputs {
                println!("{}    {:?}", indent, input.get_data());
                debug_variable(input.clone(), indent_num + 2usize);
            }
            println!("{}  outputs", indent);
            let outputs = creator.borrow().get_outputs();
            for output in outputs {
                let tmp_output = output.upgrade().unwrap();
                println!("{}    {:?}", indent, tmp_output.borrow().get_data());
                // debug_variable(
                //     Variable::new(tmp_output.clone()),
                //     format!("{}{}", indent, indent),
                // );
            }
        }
        _ => println!("{}  creator is None.", indent),
    }
}

/// Variable の詳細を出力する。
pub fn detail_variable<V: MathOps>(x: Variable<V>, indent_num: usize) -> Vec<String> {
    let mut result: Vec<String> = vec![];
    let indent = format!("{}", "  ".repeat(indent_num));

    result.push(format!("{}variable", indent));
    result.push(format!("{}  name: {:?}", indent, x.get_name()));
    result.push(format!("{}  data: {:?}", indent, x.get_data()));
    result.push(format!("{}  generation: {:?}", indent, x.get_generation()));

    match x.get_grad() {
        Some(_grad) => {
            result.push(format!("{}  grad", indent));
            // result.append(&mut detail_variable(grad.clone(), indent_num + 2usize));
        }
        _ => result.push(format!("{}  grad is None", indent)),
    }
    let creator = x.get_creator();
    match creator {
        Some(creator) => {
            result.push(format!(
                "{}  creator: {:?} gen: {:?}",
                indent,
                creator.borrow().get_creator().borrow().get_name(),
                creator.borrow().get_generation()
            ));
            result.push(format!("{}  inputs", indent));
            let inputs = creator.borrow().get_inputs();
            for input in inputs {
                result.push(format!(
                    "{}    name: {}, data: {:?}",
                    indent,
                    input.get_name().unwrap_or("None".to_string()),
                    input.get_data()
                ));
                //result.append(&mut detail_variable(input.clone(), indent_num + 2usize));
            }
            result.push(format!("{}  outputs", indent));
            let outputs = creator.borrow().get_outputs();
            for output in outputs {
                let tmp_output = output.upgrade().unwrap();
                result.push(format!(
                    "{}    name: {}, data: {:?}",
                    indent,
                    tmp_output.borrow().get_name().unwrap_or("None".to_string()),
                    tmp_output.borrow().get_data()
                ));
            }
        }
        _ => result.push(format!("{}  creator is None.", indent)),
    }
    result
}

/// Variable の詳細を出力する。
pub fn dump_detail_variable<V: MathOps>(
    file_path: String,
    x: Variable<V>,
    indent_num: usize,
) -> Result<()> {
    let indent = format!("{}", "  ".repeat(indent_num));

    let path = Path::new(&file_path);
    let mut file = File::create(&path)?;

    writeln!(file, "{}variable", indent)?;
    writeln!(file, "{}  name: {:?}", indent, x.get_name())?;
    writeln!(file, "{}  data: {:?}", indent, x.get_data())?;
    writeln!(file, "{}  generation: {:?}", indent, x.get_generation())?;

    match x.get_grad() {
        Some(grad) => {
            writeln!(file, "{}  grad", indent)?;
            let tmp = detail_variable(grad.clone(), indent_num + 2usize);
            for line in tmp {
                writeln!(file, "{}", line)?;
            }
        }
        _ => writeln!(file, "{}  grad is None", indent)?,
    }
    let creator = x.get_creator();
    match creator {
        Some(creator) => {
            writeln!(
                file,
                "{}  creator: {:?} gen: {:?}",
                indent,
                creator.borrow().get_creator().borrow().get_name(),
                creator.borrow().get_generation()
            )?;
            writeln!(file, "{}  inputs", indent)?;
            let inputs = creator.borrow().get_inputs();
            for input in inputs {
                writeln!(
                    file,
                    "{}    name: {}, data: {:?}",
                    indent,
                    input.get_name().unwrap_or("None".to_string()),
                    input.get_data()
                )?;
                let tmp = detail_variable(input.clone(), indent_num + 2usize);
                for line in tmp {
                    writeln!(file, "{}", line)?;
                }
            }
            writeln!(file, "{}  outputs", indent)?;
            let outputs = creator.borrow().get_outputs();
            for output in outputs {
                let tmp_output = output.upgrade().unwrap();
                writeln!(
                    file,
                    "{}    name: {}, data: {:?}",
                    indent,
                    tmp_output.borrow().get_name().unwrap_or("None".to_string()),
                    tmp_output.borrow().get_data()
                )
                .unwrap();
            }
            Ok(())
        }
        _ => writeln!(file, "{}  creator is None.", indent),
    }
}

/// Sum 関数の逆伝播の勾配についてリシェイプする。
///
/// Arguments:
/// * gy (Array<V, IxDyn>): 勾配
/// * x_shape (Vec<usize>): 順伝播時の形状
/// * axis (Option<Vec<isize>): Sum 関数で使用する軸
/// * keepdims (bool): 形状を維持するかどうか
///
/// Return:
/// * Array<V, IxDyn>: 形状変換後
pub fn reshape_sum_backward<V: MathOps>(
    gy: Variable<V>,
    x_shape: Vec<usize>,
    axis: Option<Vec<isize>>,
    keepdims: bool,
) -> Variable<V> {
    let ndim = x_shape.len();
    if ndim == 0 || axis.is_none() || keepdims {
        // No reshaping needed
        return gy;
    }

    let mut actual_axis = match axis {
        Some(axes) => axes
            .iter()
            .map(|&a| {
                if a >= 0 {
                    a as usize
                } else {
                    (a + ndim as isize) as usize
                }
            })
            .collect::<Vec<_>>(),
        None => Vec::new(),
    };
    // Start with the current shape of gy
    let mut shape = gy.get_data().shape().to_vec();

    // Insert 1s at the appropriate positions
    actual_axis.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for a in actual_axis {
        shape.insert(a, 1);
    }

    // Reshape gy to the new shape
    let reshape_gy = gy.get_data().into_shape_with_order(IxDyn(&shape)).unwrap();
    let gy_clone = gy.clone();
    gy_clone.set_data(reshape_gy);
    gy_clone
}

/// サイズが１の次元を削除する。
///
/// Arguments:
/// * arr (&Array<T, IxDyn>): 対象のテンソル
/// Return:
/// * Array<T, IxDyn>: 結果
pub fn squeeze<T: Clone>(arr: &Array<T, IxDyn>) -> Array<T, IxDyn> {
    // 長さ1でない次元を集める
    let new_shape: Vec<usize> = arr
        .shape()
        .iter()
        .filter(|&&dim| dim != 1)
        .cloned()
        .collect();

    // 新しい形状に変換
    // すべての次元が1の場合は少なくとも1次元は残す
    let final_shape = if new_shape.is_empty() {
        vec![1]
    } else {
        new_shape
    };

    arr.clone()
        .into_shape_with_order(IxDyn(&final_shape))
        .unwrap()
}

/// 数値微分
/// 関数の数値微分を行う。扱う数値は f64 固定とする。
///
/// Argumesnts:
/// * function (&mut FunctionExecutor<f64>): 評価する関数
/// * inputs (Vec<Variable<f64>>): 関数の入力値
///
/// Return:
/// * Vec<Variable<f64>>: 数値微分の結果
pub fn numerical_grad(
    function: &mut FunctionExecutor<f64>,
    inputs: Vec<Variable<f64>>,
) -> Vec<Variable<f64>> {
    let eps = 1e-4;

    // 結果を格納するベクトルを事前に確保
    let mut result: Vec<Variable<f64>> = Vec::with_capacity(inputs.len());

    // 入力の各変数について処理（インデックスも取得するためにenumerateを使用）
    for (i, input) in inputs.iter().enumerate() {
        let shape = input.get_data().shape().to_vec();
        let values = input.get_data().flatten().to_vec();
        let value_len = values.len();

        // この入力の勾配値を格納するベクトルを事前に確保
        let mut grad_values: Vec<f64> = Vec::with_capacity(value_len);

        // 各要素について数値微分を計算
        for j in 0..value_len {
            // 現在の入力値のコピーを一度だけ作成
            let mut inputs_plus = inputs.clone();
            let mut inputs_minus = inputs.clone();

            // j番目の要素にepsを加えた入力を作成
            {
                let mut values_plus = values.clone();
                values_plus[j] += eps;
                inputs_plus[i] = Variable::new(RawData::from_shape_vec(shape.clone(), values_plus));
            }

            // j番目の要素からepsを引いた入力を作成
            {
                let mut values_minus = values.clone();
                values_minus[j] -= eps;
                inputs_minus[i] =
                    Variable::new(RawData::from_shape_vec(shape.clone(), values_minus));
            }

            // 順伝播を一度だけ実行
            let y_plus = function.forward(inputs_plus);
            let y_minus = function.forward(inputs_minus);

            // 中心差分を計算
            let diff = sum(&y_plus[0] - &y_minus[0], None, false);
            let grad_value = diff.get_data().flatten().to_vec()[0] / (2.0 * eps);

            grad_values.push(grad_value);
        }

        // 形状に合わせた勾配変数を作成して結果に追加
        let grad_var = Variable::new(RawData::from_shape_vec(shape, grad_values));
        result.push(grad_var);
    }

    result
}

/// 数値微分を使ったバックプロパゲーションの結果チェック
/// バックプロパゲーションの結果と数値微分を比較し、近似するか確認する。
///
/// Argumesnts:
/// * function (&mut FunctionExecutor<f64>): 評価する関数
/// * inputs (Vec<Variable<f64>>): 関数の入力値
///
pub fn gradient_check(function: &mut FunctionExecutor<f64>, inputs: Vec<Variable<f64>>) -> bool {
    let mut func = function.to_owned();

    // テスト対象の関数について、数値微分を行う。
    let num_grads = numerical_grad(&mut func, inputs.clone());

    // テスト対象の関数について、順伝播と逆伝播を行う。
    // let y = func.forward(inputs.clone());
    let y = func.forward(inputs.clone());
    y[0].backward();

    // 入力値ごとに数値微分と逆伝播の結果が近似するか確認する。
    for i in 0..inputs.len() {
        let bp_grad = inputs[i].get_grad().unwrap().get_data();
        let num_grad = num_grads[i].get_data();

        // // 形状が一致すること
        // dbg!(&num_grad);
        // dbg!(&bp_grad);
        assert_eq!(num_grad.shape(), bp_grad.shape());

        // 近似していること
        // assert!(bp_grad.abs_diff_eq(&num_grad, 0.0000001));
        assert!(bp_grad.abs_diff_eq(&num_grad, 1e-4));
    }

    true
}

/// グラフ描画
///
/// Arguments
/// * caption (&str): 図のタイトル
/// * file_path (&str): グラフファイルのパス
/// * x_spec ((f64, f64)): 描画する X 軸の範囲
/// * y_spec ((f64, f64)): 描画する Y 軸の範囲
/// * plot_x (Vec<f64>): 学習データの X 軸の値
/// * plot_y (Vec<f64>): 学習データの Y 軸の値
/// * pred_xy (Vec<(f64, f64)>): 推論結果
/// * legend (&str): 凡例
pub fn draw_graph(
    caption: &str,
    file_path: &str,
    x_spec: (f64, f64),
    y_spec: (f64, f64),
    plot_x: Vec<f64>,
    plot_y: Vec<f64>,
    pred_xy: Vec<(f64, f64)>,
    legend: &str,
) {
    // グラフ描画
    // 描画先の Backend を初期化する。
    let root = BitMapBackend::new(&file_path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // グラフの軸の設定など
    let mut chart = ChartBuilder::on(&root)
        .caption(&caption, ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_spec.0..x_spec.1, y_spec.0..y_spec.1)
        .unwrap();
    chart.configure_mesh().draw().unwrap();

    let mut plot_data_vec = vec![];
    for i in 0..plot_x.len() {
        plot_data_vec.push(vec![plot_x[i], plot_y[i]]);
    }

    // 点グラフの定義＆描画
    let point_series = PointSeries::<_, _, Circle<_, _>, _>::new(
        plot_data_vec.iter().map(|xy| (xy[0], xy[1])),
        2,     // Circleのサイズ
        &BLUE, // 色を指定
    );
    chart.draw_series(point_series).unwrap();

    // 折れ線グラフ（関数グラフ）を描画
    chart
        .draw_series(LineSeries::new(pred_xy.iter().map(|(x, y)| (*x, *y)), &RED))
        .unwrap()
        .label(legend)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
}

/// 順序付き HashMap 構造体
#[derive(Debug, Clone)]
pub struct OrderedHashMap<T> {
    next_counter: usize,
    order: Vec<String>,
    hash_map: HashMap<String, T>,
}

impl<T: Clone> OrderedHashMap<T> {
    pub fn new() -> OrderedHashMap<T> {
        OrderedHashMap {
            next_counter: 0,
            order: Vec::new(),
            hash_map: HashMap::new(),
        }
    }

    /// キーと値を追加する。
    pub fn insert(&mut self, key: &str, value: T) {
        self.order.push(key.to_string());
        self.hash_map.insert(key.to_string(), value);
    }

    /// 値を取得する。
    pub fn get(&self, key: &str) -> &T {
        self.hash_map.get(&key.to_string()).unwrap()
    }

    /// 要素数を取得する。
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// イテレータ
    ///
    /// Retrun
    /// * HashMap の登録順にキーを返す。
    pub fn iter(&self) -> Iter<'_, String> {
        self.order.iter()
    }
}

impl<T: Clone> Iterator for OrderedHashMap<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_counter > self.order.len() {
            None
        } else {
            let key = self.order.get(self.next_counter).unwrap();
            let result = self.hash_map.get(key).unwrap().clone();
            self.next_counter += 1;
            Some(result)
        }
    }
}

// #[derive(Debug, Clone)]
// pub enum SliceElem {
//     Slice {
//         /// start index; negative are counted from the back of the axis
//         start: Option<isize>,
//         /// end index; negative are counted from the back of the axis; when not present
//         /// the default is the full length of the axis.
//         end: Option<isize>,
//         /// step size in elements; the default is 1, for every element.
//         step: isize,
//     },
//     /// A single index.
//     Index(Vec<usize>),
//     /// A new axis of length 1.
//     NewAxis,
// }

/// スライスの指定を表す構造体
#[derive(Debug, Clone)]
pub enum DynamicSlice {
    Range {
        start: Option<isize>,
        end: Option<isize>,
        step: isize,
    },
    Index(isize),
    Indices(Vec<isize>),
    MultidimIndices(Vec<Vec<isize>>),
    Full,
}

/// 動的スライサー構造体
#[derive(Debug, Clone)]
pub struct DynamicSlicer {
    slices: Vec<Option<DynamicSlice>>,
}

impl DynamicSlicer {
    pub fn new(dims: usize) -> Self {
        Self {
            slices: vec![None; dims],
        }
    }

    pub fn set_slice(&mut self, dim: usize, slice: DynamicSlice) -> &mut Self {
        if dim < self.slices.len() {
            self.slices[dim] = Some(slice);
        } else {
            self.slices.resize(dim + 1, None);
            self.slices[dim] = Some(slice);
        }
        self
    }

    // 戻り値の型を変更：ArrayView ではなく Array を返す
    pub fn slice<T>(&self, array: &Array<T, IxDyn>) -> Array<T, IxDyn>
    where
        T: Clone + std::fmt::Debug,
    {
        // 多次元インデックスが含まれている場合はファンシーインデックス用の処理
        for (dim, slice_opt) in self.slices.iter().enumerate() {
            if let Some(DynamicSlice::MultidimIndices(_)) = slice_opt {
                return self.fancy_index(array);
            }
        }

        // 通常のスライスの場合は結果をクローンして所有権を持つ配列を返す
        array
            .slice_each_axis(|ax| {
                let dim = ax.axis.index();
                let dim_len = array.shape()[dim];

                if dim < self.slices.len() {
                    match &self.slices[dim] {
                        Some(DynamicSlice::Range { start, end, step }) => {
                            let start_val = start.unwrap_or(0);
                            let end_val = end.map(|e| e).unwrap_or_else(|| dim_len as isize);
                            Slice {
                                start: start_val,
                                end: Some(end_val),
                                step: *step,
                            }
                        }
                        Some(DynamicSlice::Index(idx)) => Slice {
                            start: *idx,
                            end: Some(*idx + 1),
                            step: 1,
                        },
                        Some(DynamicSlice::Indices(indices)) => Slice {
                            start: 0,
                            end: None,
                            step: 1,
                        },
                        Some(DynamicSlice::MultidimIndices(_)) => Slice {
                            start: 0,
                            end: None,
                            step: 1,
                        },
                        Some(DynamicSlice::Full) | None => Slice {
                            start: 0,
                            end: None,
                            step: 1,
                        },
                    }
                } else {
                    Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    }
                }
            })
            .squeeze()
            .to_owned()
    }

    // ファンシーインデックス用のメソッド - 所有権を持つ配列を返す
    fn fancy_index<T>(&self, array: &Array<T, IxDyn>) -> Array<T, IxDyn>
    where
        T: Clone + std::fmt::Debug,
    {
        // 多次元インデックス配列を探す
        for (dim, slice_opt) in self.slices.iter().enumerate() {
            println!("処理する多次元インデックス(fancy): {:?}", slice_opt);
            if let Some(DynamicSlice::MultidimIndices(indices)) = slice_opt {
                // 結果を格納する配列を準備
                let mut result_views = Vec::new();

                // 各行に対して処理
                for row_indices in indices.iter() {
                    println!("fancy 処理する行 {:?}", row_indices);
                    let mut row_views = Vec::new();

                    // 各列に対して処理
                    for &idx in row_indices.iter() {
                        println!("fancy 処理する列 {:?}", &idx);
                        // その次元のインデックスに対応する部分配列を取得
                        let mut slicer = DynamicSlicer::new(array.ndim());
                        slicer.set_slice(dim, DynamicSlice::Index(idx));

                        // 他の次元のスライスをコピー
                        for (other_dim, other_slice) in self.slices.iter().enumerate() {
                            if other_dim != dim && other_slice.is_some() {
                                slicer.set_slice(other_dim, other_slice.clone().unwrap());
                            }
                        }

                        // 基本的なスライシングは to_owned で所有権を持つ配列にして追加
                        let view = slicer.slice(array);
                        row_views.push(view);
                    }

                    // 行方向に結合
                    if !row_views.is_empty() {
                        // ここで row_views を結合したものを result_views に追加
                        let stacked = stack(
                            Axis(0),
                            &row_views.iter().map(|a| a.view()).collect::<Vec<_>>(),
                        )
                        .expect("Failed to stack arrays");
                        result_views.push(stacked);
                    }
                }

                // 列方向に結合
                if !result_views.is_empty() {
                    let final_array = stack(
                        Axis(0),
                        &result_views.iter().map(|a| a.view()).collect::<Vec<_>>(),
                    )
                    .expect("Failed to stack arrays");
                    return final_array.into_dyn().squeeze();
                }
            }
        }

        // デフォルトとして元の配列のコピーを返す
        array.to_owned().squeeze()
    }

    // add.at機能を実装 - 指定された位置に値を加算する
    pub fn add_at<T>(&self, array: &mut Array<T, IxDyn>, values: &Array<T, IxDyn>)
    where
        T: Clone + AddAssign + Debug,
    {
        // まず特殊なスライス（Indices）を持つ次元を探す
        for (i, slice_opt) in self.slices.iter().enumerate() {
            if let Some(DynamicSlice::Indices(idxs)) = slice_opt {
                self.add_at_indices(array, idxs, values, i);
                return;
            } else if let Some(DynamicSlice::MultidimIndices(multi_idxs)) = slice_opt {
                self.add_at_multidim_indices(array, multi_idxs, values, i);
                return;
            }
        }

        // 単一インデックスケース (例: x[0, 1] += 5)
        let mut is_single_index = true;
        let mut indices = Vec::new();

        // スライスの種類に応じて処理を分岐する。
        // 複数インデックス指定や多次元インデックス指定の場合は他の処理を呼び出す。
        for (i, slice_opt) in self.slices.iter().enumerate() {
            match slice_opt {
                Some(DynamicSlice::Index(idx)) => {
                    indices.push(*idx as usize);
                }
                _ => {
                    is_single_index = false;
                    break;
                }
            }
        }

        // 単一要素への加算
        if is_single_index && indices.len() == array.ndim() {
            let index = ndarray::IxDyn(&indices);
            array[&index] += values.flatten().to_vec()[0].clone();
            return;
        }

        // その他のケース: スライスを取得して値を加算
        let mut slice_info = self.create_slice_info(array.shape());

        // スコープを使って mutable borrowing を制限
        {
            let mut target = array.slice_mut(slice_info.as_ref());

            if target.shape() == values.shape() {
                // シェイプが一致する場合、要素ごとに加算
                target += values;
            } else {
                // シェイプが異なる場合はブロードキャストを試みる
                let shaped_values = values.broadcast(target.shape()).unwrap();
                target += &shaped_values;
            }
        }
    }

    /// 単一次元の複数インデックスに対する add_at
    ///
    /// Arguments
    /// * array (&mut Array<T, IxDyn>): 対象の多次元配列
    /// * indices (&[isize]): 選択する複数のインデックス値
    /// * values (&Array<T, IxDyn>): 加算する値の配列
    /// * dim (usize): インデックスを適用する次元
    fn add_at_indices<T>(
        &self,
        array: &mut Array<T, IxDyn>, // 加算対象の多次元配列
        indices: &[isize],           // 選択する複数のインデックス値
        values: &Array<T, IxDyn>,    // 加算する値の配列
        dim: usize,                  // インデックスを適用する次元
    ) where
        T: Clone + AddAssign + Debug,
    {
        // 各インデックスに対して処理
        for (i, &idx) in indices.iter().enumerate() {
            // インデックス用の Slicer を作成
            let mut index_slicer = DynamicSlicer::new(array.ndim());
            index_slicer.set_slice(dim, DynamicSlice::Index(idx));

            // 対象の次元以外の次元について、スライスをコピー
            for (other_dim, other_slice) in self.slices.iter().enumerate() {
                if other_dim != dim && other_slice.is_some() {
                    index_slicer.set_slice(other_dim, other_slice.clone().unwrap());
                }
            }

            // 値を取得 (values 配列の対応する部分)
            // 現在処理中のインデックス（i）が値配列の第一次元の範囲内にあるか
            let value_slice = if values.ndim() > 0 && i < values.shape()[0] {
                values.index_axis(Axis(0), i).to_owned()
            } else {
                values.clone() // 単一値として使用
            };

            // スライスを取得して値を加算
            let slice_info = index_slicer.create_slice_info(array.shape());

            {
                let mut target = array.slice_mut(slice_info.as_ref());
                dbg!(&target);
                if target.shape() == value_slice.shape() {
                    target += &value_slice;
                } else {
                    // let shaped_value_slice =
                    //     value_slice.broadcast(target.shape()).unwrap_or(panic!(
                    //         "Shape mismatch at index {}: {:?} vs {:?}",
                    //         idx,
                    //         target.shape(),
                    //         value_slice.shape()
                    //     ));
                    let shaped_value_slice = value_slice.broadcast(target.shape()).unwrap();
                    dbg!(&shaped_value_slice);

                    target += &shaped_value_slice;
                }
            }
        }
    }

    // 多次元インデックス配列に対するadd_at
    fn add_at_multidim_indices<T>(
        &self,
        array: &mut Array<T, IxDyn>,
        indices: &[Vec<isize>],
        values: &Array<T, IxDyn>,
        dim: usize,
    ) where
        T: Clone + AddAssign + Debug,
    {
        println!("add_at_multi 処理する多次元インデックス: {:?}", indices);

        for (i, idx_vec) in indices.iter().enumerate() {
            println!("add_at_multi i: {}, idx_vec: {:?}", i, idx_vec);
            // 配列のインデックスを作成
            let mut index = Vec::with_capacity(array.ndim());
            for j in 0..array.ndim() {
                if j < idx_vec.len() {
                    // 多次元インデックスの要素を使用
                    index.push(idx_vec[j] as usize);
                } else {
                    // 不足している次元は0で埋める
                    index.push(0);
                }
            }

            println!("インデックス位置 {}: {:?}", i, index);

            // 値を取得
            let value = if values.ndim() > 0 && i < values.shape()[0] {
                println!(
                    "values ndim: {}, shape: {:?}, len: {}",
                    values.ndim(),
                    values.shape(),
                    values.len()
                );
                //dbg!(&values);
                // let idx = IxDyn(&index);
                let idx = IxDyn(&index);
                println!(
                    "index: {:?}, idx: {:?} /  array[idx]: {:?}",
                    &index, idx, array[&idx]
                );

                dbg!(&values[&idx]);
                // values[i].clone()
                values[&idx].clone()
            } else {
                values.iter().next().unwrap().clone() // 最初の値を使用
            };

            println!("加算する値: {:?}", value);

            // 指定位置に値を加算
            let idx = IxDyn(&index);
            println!("index: {:?}, idx: {:?}", &index, idx);
            dbg!(&array);
            dbg!(&array[&idx]);
            array[&idx] += value;
        }
    }

    /// SliceInfoElem を使用したスライス情報を作成するメソッド
    fn create_slice_info(
        &self,
        shape: &[usize],
    ) -> ndarray::SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn> {
        let ndim = shape.len();
        let mut slice_info_elems = Vec::with_capacity(ndim);

        // 各次元に対してスライス情報を構築する。
        for dim in 0..ndim {
            let dim_len = shape[dim];

            // その次元に対応するスライスが存在する場合、スライスの種類に応じて処理する。
            if dim < self.slices.len() {
                match &self.slices[dim] {
                    // レンジ指定のスライスの場合
                    Some(DynamicSlice::Range { start, end, step }) => {
                        let start_val = start.unwrap_or(0);
                        let end_val = end.map(|e| e).unwrap_or_else(|| dim_len as isize);
                        slice_info_elems.push(SliceInfoElem::Slice {
                            start: start_val,
                            end: Some(end_val),
                            step: *step,
                        });
                    }
                    // 単一インデックスのスライスの場合
                    Some(DynamicSlice::Index(idx)) => {
                        slice_info_elems.push(SliceInfoElem::Index(*idx));
                    }
                    // 複数インデックスのスライスの場合
                    Some(DynamicSlice::Indices(_)) | Some(DynamicSlice::MultidimIndices(_)) => {
                        // これらは通常のスライス操作ではサポートされていないので、フル範囲として扱う
                        slice_info_elems.push(SliceInfoElem::Slice {
                            start: 0,
                            end: None,
                            step: 1,
                        });
                    }
                    // 全範囲もしくはスライス指定なしの場合、全要素を選択する。
                    Some(DynamicSlice::Full) | None => {
                        slice_info_elems.push(SliceInfoElem::Slice {
                            start: 0,
                            end: None,
                            step: 1,
                        });
                    }
                }
            } else {
                // スライス指定のない次元の場合、全範囲とする。
                slice_info_elems.push(SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                });
            }
        }

        SliceInfo::<_, _, _>::try_from(slice_info_elems).unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use ndarray::Array;
    use ndarray_rand::RandomExt;
    // use rand::prelude::*;
    use rand::{distributions::Uniform, SeedableRng};
    use rand_isaac::Isaac64Rng;

    // #[test]
    // fn test_add_at_multidim_04() {
    //     // 4次元配列（2x3x4x2）を作成
    //     let mut array = Array::zeros((2, 3, 4, 2)).into_dyn();

    //     // 第2次元と第3次元の特定の組み合わせを指定
    //     let multi_indices = vec![
    //         vec![1, 2, 0, 0], // 第0次元=1, 第1次元=2, 第2次元=0, 第3次元=0
    //         vec![1, 2, 1, 1], // 第0次元=1, 第1次元=2, 第2次元=1, 第3次元=1
    //         vec![1, 2, 2, 0], // 第0次元=1, 第1次元=2, 第2次元=2, 第3次元=0
    //         vec![1, 2, 3, 1], // 第0次元=1, 第1次元=2, 第2次元=3, 第3次元=1
    //     ];

    //     // 加算する値
    //     let values = Array::from_vec(vec![5.0, 10.0, 15.0, 20.0]).into_dyn();

    //     // スライサーを設定
    //     let mut slicer = DynamicSlicer::new(4);
    //     slicer.set_slice(0, DynamicSlice::Index(1)); // 第0次元は固定
    //     slicer.set_slice(1, DynamicSlice::Index(2)); // 第1次元も固定
    //     slicer.set_slice(2, DynamicSlice::MultidimIndices(multi_indices)); // 第2・第3次元の組み合わせ

    //     // add_atを実行
    //     slicer.add_at(&mut array, &values);

    //     println!("4次元配列の特定位置に加算後:");

    //     println!("array: {:?}", array.slice(s![.., 0..2, .., ..]));
    //     assert_eq!(
    //         std::iter::repeat(0.0).take(32).collect::<Vec<_>>(),
    //         array.slice(s![.., 0..2, .., ..]).flatten().to_vec()
    //     );

    //     println!("array: {:?}", array.slice(s![1, 2, .., ..]));
    //     assert_eq!(
    //         vec![5.0, 0.0, 0.0, 10.0, 15.0, 0.0, 0.0, 20.0],
    //         array.slice(s![1, 2, .., ..]).flatten().to_vec()
    //     );
    // }

    // #[test]
    // fn test_add_at_multidim_03() {
    //     // 8x8のチェスボード状の行列を作成
    //     let mut array = Array::zeros((8, 8)).into_dyn();

    //     // チェス盤のナイトの動きのような位置を指定
    //     let multi_indices = vec![
    //         vec![0, 0], // 起点
    //         vec![1, 2], // ナイトの動き1
    //         vec![2, 4], // ナイトの動き2
    //         vec![3, 6], // ナイトの動き3
    //         vec![4, 7], // 別の位置
    //         vec![5, 5], // 別の位置
    //         vec![6, 3], // 別の位置
    //         vec![7, 1], // 別の位置
    //     ];

    //     // すべての位置に同じ値を加算
    //     let values = Array::from_elem(8, 1.0).into_dyn();

    //     // スライサーを設定
    //     let mut slicer = DynamicSlicer::new(2);
    //     slicer.set_slice(0, DynamicSlice::MultidimIndices(multi_indices));

    //     // add_atを実行
    //     slicer.add_at(&mut array, &values);

    //     println!("パターン化された位置に加算後の配列:");
    //     println!("{:?}", array);

    //     assert_eq!(array.slice(s![0, 0]).flatten().to_vec(), vec!(1.0));
    //     assert_eq!(array.slice(s![1, 2]).flatten().to_vec(), vec!(1.0));
    //     assert_eq!(array.slice(s![2, 4]).flatten().to_vec(), vec!(1.0));
    //     assert_eq!(array.slice(s![3, 6]).flatten().to_vec(), vec!(1.0));
    //     assert_eq!(array.slice(s![4, 7]).flatten().to_vec(), vec!(1.0));
    //     assert_eq!(array.slice(s![5, 5]).flatten().to_vec(), vec!(1.0));
    //     assert_eq!(array.slice(s![6, 3]).flatten().to_vec(), vec!(1.0));
    //     assert_eq!(array.slice(s![7, 1]).flatten().to_vec(), vec!(1.0));
    // }
    // #[test]
    // fn test_add_at_multidim_02() {
    //     // 3次元配列（3x3x3）を作成
    //     let mut array = Array::zeros((3, 3, 3)).into_dyn();

    //     // 3D座標を指定（四隅）
    //     let multi_indices = vec![
    //         vec![0, 0, 0], // 原点
    //         vec![0, 0, 2], // z軸上の端点
    //         vec![0, 2, 0], // y軸上の端点
    //         vec![2, 0, 0], // x軸上の端点
    //     ];

    //     // 加算する値
    //     let values = Array::from_vec(vec![100.0, 200.0, 300.0, 400.0]).into_dyn();

    //     // スライサーを設定
    //     let mut slicer = DynamicSlicer::new(3);
    //     slicer.set_slice(0, DynamicSlice::MultidimIndices(multi_indices));

    //     // add_atを実行
    //     slicer.add_at(&mut array, &values);

    //     println!("加算後の3D配列（断面表示）:");
    //     for i in 0..3 {
    //         println!("z = {}:", i);
    //         println!("{:?}", array.slice(s![.., .., i]));
    //     }
    //     assert_eq!(array.slice(s![0, 0, 0]).flatten().to_vec(), vec![100.0]);
    //     assert_eq!(array.slice(s![0, 0, 2]).flatten().to_vec(), vec![200.0]);
    //     assert_eq!(array.slice(s![0, 2, 0]).flatten().to_vec(), vec![300.0]);
    //     assert_eq!(array.slice(s![2, 0, 0]).flatten().to_vec(), vec![400.0]);
    // }

    // #[test]
    // fn test_add_at_multidim_01() {
    //     // 2次元配列（5x5）を作成
    //     let mut x = Array::zeros((5, 5)).into_dyn();
    //     println!("Original array:");
    //     println!("{:?}", x);

    //     // 多次元インデックスを設定（対角線の位置を指定）
    //     let multi_indices = vec![
    //         vec![0, 0], // 左上
    //         vec![1, 1], // 対角線上の2番目
    //         vec![2, 2], // 対角線上の3番目
    //         vec![3, 3], // 対角線上の4番目
    //     ];

    //     // 加算する値
    //     let values = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]).into_dyn();

    //     // スライサーを設定
    //     let mut slicer = DynamicSlicer::new(2);
    //     slicer.set_slice(0, DynamicSlice::MultidimIndices(multi_indices));

    //     // add_atを実行
    //     slicer.add_at(&mut x, &values);

    //     println!("\n加算後の配列:");
    //     println!("{:?}", x);
    // }

    #[test]
    fn test_add_at_2dim_multi_cols() {
        // 3次元配列の作成 (2×3x4)
        let mut x = Array::zeros((2, 3, 4)).into_dyn();
        println!("Original array:");
        println!("{:?}", x);

        // 加算する値 (インデックス0 に 10、インデックス2 に 20 を加算)
        let values = Array::from_vec(vec![10.0, 20.0]).into_dyn();
        let mut slicer = DynamicSlicer::new(3);

        // スライスの設定
        slicer.set_slice(2, DynamicSlice::Indices(vec![0, 2]));

        slicer.add_at(&mut x, &values.into_dyn());

        println!("\nAfter");
        println!("{:?}", x);

        assert_eq!(vec![2, 3, 4], x.shape().to_vec());
        assert_eq!(
            vec![
                10., 0., 20., 0., 10., 0., 20., 0., 10., 0., 20., 0., 10., 0., 20., 0., 10., 0.,
                20., 0., 10., 0., 20., 0.,
            ],
            x.flatten().to_vec()
        );
    }

    #[test]
    fn test_add_at_2dim_multi_rows() {
        // 2次元配列の作成（3行4列）
        let mut x = Array::zeros((3, 4)).into_dyn();
        x.fill(1.0); // すべての要素を1.0に設定
        println!("Original array:");
        println!("{:?}", x);

        // 加算する値
        let values = Array::from_shape_vec(
            (2, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, // 1行目への加算値
                5.0, 6.0, 7.0, 8.0, // 2行目への加算値
            ],
        )
        .unwrap()
        .into_dyn();

        // インデックス [0, 2] の行に値を加算（行 = 次元0）
        let mut slicer = DynamicSlicer::new(0);
        slicer.set_slice(0, DynamicSlice::Indices(vec![0, 2]));

        slicer.add_at(&mut x, &values.into_dyn());

        println!("\nAfter");
        println!("{:?}", x);

        assert_eq!(vec![3, 4], x.shape().to_vec());
        assert_eq!(
            vec![2., 3., 4., 5., 1., 1., 1., 1., 6., 7., 8., 9.,],
            x.flatten().to_vec()
        );
    }

    #[test]
    fn test_add_at_1dim_indices_01() {
        // 1次元配列の作成
        let mut x = Array::linspace(0., 9., 10).into_dyn(); // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        println!("Original array:");
        println!("{:?}", x);

        let values = Array::from_vec(vec![10.0, 20.0, 30.0]).into_dyn();

        // インデックス [1, 3, 7] の位置に値を加算
        let mut slicer = DynamicSlicer::new(0);
        slicer.set_slice(0, DynamicSlice::Indices(vec![1, 3, 7])); // 2行目

        slicer.add_at(&mut x, &values.into_dyn());

        println!("\nAfter");
        println!("{:?}", x);
        assert_eq!(
            vec![0.0, 11.0, 2.0, 23.0, 4.0, 5.0, 6.0, 37.0, 8.0, 9.0],
            x.flatten().to_vec()
        );
    }

    /// 2行目の2列目に10を足す。
    /// [1, 1] += 10 と同等の操作
    #[test]
    fn test_add_at_simple_index_01() {
        // テスト用の配列を作成 (3x3の2次元配列)
        let mut x = Array::from_shape_vec((3, 3), (0..9).collect())
            .unwrap()
            .into_dyn();
        println!("Original array:");
        println!("{:?}", x);

        // x[1, 1] += 10 と同等の操作
        let mut slicer = DynamicSlicer::new(2);
        slicer.set_slice(0, DynamicSlice::Index(1)); // 2行目
        slicer.set_slice(1, DynamicSlice::Index(1)); // 2列目

        let values = Array::from_elem((), 10);
        slicer.add_at(&mut x, &values.into_dyn());

        println!("\nAfter x[1, 1] += 10:");
        println!("{:?}", x);

        assert_eq!(vec![3, 3], x.shape().to_vec());
        assert_eq!(vec![0, 1, 2, 3, 14, 5, 6, 7, 8], x.flatten().to_vec());
    }

    /// 1行目と3行目に1を足す。
    /// x[[0, 2], :] += 1 と同等の操作
    #[test]
    fn test_add_at_slice_01() {
        // テスト用の配列を作成 (3x3の2次元配列)
        let mut x = Array::from_shape_vec((3, 3), (0..9).collect())
            .unwrap()
            .into_dyn();
        println!("Original array:");
        println!("{:?}", x);

        let mut slicer = DynamicSlicer::new(1);
        slicer.set_slice(0, DynamicSlice::MultidimIndices(vec![vec![0, 2]])); // 1行目と3行目

        dbg!(&slicer);

        let sliced_result = slicer.slice(&x);
        dbg!(&sliced_result);

        // x[[0, 2], :] += 1 と同等の操作
        let values = Array::from_elem(vec![2, 3], 1); // 2行3列の配列で、全て1
        slicer.add_at(&mut x, &values.into_dyn());

        println!("\nAfter x[[0, 2], :] += 1:");
        println!("{:?}", x);

        // assert_eq!(vec![3, 3], x.shape().to_vec());
        // assert_eq!(vec![1, 2, 3, 3, 4, 5, 7, 8, 9], x.flatten().to_vec());
    }

    /// 1行目と3行目に1を足す。
    /// x[[[0, 2], [1, 1]], :] += 5 と同等の操作
    // #[test]
    // fn test_add_at_multi_indices_01() {
    //     // テスト用の配列を作成 (3x3の2次元配列)
    //     let mut x = Array::from_shape_vec((3, 3), (0..9).collect())
    //         .unwrap()
    //         .into_dyn();
    //     println!("Original array:");
    //     println!("{:?}", x);

    //     let mut slicer = DynamicSlicer::new(2);
    //     slicer.set_slice(
    //         0,
    //         DynamicSlice::MultidimIndices(vec![vec![0, 2], vec![1, 1]]),
    //     );

    //     // x[[[0, 2], [1, 1]], :] += 5 と同等の操作
    //     // let values = Array::from_elem(vec![2, 2, 3], 5); // 2x2x3の配列で、全て5
    //     let values = Array::from_elem(vec![2], 10); // 2x2x3の配列で、全て5

    //     let result = slicer.slice(&x);
    //     dbg!(&result);

    //     slicer.add_at(&mut x, &values.into_dyn());

    //     println!("\nAfter x[[[0, 2], [1, 1]], :] += 5:");
    //     println!("{:?}", x);

    //     assert_eq!(vec![3, 3], x.shape().to_vec());
    //     assert_eq!(vec![0, 1, 12, 3, 14, 5, 6, 7, 8], x.flatten().to_vec());
    // }

    #[test]
    fn test_slice_multi_01() {
        // Python の例
        // x = np.arange(9).reshape((3, 3))
        // result = x[[[0,2],[1,1]]]
        // x: [[0 1 2] [3 4 5] [6 7 8]]
        // result: [[[0 1 2] [6 7 8]] [[3 4 5] [3 4 5]]]

        let array = Array::from_shape_vec(vec![3, 3], (0..=8).collect()).unwrap();
        println!("array: {:?}", array);
        let mut slicer = DynamicSlicer::new(array.ndim());

        // 第1次元 (dim=0) のみをスライス
        slicer.set_slice(0, DynamicSlice::Index(0)); // 第1次元の0のみを選択
                                                     // slicer.set_slice(2, DynamicSlice::Index(2)); // 第1次元の0のみを選択
                                                     // slicer.set_slice(1, DynamicSlice::Index(1)); // 第1次元の0のみを選択
                                                     // slicer.set_slice(1, DynamicSlice::Index(1)); // 第1次元の0のみを選択

        let dim0_slice = slicer.slice(&array);
        println!("\nDim 0, Index 0:");
        println!("Sliced shape: {:?}", dim0_slice.shape());
        println!("Sliced: {:?}", dim0_slice);

        // assert_eq!(vec![2, 3], dim0_slice.shape().to_vec());
        // assert_eq!(vec![0, 1, 2, 3, 4, 5], dim0_slice.flatten().to_vec());
    }

    #[test]
    fn test_slice_index_01() {
        // Python の例
        // x_data = np.arange(12).reshape((2, 2, 3))
        // x = Variable(x_data)
        // y = F.get_item(x, 0)
        // self.assertTrue(array_allclose(y.data, x_data[0]))
        // [[0 1 2] [3 4 5]]
        let array = Array::from_shape_vec(vec![2, 2, 3], (0..=11).collect()).unwrap();
        println!("array: {:?}", array);

        let mut slicer = DynamicSlicer::new(array.ndim());
        // 第1次元 (dim=0) のみをスライス
        slicer.set_slice(0, DynamicSlice::Index(0)); // 第1次元の0のみを選択

        let dim0_slice = slicer.slice(&array);
        println!("\nDim 0, Index 0:");
        println!("Sliced shape: {:?}", dim0_slice.shape());
        println!("Sliced: {:?}", dim0_slice);
        assert_eq!(vec![2, 3], dim0_slice.shape().to_vec());
        assert_eq!(vec![0, 1, 2, 3, 4, 5], dim0_slice.flatten().to_vec());
    }

    #[test]
    fn test_slice_index_and_range_01() {
        // Python の例
        // x_data = np.arange(12).reshape((2, 2, 3))
        // x = Variable(x_data)
        // y = F.get_item(x, (0, 0, slice(0, 2, 1)))
        // self.assertTrue(array_allclose(y.data, x_data[0, 0, 0:2:1]))
        // [[0 1]

        let array = Array::from_shape_vec(vec![2, 2, 3], (0..=11).collect()).unwrap();

        println!("array: {:?}", array);

        let slice = DynamicSlice::Range {
            start: Some(0),
            end: Some(2),
            step: 1,
        };

        let mut slicer = DynamicSlicer::new(array.ndim());
        slicer
            .set_slice(0, DynamicSlice::Index(0)) // 第1次元: インデックス0
            .set_slice(1, DynamicSlice::Index(0)) // 第2次元: インデックス0
            .set_slice(
                2,
                DynamicSlice::Range {
                    // 第3次元: 0から2未満 (0, 1)
                    start: Some(0),
                    end: Some(2),
                    step: 1,
                },
            );

        let dim0_slice = slicer.slice(&array);
        println!("\nDim 0, Index 0:");
        println!("Sliced shape: {:?}", dim0_slice.shape());
        println!("Sliced: {:?}", dim0_slice);
        assert_eq!(vec![2], dim0_slice.shape().to_vec());
        assert_eq!(vec![0, 1], dim0_slice.flatten().to_vec());
    }

    #[test]
    fn test_slice_range_01() {
        // Python の例
        // x_data = np.arange(12).reshape((2, 2, 3))
        //   x = Variable(x_data)
        //   y = F.get_item(x, (Ellipsis, 2))
        //   self.assertTrue(array_allclose(y.data, x_data[..., 2]))
        // [[ 2  5] [ 8 11]]
        let array = Array::from_shape_vec(vec![2, 2, 3], (0..=11).collect()).unwrap();

        println!("array: {:?}", array);

        let slice = DynamicSlice::Range {
            start: Some(0),
            end: Some(2),
            step: 1,
        };

        let mut slicer = DynamicSlicer::new(array.ndim());
        // 全ての次元の最後の要素を取得
        slicer.set_slice(
            2,
            DynamicSlice::Range {
                start: Some(2),
                end: Some(3),
                step: 1,
            },
        );

        let dim0_slice = slicer.slice(&array);
        println!("\nDim 0, Index 0:");
        println!("Sliced shape: {:?}", dim0_slice.shape());
        println!("Sliced: {:?}", dim0_slice);
        assert_eq!(vec![2, 2], dim0_slice.shape().to_vec());
        assert_eq!(vec![2, 5, 8, 11], dim0_slice.flatten().to_vec());
    }

    #[test]
    fn test_slice_fancy_01() {
        // Python の例
        // x_data = np.array([[1,2,3],[4,5,6]])
        // indices = np.array([0, 0, 1])
        //   x = Variable(x_data)
        //   y = x[self.slices]
        // [[1 2 3] [1 2 3] [4 5 6]
        let array = Array::from_shape_vec(vec![2, 3], (0..=5).collect()).unwrap();

        println!("array: {:?}", array);

        let indices = vec![vec![0, 0, 1]];

        let mut slicer = DynamicSlicer::new(2);
        slicer.set_slice(0, DynamicSlice::MultidimIndices(indices));

        let result = slicer.slice(&array);

        println!("Sliced shape: {:?}", result.shape());
        println!("Sliced: {:?}", result);
        assert_eq!(vec![3, 3], result.shape().to_vec());
        assert_eq!(vec![0, 1, 2, 0, 1, 2, 3, 4, 5], result.flatten().to_vec());
    }

    #[test]
    fn test_perform_1() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let x_var = Array::random_using((10, 20), Uniform::new(0., 10.), &mut rng);
        let x = Variable::new(RawData::from_shape_vec(
            vec![10, 20],
            x_var.flatten().to_vec(),
        ));

        let w_var = Array::random_using((20, 30), Uniform::new(0., 10.), &mut rng);
        let w = Variable::new(RawData::from_shape_vec(
            vec![20, 30],
            w_var.flatten().to_vec(),
        ));

        let mut linear: FunctionExecutor<_> =
            FunctionExecutor::new(Rc::new(RefCell::new(LinearFunction {})));
        utils::gradient_check(&mut linear, vec![x.clone(), w.clone()]);
    }

    #[test]
    fn test_num_grad() {
        let x0_values = vec![
            0.48905058, 0.12052497, 0.25442758, 0.92318036, 0.38172492, 0.9410574, 0.00147813,
            0.80549311, 0.4386291, 0.02845124,
        ];
        let x1_values = vec![
            0.82280806, 0.09954563, 0.68377267, 0.59132052, 0.69092459, 0.24181232, 0.66857664,
            0.86462292, 0.25217713, 0.14296797,
        ];

        let x0 = Variable::new(RawData::from_shape_vec(vec![1, 10], x0_values));
        let x1 = Variable::new(RawData::from_shape_vec(vec![1, 10], x1_values));

        let mut mean_squared_error =
            FunctionExecutor::new(Rc::new(RefCell::new(MeanSquaredErrorFunction {})));

        gradient_check(&mut mean_squared_error, vec![x0.clone(), x1.clone()]);
    }

    #[test]
    fn test_squeeze() {
        let a =
            Array::from_shape_vec(IxDyn(&[2, 1, 3, 1, 4, 1, 1, 5]), (0..120).collect()).unwrap();

        println!("元の形状: {:?}", a.shape());

        // squeeze適用後、形状は[2, 3]になる
        let b = squeeze(&a);
        println!("squeeze後の形状: {:?}", b.shape());
        assert_eq!(vec![2, 3, 4, 5], b.shape());
    }

    #[test]
    fn test_debug_variable() {
        let x1 = Variable::new(RawData::new(5.0f32));
        x1.set_name("x1".to_string());
        let x2 = Variable::new(RawData::new(10.0f32));
        x2.set_name("x2".to_string());
        let x3 = Variable::new(RawData::new(15.0f32));
        x3.set_name("x3".to_string());

        let result = &(&x1 * &x2) + &x3;
        debug_variable(result, 1);
    }

    #[test]
    fn test_plot_dot_graph_1() {
        let x = Variable::new(RawData::new(1));
        x.set_name("x".to_string());
        let y = Variable::new(RawData::new(1));
        y.set_name("y".to_string());
        let z = matyas(x.clone(), y.clone());

        let file_name = "test_plot_dot_graph_1.png";

        plot_dot_graph!(z, file_name, true);
    }

    /// get_dot_graph のテスト
    /// 掛け算と足し算
    #[test]
    fn test_get_dot_graph_1() {
        let x1 = Variable::new(RawData::new(5.0f32));
        x1.set_name("x1".to_string());
        let x2 = Variable::new(RawData::new(10.0f32));
        x2.set_name("x2".to_string());
        let x3 = Variable::new(RawData::new(15.0f32));
        x3.set_name("x3".to_string());

        let result = &(&x1 * &x2) + &x3;
        let dot_txt = get_dot_graph!(result, true);
        println!("{}", dot_txt)
    }

    /// get_dot_graph のテスト
    /// Sphere
    #[test]
    fn test_get_dot_graph_sphere() {
        let x = Variable::new(RawData::new(1));
        x.set_name("x".to_string());
        let y = Variable::new(RawData::new(1));
        y.set_name("y".to_string());
        let z = sphere(x.clone(), y.clone());

        let dot_txt = get_dot_graph!(z, true);
        println!("{}", dot_txt)
    }

    /// get_dot_graph のテスト
    /// matyas
    #[test]
    fn test_get_dot_graph_matyas() {
        let x = Variable::new(RawData::new(1));
        x.set_name("x".to_string());
        let y = Variable::new(RawData::new(1));
        y.set_name("y".to_string());
        let z = matyas(x.clone(), y.clone());

        let dot_txt = get_dot_graph!(z, true);
        println!("{}", dot_txt)
    }

    /// get_dot_graph のテスト
    /// goldstein
    #[test]
    fn test_get_dot_graph_goldstein() {
        let x = Variable::new(RawData::new(1));
        x.set_name("x".to_string());
        let y = Variable::new(RawData::new(1));
        y.set_name("y".to_string());
        let z = goldstein(x.clone(), y.clone());

        let dot_txt = get_dot_graph!(z, true);
        println!("{}", dot_txt)
    }

    /// dot_func のテスト
    /// 掛け算１つのみ。
    #[test]
    fn test_dot_func_1() {
        let x1 = Variable::new(RawData::new(5.0f32));
        let x2 = Variable::new(RawData::new(10.0f32));

        let result = &x1 * &x2;
        let txt = dot_func(result.get_creator().unwrap());
        println!("{}", txt);
    }

    /// dot_func のテスト
    /// 掛け算と足し算
    #[test]
    fn test_dot_func_2() {
        let x1 = Variable::new(RawData::new(5.0f32));
        let x2 = Variable::new(RawData::new(10.0f32));
        let x3 = Variable::new(RawData::new(15.0f32));

        let result = &(&x1 * &x2) + &x3;
        let mut creators = FunctionExecutor::extract_creators(vec![result.clone()]);
        while let Some(creator) = creators.pop() {
            let txt = dot_func(creator.1.clone());
            println!("{}", txt);
        }
    }

    /// dot_var のテスト
    /// スカラ値、詳細情報を出力
    #[test]
    fn test_dot_var_1() {
        let var = Variable::new(RawData::new(2.0));
        var.set_name("x".to_string());
        let result = dot_var!(var, true);
        println!("{}", result);
    }

    /// dot_var のテスト
    /// 行列、詳細情報を出力
    #[test]
    fn test_dot_var_2() {
        let var = Variable::new(RawData::from_shape_vec(vec![2, 2], vec![10, 20, 30, 40]));
        var.set_name("2x2dim".to_string());
        let result = dot_var!(var, true);
        println!("{}", result);
    }

    /// dot_var のテスト
    /// 行列、詳細情報なし
    #[test]
    fn test_dot_var_3() {
        let var = Variable::new(RawData::from_shape_vec(vec![2, 2], vec![10, 20, 30, 40]));
        var.set_name("2x2dim".to_string());
        let result = dot_var!(var);
        println!("{}", result);
    }
}
