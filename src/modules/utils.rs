// ライブラリを一括でインポート
use crate::modules::*;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use plotters::chart::ChartBuilder;
use plotters::prelude::{BitMapBackend, Circle, IntoDrawingArea, PathElement};
use plotters::series::{LineSeries, PointSeries};
use plotters::style::{Color, IntoFont, BLACK, BLUE, RED, WHITE};
use std::cell::RefCell;
use std::fs::File;
use std::io::{Result, Write};
use std::path::Path;
use std::rc::Rc;

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
            if let Some(tmp_v_name) = $variable.raw().borrow().get_name() {
                v_name = format!("{}({}): ", tmp_v_name, $variable.raw().borrow().get_data());
            } else {
                v_name = format!("({})", $variable.raw().borrow().get_data());
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

            // let outputs = creator.1.borrow().get_outputs();
            // outputs.iter().for_each(|output| {
            //     let name = output.upgrade().unwrap().borrow().get_name();
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
    println!("{}  name: {:?}", indent, x.borrow().get_name());
    println!("{}  data: {:?}", indent, x.borrow().get_data());
    println!("{}  generation: {:?}", indent, x.borrow().get_generation());

    match x.borrow().get_grad() {
        Some(grad) => {
            println!("{}  grad", indent);
            debug_variable(grad.clone(), indent_num + 2usize);
        }
        _ => println!("{}  grad is None", indent),
    }
    let creator = x.borrow().get_creator();
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
                println!("{}    {:?}", indent, input.borrow().get_data());
                debug_variable(input.clone(), indent_num + 2usize);
            }
            println!("{}  outputs", indent);
            let outputs = creator.borrow().get_outputs();
            for output in outputs {
                let tmp_output = output.upgrade().unwrap();
                println!("{}    {:?}", indent, tmp_output.borrow().get_data());
                // debug_variable(
                //     Variable::new(tmp_output.borrow().clone()),
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
    result.push(format!("{}  name: {:?}", indent, x.borrow().get_name()));
    result.push(format!("{}  data: {:?}", indent, x.borrow().get_data()));
    result.push(format!(
        "{}  generation: {:?}",
        indent,
        x.borrow().get_generation()
    ));

    match x.borrow().get_grad() {
        Some(_grad) => {
            result.push(format!("{}  grad", indent));
            // result.append(&mut detail_variable(grad.clone(), indent_num + 2usize));
        }
        _ => result.push(format!("{}  grad is None", indent)),
    }
    let creator = x.borrow().get_creator();
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
                    input.borrow().get_name().unwrap_or("None".to_string()),
                    input.borrow().get_data()
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
                // debug_variable(
                //     Variable::new(tmp_output.borrow().clone()),
                //     format!("{}{}", indent, indent),
                // );
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
    writeln!(file, "{}  name: {:?}", indent, x.borrow().get_name())?;
    writeln!(file, "{}  data: {:?}", indent, x.borrow().get_data())?;
    writeln!(
        file,
        "{}  generation: {:?}",
        indent,
        x.borrow().get_generation()
    )?;

    match x.borrow().get_grad() {
        Some(grad) => {
            writeln!(file, "{}  grad", indent)?;
            let tmp = detail_variable(grad.clone(), indent_num + 2usize);
            for line in tmp {
                writeln!(file, "{}", line)?;
            }
        }
        _ => writeln!(file, "{}  grad is None", indent)?,
    }
    let creator = x.borrow().get_creator();
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
                    input.borrow().get_name().unwrap_or("None".to_string()),
                    input.borrow().get_data()
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
                // debug_variable(
                //     Variable::new(tmp_output.borrow().clone()),
                //     format!("{}{}", indent, indent),
                // );
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
    let mut shape = gy.borrow().get_data().shape().to_vec();

    // Insert 1s at the appropriate positions
    actual_axis.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for a in actual_axis {
        shape.insert(a, 1);
    }

    // Reshape gy to the new shape
    let reshape_gy = gy
        .borrow()
        .get_data()
        .into_shape_with_order(IxDyn(&shape))
        .unwrap();
    let mut gy_clone = gy.clone();
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
        let shape = input.borrow().get_data().shape().to_vec();
        let values = input.borrow().get_data().flatten().to_vec();
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
                inputs_plus[i] =
                    Variable::new(RawVariable::from_shape_vec(shape.clone(), values_plus));
            }

            // j番目の要素からepsを引いた入力を作成
            {
                let mut values_minus = values.clone();
                values_minus[j] -= eps;
                inputs_minus[i] =
                    Variable::new(RawVariable::from_shape_vec(shape.clone(), values_minus));
            }

            // 順伝播を一度だけ実行
            let y_plus = function.forward(inputs_plus);
            let y_minus = function.forward(inputs_minus);

            // 中心差分を計算
            let diff = sum(&y_plus[0] - &y_minus[0], None, false);
            let grad_value = diff.borrow().get_data().flatten().to_vec()[0] / (2.0 * eps);

            grad_values.push(grad_value);
        }

        // 形状に合わせた勾配変数を作成して結果に追加
        let grad_var = Variable::new(RawVariable::from_shape_vec(shape, grad_values));
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
        let bp_grad = inputs[i].borrow().get_grad().unwrap().borrow().get_data();
        let num_grad = num_grads[i].borrow().get_data();

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
/// * plot_x (Vec<f64>): 学習データの X 軸の値
/// * plot_y (Vec<f64>): 学習データの Y 軸の値
/// * pred_xy (Vec<(f64, f64)>): 推論結果
/// * legend (&str): 凡例
pub fn draw_graph(
    caption: &str,
    file_path: &str,
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
        .build_cartesian_2d(0.0..1.0, -1.0..2.0)
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

#[cfg(test)]
mod test {
    use super::*;

    use ndarray::Array;
    use ndarray_rand::RandomExt;
    // use rand::prelude::*;
    use rand::{distributions::Uniform, SeedableRng};
    use rand_isaac::Isaac64Rng;

    // use std::time::{Duration, Instant};

    #[test]
    fn test_perform_1() {
        let seed = 0;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let x_var = Array::random_using((10, 20), Uniform::new(0., 10.), &mut rng);
        let x = Variable::new(RawVariable::from_shape_vec(
            vec![10, 20],
            x_var.flatten().to_vec(),
        ));

        let w_var = Array::random_using((20, 30), Uniform::new(0., 10.), &mut rng);
        let w = Variable::new(RawVariable::from_shape_vec(
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

        let x0 = Variable::new(RawVariable::from_shape_vec(vec![1, 10], x0_values));
        let x1 = Variable::new(RawVariable::from_shape_vec(vec![1, 10], x1_values));

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
        let x1 = Variable::new(RawVariable::new(5.0f32));
        x1.borrow_mut().set_name("x1".to_string());
        let x2 = Variable::new(RawVariable::new(10.0f32));
        x2.borrow_mut().set_name("x2".to_string());
        let x3 = Variable::new(RawVariable::new(15.0f32));
        x3.borrow_mut().set_name("x3".to_string());

        let result = &(&x1 * &x2) + &x3;
        debug_variable(result, 1);
    }

    #[test]
    fn test_plot_dot_graph_1() {
        let x = Variable::new(RawVariable::new(1));
        x.borrow_mut().set_name("x".to_string());
        let y = Variable::new(RawVariable::new(1));
        y.borrow_mut().set_name("y".to_string());
        let z = matyas(x.clone(), y.clone());

        let file_name = "test_plot_dot_graph_1.png";

        plot_dot_graph!(z, file_name, true);
    }

    /// get_dot_graph のテスト
    /// 掛け算と足し算
    #[test]
    fn test_get_dot_graph_1() {
        let x1 = Variable::new(RawVariable::new(5.0f32));
        x1.borrow_mut().set_name("x1".to_string());
        let x2 = Variable::new(RawVariable::new(10.0f32));
        x2.borrow_mut().set_name("x2".to_string());
        let x3 = Variable::new(RawVariable::new(15.0f32));
        x3.borrow_mut().set_name("x3".to_string());

        let result = &(&x1 * &x2) + &x3;
        let dot_txt = get_dot_graph!(result, true);
        println!("{}", dot_txt)
    }

    /// get_dot_graph のテスト
    /// Sphere
    #[test]
    fn test_get_dot_graph_sphere() {
        let x = Variable::new(RawVariable::new(1));
        x.borrow_mut().set_name("x".to_string());
        let y = Variable::new(RawVariable::new(1));
        y.borrow_mut().set_name("y".to_string());
        let z = sphere(x.clone(), y.clone());

        let dot_txt = get_dot_graph!(z, true);
        println!("{}", dot_txt)
    }

    /// get_dot_graph のテスト
    /// matyas
    #[test]
    fn test_get_dot_graph_matyas() {
        let x = Variable::new(RawVariable::new(1));
        x.borrow_mut().set_name("x".to_string());
        let y = Variable::new(RawVariable::new(1));
        y.borrow_mut().set_name("y".to_string());
        let z = matyas(x.clone(), y.clone());

        let dot_txt = get_dot_graph!(z, true);
        println!("{}", dot_txt)
    }

    /// get_dot_graph のテスト
    /// goldstein
    #[test]
    fn test_get_dot_graph_goldstein() {
        let x = Variable::new(RawVariable::new(1));
        x.borrow_mut().set_name("x".to_string());
        let y = Variable::new(RawVariable::new(1));
        y.borrow_mut().set_name("y".to_string());
        let z = goldstein(x.clone(), y.clone());

        let dot_txt = get_dot_graph!(z, true);
        println!("{}", dot_txt)
    }

    /// dot_func のテスト
    /// 掛け算１つのみ。
    #[test]
    fn test_dot_func_1() {
        let x1 = Variable::new(RawVariable::new(5.0f32));
        let x2 = Variable::new(RawVariable::new(10.0f32));

        let result = &x1 * &x2;
        let txt = dot_func(result.borrow().get_creator().unwrap());
        println!("{}", txt);
    }

    /// dot_func のテスト
    /// 掛け算と足し算
    #[test]
    fn test_dot_func_2() {
        let x1 = Variable::new(RawVariable::new(5.0f32));
        let x2 = Variable::new(RawVariable::new(10.0f32));
        let x3 = Variable::new(RawVariable::new(15.0f32));

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
        let var = Variable::new(RawVariable::new(2.0));
        var.borrow_mut().set_name("x".to_string());
        let result = dot_var!(var, true);
        println!("{}", result);
    }

    /// dot_var のテスト
    /// 行列、詳細情報を出力
    #[test]
    fn test_dot_var_2() {
        let var = Variable::new(RawVariable::from_shape_vec(
            vec![2, 2],
            vec![10, 20, 30, 40],
        ));
        var.borrow_mut().set_name("2x2dim".to_string());
        let result = dot_var!(var, true);
        println!("{}", result);
    }

    /// dot_var のテスト
    /// 行列、詳細情報なし
    #[test]
    fn test_dot_var_3() {
        let var = Variable::new(RawVariable::from_shape_vec(
            vec![2, 2],
            vec![10, 20, 30, 40],
        ));
        var.borrow_mut().set_name("2x2dim".to_string());
        let result = dot_var!(var);
        println!("{}", result);
    }
}
