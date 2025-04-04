// ライブラリを一括でインポート
use crate::modules::*;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::cell::RefCell;
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
                v_name = format!("{}: ", tmp_v_name);
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

        let mut creators = FunctionExecutor::extract_creators(vec![$variable.clone()]);

        while let Some(creator) = creators.pop() {
            let local_dot_func_txt = dot_func(creator.1.clone());
            txt = format!("{}{}", txt, local_dot_func_txt);

            let inputs = creator.1.borrow().get_inputs();

            for input in inputs {
                let local_dot_var_txt = $crate::dot_var!(input, $verbose);
                txt = format!("{}{}", txt, local_dot_var_txt);
            }
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

#[cfg(test)]
mod test {
    use super::*;

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
