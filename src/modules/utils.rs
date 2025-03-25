// ライブラリを一括でインポート
use crate::modules::*;

use ndarray::{Array, IxDyn};
use std::any::Any;
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

/// Variable を graphviz の DOT 言語で出力する。
///
/// Arguments:
/// * variable (Variable<V>): 出力対象の変数
/// * verbose (bool): 詳細を出力するかどうか(省略可能)
///
/// Returns:
/// * String: 変数の DOT 言語表記
macro_rules! dot_var {
    // verbose を指定しない場合は false で実行する。
    ($variable:ident) => {{
        dot_var!($variable, false)
    }};
    // verbose の指定がある場合の実行
    ($variable:ident, $verbose:ident) => {{
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
        let v_id = &$variable as *const _ as usize;
        let result = format!(
            "{} [label=\"{}\", color=orange, style=filled]\n",
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
fn dot_func<V: MathOps>(fe: Rc<RefCell<FunctionExecutor<V>>>) -> String {
    let inputs = fe.borrow().get_inputs();
    let outputs = fe.borrow().get_outputs();

    // この関数のポインタを ID として使用する。
    let f_id = &fe.borrow().get_creator().as_ptr();
    let f_name = &fe.borrow().get_creator().borrow().get_name();

    // 関数の情報を DOT 表記する。
    let mut txt = format!(
        "{:?} [label={:?}, color=lightblue, style=filled, shape=box]\n",
        &f_id, &f_name
    );

    // 入力値と関数の関係を DOT 表記する。
    // input のポインタを入力値の ID とする。
    for input in inputs {
        let input_id = &input.as_ref().as_ptr();
        let local_txt = format!("{:?} -> {:?}\n", &input_id, &f_id);
        txt = format!("{}{}", txt, local_txt);
    }

    // 出力値と関数の関係を DOT 表記する。
    // output のポインタを出力値の ID とする。
    for output in outputs {
        let output_id = &output.upgrade().unwrap().as_ptr();
        let local_txt = format!("{:?} -> {:?}\n", &f_id, &output_id);
        txt = format!("{}{}", txt, local_txt);
    }

    txt.to_string()
}

#[cfg(test)]
mod test {
    use super::*;

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
