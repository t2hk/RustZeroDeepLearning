// ライブラリを一括でインポート
use crate::modules::*;

use ndarray::{Array, IxDyn};
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
            "{} [label=\"{}\", color=orange, style=filled]¥n",
            v_id,
            v_name.to_string()
        );
        result
    }};
}

#[cfg(test)]
mod test {
    use super::*;

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
