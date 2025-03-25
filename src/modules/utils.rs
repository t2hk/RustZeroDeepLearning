// ライブラリを一括でインポート
use crate::modules::*;

use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

// fn dot_var()
macro_rules! dot_var {
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

    #[test]
    fn test_dot_var_1() {
        let var = Variable::new(RawVariable::new(2.0));
        var.borrow_mut().set_name("x".to_string());
        let result = dot_var!(var, true);
        println!("{}", result);
    }

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
}
