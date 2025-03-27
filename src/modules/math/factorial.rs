// ライブラリを一括でインポート
use crate::modules::math::*;

use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use num_bigint::{BigInt, BigUint};
use num_traits::One;
use std::cell::RefCell;
use std::ops::Div;
use std::rc::Rc;

/// 階乗関数
pub fn factorial(n: u64) -> BigUint {
    if n == 0 {
        return BigUint::one();
    }

    let mut result: BigUint = One::one();
    for i in 1..=n {
        result *= i;
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    #[test]
    /// 除算のテスト(f32)
    fn test_div_1() {
        // 順伝播
        // let x1 = Variable::new(RawVariable::new(10.0f32));
        // let x2 = Variable::new(RawVariable::new(2.0f32));
        // let expected = RawVariable::new(5.0f32);

        // let result = div(x1, x2);
        // assert_eq!(expected.get_data(), result.borrow().get_data());

        let value = "200001";
        match value.parse::<u64>() {
            Ok(n) => {
                let fact = factorial(n);
                println!("{}の階乗: {}", n, fact);
                println!("桁数: {}", fact.to_string().len());
            }
            _ => {
                println!("エラー");
            }
        }
    }
}
