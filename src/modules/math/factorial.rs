// ライブラリを一括でインポート
use crate::modules::math::*;

use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use num_bigint::{BigInt, BigUint};
use num_traits::{abs, Num, NumCast, One, Signed};
use std::cell::RefCell;
use std::ops::Div;
use std::rc::Rc;

/// 階乗関数
pub fn factorial(n: u64) -> BigInt {
    if n == 0 {
        return BigInt::one();
    }

    let mut result: BigInt = One::one();
    for i in 1..=n {
        result *= i;
    }
    result
}

pub fn my_sin<V: MathOps + Signed + PartialEq + PartialOrd>(x: Variable<V>) -> Variable<V> {
    let threshold = 0.0001;

    let mut y = Variable::new(RawVariable::new(V::from(0).unwrap()));

    for i in 1..=100000 {
        let num_2mul_i_1 = 2 * i + 1;
        let fact = factorial(num_2mul_i_1);
        let mut fact_raw_var = RawVariable::new(V::from(0).unwrap());
        fact_raw_var.set_bigint(Array::from_elem(IxDyn(&[]), fact));
        let fact_var = Variable::new(fact_raw_var);

        let minus1powi = (-1i64).pow((i as u64).try_into().unwrap());
        let c = minus1powi / &fact_var;
        let t = &c * &(&x ^ minus1powi as usize);
        dbg!(&t);

        y = &y + &t;
        let th = t.borrow().get_data()[[]];
        if abs(th) < V::from(threshold).unwrap() {
            break;
        }
    }

    return y;
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_val() {
        let x = RawVariable::new(0i32);
        let val = x.get_data();
        let num = val[[]];

        dbg!(&num);
    }

    #[test]
    fn test_my_sin() {
        let x = Variable::new(RawVariable::new(PI / 4.0));
        let result = my_sin(x);
        dbg!(&result);
    }

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
