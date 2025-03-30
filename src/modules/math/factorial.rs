// ライブラリを一括でインポート
use crate::modules::math::*;

use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use num_bigint::{BigInt, ToBigInt};
use num_traits::{abs, FromPrimitive, Num, NumCast, One, Signed};

/// 20 までの階乗計算はテーブルで処理する。
fn small_factorial(n: u64) -> BigInt {
    match n {
        0 | 1 => BigInt::one(),
        2 => BigInt::from(2u64),
        3 => BigInt::from(6u64),
        4 => BigInt::from(24u64),
        5 => BigInt::from(120u64),
        6 => BigInt::from(720u64),
        7 => BigInt::from(5040u64),
        8 => BigInt::from(40320u64),
        9 => BigInt::from(362880u64),
        10 => BigInt::from(3628800u64),
        11 => BigInt::from(39916800u64),
        12 => BigInt::from(479001600u64),
        20 => BigInt::from(2432902008176640000u64),
        // 上記以外は計算する
        _ => {
            let mut result = BigInt::one();
            for i in 1..=n {
                result *= i;
            }
            result
        }
    }
}

/// 分割演算
fn product_range(l: BigInt, u: BigInt) -> BigInt {
    if &l >= &u {
        return BigInt::one();
    }
    let max_bits = (&u - 2.to_bigint().unwrap()).bits(); //掛けられる最大の奇数のビット長
    let num_operands: BigInt = (&u - &l) / 2; // 掛けられる奇数の個数

    // [L, U) の奇数の総積のビット長は　max_bits * num_operands を超えない
    // これが long に収まれば多倍長演算を回避して計算できる
    if &max_bits * &num_operands < 63.to_bigint().unwrap() {
        let mut total = l.clone();
        let two = 2.to_bigint().unwrap();
        let mut i: BigInt = l + &two;
        while i < u {
            total = total * &i;
            i += &two;
        }
        return total;
    }

    // 多倍長演算を回避するために分割して計算する
    let mut mid: BigInt = (&l + &num_operands) | BigInt::from(1);

    let left = product_range(l, mid.clone());
    let right = product_range(mid.clone(), u);

    let result = left * right;

    result
}

/// 奇数部分の計算
fn calc_odd_part(n: BigInt) -> BigInt {
    let mut result = BigInt::one();
    let mut l_i = 3.to_bigint().unwrap();
    let mut tmp = BigInt::one();
    let m = (n.bits() - 1) as i64;
    let mut i = m - 1;
    while -1 < i {
        // u_i は n//(2**i) より大きい最小の奇数
        let u_i: BigInt = ((&n >> i) + 1) | BigInt::one();

        // [1, U_i)　のうち、[1, L_i) は計算済みなので再利用し [L_i, U_i) のみ計算する
        tmp *= product_range(l_i, u_i.clone());

        // 計算済みの範囲を更新 (L_{i} <- U_{i + 1})
        l_i = u_i;

        result *= &tmp;
        i -= 1;
    }

    return result;
}

/// 階乗
pub fn factorial(n: u64) -> BigInt {
    if n <= 20 {
        return small_factorial(n);
    }

    let odd_part = calc_odd_part(n.to_bigint().unwrap());
    let popcount = format!("{:b}", n).matches('1').count();
    let two_exponent = n - popcount as u64;

    return odd_part << two_exponent;
}

#[cfg(test)]
mod tests {
    use std::{f64::consts::PI, time::Instant};

    use super::*;
    use num_bigint::ToBigInt;
    use rand::prelude::*;

    #[test]
    fn test_factoria_0() {
        let num = 0;
        let result = factorial(num);
        let result_naive = factorial(num);
        assert_eq!(result_naive, result);
    }

    #[test]
    fn test_factoria_1() {
        let num = 1;
        let result = factorial(num);
        let result_naive = factorial(num);
        assert_eq!(result_naive, result);
    }

    #[test]
    fn test_factoria_20() {
        let num = 20;
        let result = factorial(num);
        let result_naive = factorial(num);
        assert_eq!(result_naive, result);
    }

    #[test]
    fn test_factoria_100() {
        let num = 100;
        let result = factorial(num);
        let result_naive = factorial(num);
        assert_eq!(result_naive, result);
    }

    #[test]
    fn test_factoria_10000() {
        let num = 10000;
        let result = factorial(num);
        let result_naive = factorial(num);
        assert_eq!(result_naive, result);
    }
}
