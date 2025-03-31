// ライブラリを一括でインポート
use crate::modules::*;

use core::ops::{Add, Div, Mul, Rem, Sub};
use num_bigint::BigInt;
use num_traits::{Bounded, Num, NumCast, One, Zero};
use std::fmt;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use std::rc::Rc;
use std::str::FromStr;

/// BigInt のラッパー構造体
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BigIntWrapper(pub Rc<BigInt>);

impl BigIntWrapper {
    pub fn new(value: BigInt) -> Self {
        BigIntWrapper(Rc::new(value))
    }

    pub fn inner(&self) -> &BigInt {
        &self.0
    }
}

/// 加算
impl Add for BigIntWrapper {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        BigIntWrapper(Rc::new(self.0.as_ref().clone() + other.0.as_ref().clone()))
    }
}

/// 引き算
impl Sub for BigIntWrapper {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        BigIntWrapper(Rc::new(self.0.as_ref().clone() - other.0.as_ref().clone()))
    }
}

/// 掛け算
impl Mul for BigIntWrapper {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        BigIntWrapper(Rc::new(self.0.as_ref().clone() * other.0.as_ref().clone()))
    }
}

/// 割り算
impl Div for BigIntWrapper {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        BigIntWrapper(Rc::new(self.0.as_ref().clone() / other.0.as_ref().clone()))
    }
}

/// 剰余算
impl Rem for BigIntWrapper {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        BigIntWrapper(Rc::new(self.0.as_ref().clone() % other.0.as_ref().clone()))
    }
}

// 代入演算子
impl AddAssign for BigIntWrapper {
    fn add_assign(&mut self, other: Self) {
        self.0 = Rc::new(self.0.as_ref().clone() + other.0.as_ref().clone());
    }
}

// 代入演算子
impl MulAssign for BigIntWrapper {
    fn mul_assign(&mut self, other: Self) {
        self.0 = Rc::new(self.0.as_ref().clone() * other.0.as_ref().clone());
    }
}

// 代入演算子
impl SubAssign for BigIntWrapper {
    fn sub_assign(&mut self, other: Self) {
        self.0 = Rc::new(self.0.as_ref().clone() - other.0.as_ref().clone());
    }
}

// 代入演算子
impl DivAssign for BigIntWrapper {
    fn div_assign(&mut self, other: Self) {
        self.0 = Rc::new(self.0.as_ref().clone() / other.0.as_ref().clone());
    }
}

/// num-traitsの Zero を実装
impl Zero for BigIntWrapper {
    fn zero() -> Self {
        BigIntWrapper(Rc::new(BigInt::zero()))
    }

    fn is_zero(&self) -> bool {
        self.0 == Rc::new(BigInt::zero())
    }
}

/// num-tratis の One を実装
impl One for BigIntWrapper {
    fn one() -> Self {
        BigIntWrapper(Rc::new(BigInt::one()))
    }
}

/// num-traits 用に Displayの実装
impl fmt::Display for BigIntWrapper {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Numトレイトの実装
impl Num for BigIntWrapper {
    type FromStrRadixErr = <BigInt as Num>::FromStrRadixErr;

    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        BigInt::from_str_radix(s, radix).map(|b| BigIntWrapper(Rc::new(b)))
    }
}

// NumCastトレイトの実装
impl NumCast for BigIntWrapper {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        n.to_i64()
            .and_then(|i| Some(BigIntWrapper(Rc::new(BigInt::from(i)))))
    }
}

// FromStrトレイトの実装
impl FromStr for BigIntWrapper {
    type Err = <BigInt as FromStr>::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        BigInt::from_str(s).map(|b| BigIntWrapper(Rc::new(b)))
    }
}

// ToPrimitiveトレイトの実装（NumCastと組み合わせて使用するため）
impl num_traits::ToPrimitive for BigIntWrapper {
    fn to_i128(&self) -> Option<i128> {
        self.0.to_i128()
    }

    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    fn to_i32(&self) -> Option<i32> {
        self.0.to_i32()
    }

    fn to_i16(&self) -> Option<i16> {
        self.0.to_i16()
    }

    fn to_i8(&self) -> Option<i8> {
        self.0.to_i8()
    }

    fn to_u128(&self) -> Option<u128> {
        if *self.0.as_ref() >= BigInt::zero() {
            self.0.to_u128()
        } else {
            None
        }
    }

    fn to_u64(&self) -> Option<u64> {
        if *self.0.as_ref() >= BigInt::zero() {
            self.0.to_u64()
        } else {
            None
        }
    }

    fn to_u32(&self) -> Option<u32> {
        if *self.0.as_ref() >= BigInt::zero() {
            self.0.to_u32()
        } else {
            None
        }
    }

    fn to_u16(&self) -> Option<u16> {
        if *self.0.as_ref() >= BigInt::zero() {
            self.0.to_u16()
        } else {
            None
        }
    }

    fn to_u8(&self) -> Option<u8> {
        if *self.0.as_ref() >= BigInt::zero() {
            self.0.to_u8()
        } else {
            None
        }
    }
}

// Boundedトレイトの実装
impl Bounded for BigIntWrapper {
    fn min_value() -> Self {
        let big_min = BigInt::from(-1) * BigIntWrapper::max_value().0.as_ref();
        BigIntWrapper(Rc::new(big_min))
    }

    fn max_value() -> Self {
        let max_str = "999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999".repeat(1_000);
        let big_max = max_str.parse::<BigInt>().unwrap();
        BigIntWrapper(Rc::new(big_max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::ToBigInt;

    /// from_str のテスト
    #[test]
    fn test_from_str() {
        let result = BigIntWrapper::from_str("10000").expect("can not convert to BigInt");
        let expect = BigIntWrapper(Rc::new(10000.to_bigint().unwrap()));
        assert_eq!(expect, result);
        println!("{}", expect);
    }

    /// 加算のテスト
    #[test]
    fn test_bigint_sum() {
        let bigint1 = BigIntWrapper(Rc::new(1000.to_bigint().unwrap()));
        let bigint2 = BigIntWrapper(Rc::new(2000.to_bigint().unwrap()));

        let expected = BigIntWrapper(Rc::new(3000.to_bigint().unwrap()));

        let result = bigint1.clone().add(bigint2.clone());
        assert_eq!(expected, result);
    }

    /// 引き算のテスト
    /// 結果がプラス値
    #[test]
    fn test_bigint_minus1() {
        let bigint1 = BigIntWrapper(Rc::new(3000.to_bigint().unwrap()));
        let bigint2 = BigIntWrapper(Rc::new(1000.to_bigint().unwrap()));

        let expected = BigIntWrapper(Rc::new(2000.to_bigint().unwrap()));

        let result = bigint1.clone().sub(bigint2.clone());
        assert_eq!(expected, result);
    }

    /// 引き算のテスト
    /// 結果がマイナス値
    #[test]
    fn test_bigint_minus2() {
        let bigint1 = BigIntWrapper(Rc::new(1000.to_bigint().unwrap()));
        let bigint2 = BigIntWrapper(Rc::new(3000.to_bigint().unwrap()));

        let expected = BigIntWrapper(Rc::new(-2000.to_bigint().unwrap()));

        let result = bigint1.clone().sub(bigint2.clone());
        assert_eq!(expected, result);
    }

    /// 掛け算のテスト
    #[test]
    fn test_bigint_mul() {
        let bigint1 = BigIntWrapper(Rc::new(3000.to_bigint().unwrap()));
        let bigint2 = BigIntWrapper(Rc::new(1000.to_bigint().unwrap()));

        let expected = BigIntWrapper(Rc::new(3000000.to_bigint().unwrap()));

        let result = bigint1.clone().mul(bigint2.clone());
        assert_eq!(expected, result);
    }

    /// 割り算のテスト
    #[test]
    fn test_bigint_div() {
        let bigint1 = BigIntWrapper(Rc::new(3000.to_bigint().unwrap()));
        let bigint2 = BigIntWrapper(Rc::new(1000.to_bigint().unwrap()));

        let expected = BigIntWrapper(Rc::new(3.to_bigint().unwrap()));

        let result = bigint1.clone().div(bigint2.clone());
        assert_eq!(expected, result);
    }

    /// 剰余算のテスト1
    /// 割り切れるパターン
    #[test]
    fn test_bigint_rem1() {
        let bigint1 = BigIntWrapper(Rc::new(3000.to_bigint().unwrap()));
        let bigint2 = BigIntWrapper(Rc::new(1000.to_bigint().unwrap()));

        let expected = BigIntWrapper(Rc::new(0.to_bigint().unwrap()));

        let result = bigint1.clone().rem(bigint2.clone());
        assert_eq!(expected, result);
    }

    /// 剰余算のテスト2
    /// 余りがあるパターン
    #[test]
    fn test_bigint_rem2() {
        let bigint1 = BigIntWrapper(Rc::new(31100.to_bigint().unwrap()));
        let bigint2 = BigIntWrapper(Rc::new(1000.to_bigint().unwrap()));

        let expected = BigIntWrapper(Rc::new(100.to_bigint().unwrap()));

        let result = bigint1.clone().rem(bigint2.clone());
        assert_eq!(expected, result);
    }

    #[test]
    fn test_add_assigin() {
        let mut bigint1 = BigIntWrapper(Rc::new(1000.to_bigint().unwrap()));
        let bigint2 = BigIntWrapper(Rc::new(2000.to_bigint().unwrap()));

        let expected = BigIntWrapper(Rc::new(3000.to_bigint().unwrap()));

        bigint1 += bigint2;
        assert_eq!(expected, bigint1);
    }

    #[test]
    fn test_mul_assigin() {
        let mut bigint1 = BigIntWrapper(Rc::new(1000.to_bigint().unwrap()));
        let bigint2 = BigIntWrapper(Rc::new(2000.to_bigint().unwrap()));

        let expected = BigIntWrapper(Rc::new(2000000.to_bigint().unwrap()));

        bigint1 *= bigint2;
        assert_eq!(expected, bigint1);
    }

    #[test]
    fn test_sub_assigin() {
        let mut bigint1 = BigIntWrapper(Rc::new(3000.to_bigint().unwrap()));
        let bigint2 = BigIntWrapper(Rc::new(1000.to_bigint().unwrap()));

        let expected = BigIntWrapper(Rc::new(2000.to_bigint().unwrap()));

        bigint1 -= bigint2;
        assert_eq!(expected, bigint1);
    }

    #[test]
    fn test_div_assigin() {
        let mut bigint1 = BigIntWrapper(Rc::new(3000.to_bigint().unwrap()));
        let bigint2 = BigIntWrapper(Rc::new(200.to_bigint().unwrap()));

        let expected = BigIntWrapper(Rc::new(15.to_bigint().unwrap()));

        bigint1 /= bigint2;
        assert_eq!(expected, bigint1);
    }
}
