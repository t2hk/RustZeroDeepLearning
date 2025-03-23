// ライブラリを一括でインポート
use crate::modules::math::*;

use core::fmt::Debug;
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::ops::Add;
use std::rc::Rc;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    /// 割り算のテスト
    #[test]
    fn test_div() {}
}
