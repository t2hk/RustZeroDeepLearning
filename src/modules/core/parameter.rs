// ライブラリを一括でインポート
use crate::modules::*;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ndarray::{Array, IxDyn};
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

use super::math::{reshape, transpose};

/// Parameter 構造体
/// RawData 構造体のラッパーである。
/// 順伝播や逆伝播について、所有権の共有や内部可変が必要であるため
/// Rc と RefCell で RawData を保持する。
#[derive(Debug, Clone)]
pub struct Parameter<V: MathOps> {
    raw: Rc<RefCell<RawData<V>>>,
}
