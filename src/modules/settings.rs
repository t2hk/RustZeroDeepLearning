// ライブラリを一括でインポート
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use num_traits::{Num, NumCast};
use std::cell::RefCell;
use std::rc::Rc;

pub trait MathOps:
    Num
    + NumCast
    + Clone
    + std::fmt::Debug
    + std::fmt::Display
    + std::ops::AddAssign
    + PartialOrd
    + 'static
{
}
impl<V> MathOps for V where
    V: Num
        + NumCast
        + Clone
        + std::fmt::Debug
        + std::fmt::Display
        + std::ops::AddAssign<V>
        + PartialOrd
        + 'static
{
}

thread_local!(
  static SETTING: Rc<RefCell<Setting>> = {
      Rc::new(RefCell::new(Setting { enable_backprop: true, retain_grad: false }))
  }
);

pub struct Setting {
    enable_backprop: bool,
    retain_grad: bool,
}
impl Setting {
    /// 逆伝播を有効にする。
    pub fn set_backprop_enabled() {
        SETTING.with(|setting| {
            let mut s = setting.borrow_mut();
            s.enable_backprop = true;
        });
    }

    /// 逆伝播を行わない。
    pub fn set_backprop_disabled() {
        SETTING.with(|setting| {
            let mut s = setting.borrow_mut();
            s.enable_backprop = false;
        });
    }

    /// 逆伝播が有効かどうか。
    ///
    /// Return
    /// * bool: 逆伝播が有効な場合は true を返す。
    /// デフォルトは true である。
    pub fn is_enable_backprop() -> bool {
        SETTING.with(|setting| setting.borrow().enable_backprop)
    }

    /// メモリ削減のため、逆伝播の中間変数について微分値を保持する。
    /// 初期値は false (微分値を保持しない)
    pub fn set_retain_grad_enabled() {
        SETTING.with(|setting| {
            let mut s = setting.borrow_mut();
            s.retain_grad = true;
        });
    }

    /// 中間変数の微分を保持しない(デフォルト)。
    pub fn set_retain_grad_disabled() {
        SETTING.with(|setting| {
            let mut s = setting.borrow_mut();
            s.retain_grad = false;
        });
    }

    /// 中間変数の微分を保持するかどうか。
    ///
    /// Return
    /// * bool: 中間変数の微分を保持する場合は true を返す。
    /// デフォルトは false である。
    pub fn is_enable_retain_grad() -> bool {
        SETTING.with(|setting| setting.borrow().retain_grad)
    }
}

/// 逆伝播を一時的に無効にするガード
/// 生成時に逆伝播を無効にし、スコープを抜けて破棄される際に元の設定に戻す。
pub struct NoGradGuard {
    prev: bool,
}

impl NoGradGuard {
    /// 逆伝播を無効にし、元の設定を保持したガードを生成する。
    pub fn new() -> Self {
        let prev = Setting::is_enable_backprop();
        Setting::set_backprop_disabled();
        NoGradGuard { prev }
    }
}

impl Drop for NoGradGuard {
    /// 元の設定が有効だった場合、逆伝播を有効に戻す。
    fn drop(&mut self) {
        if self.prev {
            Setting::set_backprop_enabled();
        }
    }
}
