use std::cell::RefCell;
use std::rc::{Rc, Weak};

#[derive(Debug)]
struct Person {
    name: String,
    // 親子の参照はWeakで保持することで循環参照を防ぐ
    parent: Option<Weak<RefCell<Person>>>,
    child: Option<Weak<RefCell<Person>>>,
}

fn main() {
    let person_1st = Rc::new(RefCell::new(Person {
        name: "taro".to_string(),
        parent: None,
        child: None,
    }));

    let person_2nd = Rc::new(RefCell::new(Person {
        name: "jiro".to_string(),
        parent: None,
        child: None,
    }));

    let person_3rd = Rc::new(RefCell::new(Person {
        name: "saburo".to_string(),
        parent: None,
        child: None,
    }));

    let person_4th = Rc::new(RefCell::new(Person {
        name: "shiro".to_string(),
        parent: None,
        child: None,
    }));

    // 親子関係を設定する。
    person_1st.borrow_mut().child = Some(Rc::downgrade(&person_2nd.clone()));
    person_2nd.borrow_mut().parent = Some(Rc::downgrade(&person_1st.clone()));
    person_2nd.borrow_mut().child = Some(Rc::downgrade(&person_3rd.clone()));
    person_3rd.borrow_mut().parent = Some(Rc::downgrade(&person_2nd.clone()));
    person_3rd.borrow_mut().child = Some(Rc::downgrade(&person_4th.clone()));
    person_4th.borrow_mut().parent = Some(Rc::downgrade(&person_3rd.clone()));

    // 親を参照する。
    if let Some(weak_parent) = &person_4th.borrow().parent {
        if let Some(strong_parent) = weak_parent.upgrade() {
            println!("parent name: {:?}", strong_parent.borrow().name);
        } else {
            println!("unknow paret.");
        }
    };
}
