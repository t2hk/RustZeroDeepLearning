use std::any::Any;
use std::cell::RefCell;

use std::rc::{Rc, Weak};

#[derive(Debug, Clone)]
struct Person {
    name: String,
    // 親子の参照はWeakで保持することで循環参照を防ぐ
    parent: Option<Rc<RefCell<Person>>>,
    child: Option<Rc<RefCell<Person>>>,
    creator: Option<Rc<RefCell<dyn Function>>>,
}

trait Function: Any + std::fmt::Debug {
    fn as_any(&self) -> &dyn Any;
    fn set_name(&mut self, name: String);
    fn get_name(&self) -> &String;
    fn set_creator(&self) -> Rc<RefCell<Person>>;
}

#[derive(Debug, Clone)]

struct Func1 {
    name: String,
}
impl Function for Func1 {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn set_name(&mut self, name: String) {
        self.name = name;
    }

    fn get_name(&self) -> &String {
        &self.name
    }

    fn set_creator(&self) -> Rc<RefCell<Person>> {
        let mut person_1st = Rc::new(RefCell::new(Person {
            name: "taro".to_string(),
            parent: None,
            child: None,
            creator: None,
        }));
        person_1st.borrow_mut().creator = Some(Rc::new(RefCell::new(self)));

        person_1st
    }
}
#[derive(Debug, Clone)]
struct Func2 {
    name: String,
}
impl Function for Func2 {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
    fn get_name(&self) -> &String {
        &self.name
    }
}

#[derive(Debug, Clone)]
struct Func3 {
    name: String,
}
impl Function for Func3 {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
    fn get_name(&self) -> &String {
        &self.name
    }
}

#[derive(Debug, Clone)]
struct Func4 {
    name: String,
}
impl Function for Func4 {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
    fn get_name(&self) -> &String {
        &self.name
    }
}

fn main() {
    let person_1st = Rc::new(RefCell::new(Person {
        name: "taro".to_string(),
        parent: None,
        child: None,
        creator: None,
    }));

    let person_2nd = Rc::new(RefCell::new(Person {
        name: "jiro".to_string(),
        parent: None,
        child: None,
        creator: None,
    }));

    let person_3rd = Rc::new(RefCell::new(Person {
        name: "saburo".to_string(),
        parent: None,
        child: None,
        creator: None,
    }));

    let person_4th = Rc::new(RefCell::new(Person {
        name: "shiro".to_string(),
        parent: None,
        child: None,
        creator: None,
    }));

    let f1 = Func1 {
        name: "func1".to_string(),
    };

    let f2 = Func2 {
        name: "func2".to_string(),
    };

    let f3 = Func3 {
        name: "func3".to_string(),
    };

    let f4 = Func4 {
        name: "func4".to_string(),
    };

    // 親子関係を設定する。
    person_1st.borrow_mut().child = Some(person_2nd.clone());
    person_1st.borrow_mut().creator = Some(Rc::new(RefCell::new(f1.clone())));
    person_2nd.borrow_mut().parent = Some(person_1st.clone());
    person_2nd.borrow_mut().creator = Some(Rc::new(RefCell::new(f2.clone())));
    person_2nd.borrow_mut().child = Some(person_3rd.clone());
    person_3rd.borrow_mut().parent = Some(person_2nd.clone());
    person_3rd.borrow_mut().creator = Some(Rc::new(RefCell::new(f3.clone())));
    person_3rd.borrow_mut().child = Some(person_4th.clone());
    person_4th.borrow_mut().parent = Some(person_3rd.clone());
    person_4th.borrow_mut().creator = Some(Rc::new(RefCell::new(f4.clone())));

    // 親を参照する。
    if let Some(parent) = &person_4th.borrow().parent {
        println!("parent name: {:?}", parent.borrow().name);
    } else {
        println!("unknow paret.");
    };

    // 親を繰り返し辿る。
    let mut current_person = person_4th.clone();
    loop {
        let parent = {
            if let Some(strong_parent) = &current_person.borrow().parent {
                println!(
                    "{:?}'s parent name: {:?}, creator: {:?}",
                    &current_person.borrow().name,
                    strong_parent.borrow().name,
                    &current_person
                        .borrow()
                        .creator
                        .clone()
                        .unwrap()
                        .borrow()
                        .get_name()
                );
                Some(Rc::clone(strong_parent))
            } else {
                println!("unknown parent.");
                None
            }
        };

        match parent {
            Some(next_parent) => current_person = next_parent,
            None => break,
        }
    }
}
