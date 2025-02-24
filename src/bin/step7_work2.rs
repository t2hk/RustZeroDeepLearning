use core::fmt::Debug;
use std::any::Any;

trait Function: Any + std::fmt::Debug {
    fn call(&self);
    fn set_creator(&mut self, creator: Box<dyn Function>);
    fn get_creator(&self) -> Option<&Box<dyn Function>>;
    fn as_any(&self) -> &dyn Any;
}

#[derive(Debug)]
struct Square {
    creator: Option<Box<dyn Function>>,
}

#[derive(Debug)]
struct Exp {
    creator: Option<Box<dyn Function>>,
}

impl Function for Square {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn call(&self) {
        println!("Square");
    }

    fn set_creator(&mut self, creator: Box<dyn Function>) {
        self.creator = Some(creator);
    }

    fn get_creator(&self) -> Option<&Box<dyn Function>> {
        self.creator.as_ref()
    }
}

impl Function for Exp {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn call(&self) {
        println!("Exp");
    }
    fn set_creator(&mut self, creator: Box<dyn Function>) {
        self.creator = Some(creator);
    }
    fn get_creator(&self) -> Option<&Box<dyn Function>> {
        self.creator.as_ref()
    }
}

fn main() {
    let mut square_1 = Box::new(Square { creator: None });
    let mut exp_1 = Box::new(Exp { creator: None });
    let mut square_2 = Box::new(Square { creator: None });
    let mut exp_2 = Box::new(Exp { creator: None });

    square_1.set_creator(exp_1);
    exp_2.set_creator(square_1);
    square_2.set_creator(exp_2);

    let mut current_func: &dyn Function = square_2.as_ref();

    while let Some(creator) = current_func.get_creator() {
        println!("=== creator ===");
        creator.call();
        println!("====================");
        current_func = creator.as_ref();
    }
}
