use std::convert::From;
use std::ops::{AddAssign,MulAssign,Mul,DivAssign,Div,Neg,Add,Sub};
use std::fmt::Display;

pub trait Number:Copy+MulAssign+Sub<Output=Self>+Mul<Output=Self>+Add<Output=Self>+AddAssign+Default+Display+PartialEq+PartialOrd+Div<Output=Self>+DivAssign+Display+From<u8>{}
impl Number for f64{}
impl Number for f32{}
impl Number for i64{}
impl Number for i32{}
impl Number for usize{}

pub trait Signed:Number+Neg<Output=Self>{}
impl Signed for f64{}
impl Signed for f32{}
impl Signed for i64{}
impl Signed for i32{}

pub trait Float:Signed{fn exp(self:Self)->Self;}
impl Float for f64{fn exp(self:Self)->Self {self.exp()}}
impl Float for f32{fn exp(self:Self)->Self {self.exp()}}