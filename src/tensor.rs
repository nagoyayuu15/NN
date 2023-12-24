use crate::general_type::Number;
use std::fmt::Display;
use std::ops::{Add,Mul,Index,IndexMut,RangeInclusive};
use std::fs::File;
use std::io::Write;

pub type Shape = Vec<usize>;
pub type ShapedIndex = Vec<usize>;
pub type Data<T> = Vec<T>;
pub struct Tensor<N:Number>{shape:Shape,data:Data<N>}
pub fn tensor<N:Number>(shape:Shape,data:Data<N>)->Tensor<N>{
    Tensor{shape,data}
}
pub fn zero_tensor<N:Number>(shape:Shape)->Tensor<N>{
    let mut size = 1usize;
    for v in shape.clone(){
        size *= v;
    }
    let data:Data<N> = vec![N::default();size];
    Tensor{shape,data}
}



pub trait IndedxConsistencyConfirmation{
    fn check_shaped_index(&self,index:&Vec<usize>)->bool;
    fn check_flat_index(&self,index:usize)->bool;
}
pub trait IndexConversion:IndedxConsistencyConfirmation{
    fn flatten(&self, shaped_index:&Vec<usize>)->usize;
    fn shape(&self, flat_index:usize)->Vec<usize>;
}
pub trait ElementalOperation<N> {
    fn get_value(&self, flat_index:usize)->&N;
    fn get_reference(&mut self, flat_index:usize)->&mut N;
}
pub trait MethodChain<N:Number>:IndexConversion+ElementalOperation<N>
{
    fn scaler_add(self,operand:N)->Tensor<N>;
    fn scaler_mul(self,operand:N)->Tensor<N>;
    fn inner(self,shape_degeneration:usize,operand:&Tensor<N>)->Tensor<N>;
    fn marge(self,operand:&Tensor<N>)->Tensor<N>;
    fn tensor_add(self,operand:&Tensor<N>)->Tensor<N>;
    fn tensor_mul(self,operand:&Tensor<N>)->Tensor<N>;
}
pub trait GetPart<N:Number>:ElementalOperation<N>{
    fn get_part(self,range:RangeInclusive<ShapedIndex>)->Tensor<N>;
}
pub trait Paste<N:Number>:ElementalOperation<N>{
    fn paste(self,to:ShapedIndex,data:&Tensor<N>)->Tensor<N>;
}
pub trait GrossCalculation<N:Number>{
    fn gross_sum(&self)->N;
    fn gross_product(&self)->N;
}
pub trait FileOperation<N:Number>{
    fn dump(&self,name:&String)->bool;
    fn load(&self,name:&String)->bool;
}


pub struct TupleRepeativePermutation{no_less_than:ShapedIndex,no_more_than:ShapedIndex,current:ShapedIndex}
// fn partial_tuple_repeative_permutation_i(no_less_than: ShapedIndex,no_more_than: ShapedIndex)->TupleRepeativePermutation{
//     TupleRepeativePermutation{no_less_than:no_less_than.clone(),no_more_than:no_more_than,current:no_less_than}
// }
// fn tuple_repeative_permutation_i(no_more_than: ShapedIndex)->TupleRepeativePermutation{
//     TupleRepeativePermutation{no_less_than:vec![0;no_more_than.len()],no_more_than:no_more_than.clone(),current:vec![0;no_more_than.len()]}
// }
// fn partial_tuple_repeative_permutation_e(no_less_than: ShapedIndex,mut less_than: ShapedIndex)->TupleRepeativePermutation{
//     for v in &mut less_than {*v -= 1;}
//     TupleRepeativePermutation{no_less_than:no_less_than.clone(),no_more_than:less_than,current:no_less_than}
// }
fn tuple_repeative_permutation_e(mut less_than: ShapedIndex)->TupleRepeativePermutation{
    for v in &mut less_than {
        *v -= 1;
    }
    TupleRepeativePermutation{no_less_than:vec![0;less_than.len()],no_more_than:less_than.clone(),current:vec![0;less_than.len()]}
}
impl Iterator for TupleRepeativePermutation{
    type Item = ShapedIndex;
    fn next(&mut self) -> Option<Self::Item> {
        let res = self.current.clone();
        if self.current.len() == 0 {
            self.current.push(1);
            self.no_more_than.push(0);
            return Some(vec![]);
        }
        if res[self.current.len()-1] > self.no_more_than[self.current.len()-1] {return None;}
        self.current[0] += 1;
        for i in 0..self.current.len()-1{
            if self.current[i] > self.no_more_than[i]{
                self.current[i] = self.no_less_than[i];
                self.current[i+1] += 1;
            }
        }
        return Some(res);
    }
}


// impl<N:Number> FileOperation for Tensor<N>{
//     fn dump(&self,name:&String)->bool {
        
//     }
//     fn load(&self,name:&String)->bool {
        
//     }
// }
impl<N:Number> IndedxConsistencyConfirmation for Tensor<N>{
    fn check_shaped_index(&self,index:&Vec<usize>)->bool {
        if self.shape.len()!=index.len(){return false;}
        for i in 0..self.shape.len()
        {
            if self.shape[i]<=index[i]{return false;}
        }
        return true;
    }
    fn check_flat_index(&self,index:usize)->bool {
        if index<self.data.len() {return true;}
        return false;
    }
}
impl<N:Number> IndexConversion for Tensor<N>{
    fn flatten(&self, shaped_index:&Vec<usize>)->usize {
        debug_assert!(self.check_shaped_index(shaped_index));
        let mut res: usize = 0;
        let mut weight: usize = 1;
        for i in 0..shaped_index.len() {
            res += weight*shaped_index[i];
            weight *= self.shape[i];
        }
        res
    }
    fn shape(&self, flat_index:usize)->Vec<usize> {
        debug_assert!(self.check_flat_index(flat_index));
        let mut buf = flat_index;
        let mut res=Vec::<usize>::new();
        let mut weight: usize = 1;
        for s in &self.shape{
            weight *= s;
        }
        for s in 1..=self.shape.len()
        {
            weight /= self.shape[self.shape.len()-s];
            res.insert(0,buf/weight);
            buf %= weight; 
        }
        res
    }
}
impl<N:Number> ElementalOperation<N> for Tensor<N>{
    fn get_reference(&mut self, flat_index:usize)->&mut N {
       &mut self.data[flat_index] 
    }
    fn get_value(&self, flat_index:usize)->&N {
        &self.data[flat_index]
    }
}
impl<N:Number> MethodChain<N> for Tensor<N>{
    fn scaler_add(mut self,operand:N)->Tensor<N> {
        for v in &mut self.data{
            *v += operand;
        }
        self
    }
    fn scaler_mul(mut self,operand:N)->Tensor<N> {
        for v in &mut self.data{
            *v *= operand;
        }
        self
    }
    fn tensor_add(mut self,operand:&Tensor<N>)->Tensor<N> {
        debug_assert_eq!(self.shape,operand.shape);
        for i in 0..self.data.len(){
            self.data[i] += operand.data[i];
        }
        self
    }
    fn tensor_mul(mut self,operand:&Tensor<N>)->Tensor<N> {
        debug_assert_eq!(self.shape,operand.shape);
        for i in 0..self.data.len(){
            self.data[i] *= operand.data[i];
        }
        self
    }
    fn inner(self,shape_degeneration:usize,operand:&Tensor<N>)->Tensor<N> {
        debug_assert_eq!(self.shape[self.shape.len()-shape_degeneration..],operand.shape[..shape_degeneration]);
        
        let left_shape = self.shape[..self.shape.len()-shape_degeneration].to_vec();
        let degenerated_shape = operand.shape[..shape_degeneration].to_vec();
        let right_shape = operand.shape[shape_degeneration..].to_vec();
        
        let mut result = zero_tensor([left_shape.clone(),right_shape.clone()].concat());
        
        for l in tuple_repeative_permutation_e(left_shape.clone()){
            for r in tuple_repeative_permutation_e(right_shape.clone()){
                let p = result.get_reference(
                    result.flatten(
                        &[l.clone(),r.clone()].concat()
                    )
                );
                for k in tuple_repeative_permutation_e(degenerated_shape.clone()){
                    *p += *self.get_value(self.flatten(
                        &[l.clone(),k.clone()].concat()
                    ))**operand.get_value(operand.flatten(
                        &[k.clone(),r.clone()].concat()
                    ));
                }
            }
        }

        result
    }
    fn marge(self,operand:&Tensor<N>)->Tensor<N> {
        self.inner(operand.shape.len(),operand)
    }
}
impl<N:Number> GetPart<N> for Tensor<N>{
    fn get_part(self,range:RangeInclusive<ShapedIndex>)->Tensor<N> {
        debug_assert!(self.check_shaped_index(&range.start()));
        debug_assert!(self.check_shaped_index(&range.end()));
        
        let mut new_shape:Shape = vec![0;range.start().len()];
        for i in 0..range.start().len(){
            new_shape[i] = range.end()[i] -range.start()[i] + 1;
        }

        let mut res = zero_tensor::<N>(new_shape.clone());
        for shaped_index in tuple_repeative_permutation_e(new_shape){
            let mut shaped_index = shaped_index.clone();
            let p = res.get_reference(res.flatten(&shaped_index));
            //shift shaped_index
            for i in 0..shaped_index.len(){
                shaped_index[i] += range.start()[i];
            }
            *p = *self.get_value(self.flatten(&shaped_index))
        }
        res
    }
}
impl<N:Number> Paste<N> for Tensor<N>{
    fn paste(mut self,at:ShapedIndex,board:&Tensor<N>)->Tensor<N> {
        for shaped_index in tuple_repeative_permutation_e(board.shape.clone()){
            let mut shaped_index = shaped_index.clone();
            let v = board.get_value(board.flatten(&shaped_index));
            //shift shaped_index
            for si in 0..shaped_index.len(){
                shaped_index[si] += at[si]
            }
            let p = self.get_reference(self.flatten(&shaped_index));
            *p = *v;
        }
        self
    }
}
impl<N:Number> GrossCalculation<N> for Tensor<N>{
    fn gross_product(&self)->N {
        let mut buf = N::from(1);
        for v in &self.data{
            buf *= *v;
        }
        buf
    }
    fn gross_sum(&self)->N {
        let mut buf = N::from(1);
        for v in &self.data{
            buf += *v;
        }
        buf        
    }
}
impl<N:Number> Clone for Tensor<N>{
    fn clone(&self) -> Self {
        Tensor { shape: self.shape.clone(), data: self.data.clone() }
    }
}
impl<N:Number> Display for Tensor<N>{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut buf = String::new();

        //header
        buf += "--------------------";
        buf += "(";for s in &self.shape{buf += &s.to_string();buf += ","}buf += ")";
        buf += "--------------------\n";

        //first line's prefix
        let mut k:Vec<usize> = Vec::<usize>::new();
        for shaped_index in tuple_repeative_permutation_e(self.shape.clone())
        {
            if shaped_index.len() > 1 && shaped_index[1..].to_vec() != k//when the second rightest index changes
            {   
                // other line's prefix
                buf += "\n(*,";
                for i in 1..self.shape.len(){
                    buf += &shaped_index[i].to_string();
                    buf += ",";
                }
                buf += ") \t:";
                k = shaped_index[1..].to_vec();
            }
            //value
            buf += &self.get_value(self.flatten(&shaped_index)).to_string();
            buf += "\t";
        }

        //footer
        buf += "\n--------------------";
        buf += "(";for s in &self.shape{buf += &s.to_string();buf += ","}buf += ")";
        buf += "--------------------";

        write!(f,"{}",buf)
    }
}
impl<N:Number> Add<N> for Tensor<N>{
    type Output = Tensor<N>;
    fn add(self, rhs: N) -> Self::Output {
        self.scaler_add(rhs)
    }
}
impl<N:Number> Add<Tensor<N>> for Tensor<N>{
    type Output = Tensor<N>;
    fn add(self, rhs: Tensor<N>) -> Self::Output {
        self.tensor_add(&rhs)
    }
}
impl<N:Number> Mul<N> for Tensor<N>{
    type Output = Tensor<N>;
    fn mul(self, rhs: N) -> Self::Output {
        self.scaler_mul(rhs)
    }
}
impl<N:Number> Mul<Tensor<N>> for Tensor<N>{
    type Output = Tensor<N>;
    fn mul(self, rhs: Tensor<N>) -> Self::Output {
        self.tensor_mul(&rhs)
    }
}
impl<N:Number> Index<usize> for Tensor<N>{
    type Output = N;
    fn index(&self, index: usize) -> &Self::Output {
        self.get_value(index)
    }
}
impl<N:Number> IndexMut<usize> for Tensor<N>{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_reference(index)
    }
}
impl<N:Number> Index<ShapedIndex> for Tensor<N>{
    type Output = N;
    fn index(&self, index: ShapedIndex) -> &Self::Output {
        self.get_value(self.flatten(&index))
    }
}
impl<N:Number> IndexMut<ShapedIndex> for Tensor<N>{
    fn index_mut(&mut self, index: ShapedIndex) -> &mut Self::Output {
        self.get_reference(self.flatten(&index))
    }
}


#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn inner_test(){
        let tensor1 = tensor::<i32>(vec![2,2],vec![
            -1,-2,
            -3,-4
        ]);
        let tensor2 = tensor::<i32>(vec![2,2],vec![
            1,2,
            3,4
        ]);
        println!("{:}",&tensor1);
        println!("{:}",&tensor2);
        println!("{:}",tensor1.inner(1,&tensor2))
    }
    #[test]
    fn index_test(){
        let tensor1 = zero_tensor::<i64>(vec![3,2,2]);
        for v in 0..12{
            debug_assert_eq!(tensor1.flatten(&tensor1.shape(v)),v);
        }
        let tensor2 = tensor::<i64>(vec![2,2],vec![
            1,2,
            3,4
        ]);
        println!("{}",tensor2[0]);
        println!("{}",tensor2[vec![1,1]]);
    }
    #[test]
    fn marge_test(){
        let tensor1 = tensor::<i64>(vec![3,3,2,2],vec![
            1,2,3,10,20,30,100,200,300,
            4,5,6,40,50,60,400,500,600,
            1,2,3,10,20,30,100,200,300,
            4,5,6,40,50,60,400,500,600
        ]);
        let tensor2 = tensor::<i64>(vec![2,2],vec![
            0,0,
            0,1
        ]);
        println!("{:}",&tensor1);
        println!("{:}",&tensor2);
        println!("{:}",tensor1.marge(&tensor2))
    }
    #[test]
    fn get_part_test(){
        let tensor1 = tensor::<i64>(vec![3,3,2,2],vec![
            1,2,3,10,20,30,100,200,300,
            4,5,6,40,50,60,400,500,600,
            1,2,3,10,20,30,100,200,300,
            4,5,6,40,50,60,400,500,600
        ]);
        println!("{}",tensor1);
        let tensor2 = tensor1.get_part(vec![0,0,0,0]..=vec![1,1,0,0]);
        println!("{}",tensor2);
    }
    #[test]
    fn shape_test(){
        let tensor1 = tensor::<i64>(vec![3,1,2,2],vec![
            1,2,3,4,
            5,6,7,8,
            9,10,11,12
        ]);
        println!("{}",tensor1);
        let tensor2 = tensor::<i64>(vec![3,1,2],vec![1,2,3,4,5,6]);
        println!("{}",tensor2);
    }
    #[test]
    fn paste_test(){
        let tensor1 = zero_tensor::<i64>(vec![6,6]);
        println!("{}",tensor1);
        let tensor2 = tensor::<i64>(vec![3,3],vec![1,2,3,4,5,6,7,8,9]);
        println!("{}",tensor2.clone());
        let tensor2 = tensor2.get_part(vec![0,0]..=vec![1,1]);
        println!("{}",tensor1.paste(vec![4,4], &tensor2));
    }
    #[test]
    fn calc_test(){
        let tensor1 = tensor::<i64>(vec![3,3],vec![
            2,0,0,
            0,2,0,
            0,0,2
        ]);
        let tensor2 = tensor::<i64>(vec![3,3],vec![
            1,2,3,
            4,5,6,
            7,8,9
        ]);
        println!("{}",&tensor1);
        println!("{}",&tensor2);
        println!("{}",tensor1.clone().inner(1, &tensor2));
        println!("{}",tensor1.clone() + tensor2.clone());
        println!("{}",tensor1.clone() * tensor2.clone());
        println!("{}",tensor1.clone() + 2);
        println!("{}",tensor1.clone() * 2);
        println!("{}",tensor2.gross_product());
        println!("{}",tensor2.gross_sum());
    }
}