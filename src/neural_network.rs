use crate::tensor::*;
use crate::general_type::*;

trait Layer<F:Float>{
    fn forwardpass(&self,input:Tensor<F>)->Tensor<F>;
    fn backwardpass(&self,input:Tensor<F>)->Tensor<F>;
    fn learn(&self,input:Tensor<F>)->Tensor<F>;
}

pub struct MultipleLinerRegression<F:Float>{
    input: Option<Tensor<F>>,
    weight: Tensor<F>,
    bias: Tensor<F>
}
pub fn multiple_liner_regression<F:Float>(input:usize,output:usize)->MultipleLinerRegression<F>{
    MultipleLinerRegression::<F>{
        input:None,
        weight:zero_tensor::<F>(vec![input,output]),
        bias:zero_tensor::<F>(vec![output])
    }
}
// impl<F:Float> Layer<F> for MultipleLinerRegression<F>{
//     fn forwardpass(&self,input:Tensor<F>)->Tensor<F> {
//         input.inner(1,&self.weight) + self.bias
//     }
//     fn backwardpass(&self,input:Tensor<F>)->Tensor<F> {

//     }
// }