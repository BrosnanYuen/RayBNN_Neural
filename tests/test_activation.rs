#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;
use RayBNN_Graph;
use RayBNN_Neural;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;




#[test]
fn test_neural() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);



    let inx_cpu:Vec<f64> = vec![   0.7888 ,  -0.9269  , -0.2022  ,  0.8652  ,  0.5039  , -0.6619  ,  0.4113 ];
    let inx = arrayfire::Array::new(&inx_cpu, arrayfire::Dim4::new(&[1, inx_cpu.len() as u64, 1, 1]));




    let ReLU_out = RayBNN_Neural::Network::Activation::ReLU(&inx);

    let mut ReLU_act_cpu:Vec<f64> = vec![ 0.7888 ,  0.0  , 0.0 ,  0.8652  ,  0.5039  , 0.0  ,  0.4113];
    let mut ReLU_out_cpu = vec!(f64::default();ReLU_out.elements());


    ReLU_out.host(&mut ReLU_out_cpu);

    assert_eq!(ReLU_act_cpu, ReLU_out_cpu);







}
