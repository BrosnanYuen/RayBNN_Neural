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








    let Softplus_out = RayBNN_Neural::Network::Activation::Softplus(&inx);

    let mut Softplus_act_cpu:Vec<f64> = vec![1.163386387166239 ,  0.333452485089616 ,  0.597149103121888  , 1.216537872592923  , 0.976506362205062 ,  0.415989696012672  ,  0.919795750930532   ];
    let mut Softplus_out_cpu = vec!(f64::default();Softplus_out.elements());

    Softplus_out.host(&mut Softplus_out_cpu);

    Softplus_out_cpu = Softplus_out_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    Softplus_act_cpu = Softplus_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    assert_eq!(Softplus_act_cpu, Softplus_out_cpu);











	let vA:Vec<f64> = vec![-0.2982  , -0.0831  ,  0.0214  , -0.0983  , -0.1469  ,  0.1847  , -0.2015 ];
	let A = arrayfire::Array::new(&vA, arrayfire::Dim4::new(&[1, vA.len() as u64, 1, 1]));

	let vB:Vec<f64> = vec![5.2428 ,  13.0324  ,  4.1387  ,  2.2668 , -15.8075 , -17.9467  ,  8.9090];
	let B = arrayfire::Array::new(&vB, arrayfire::Dim4::new(&[1, vA.len() as u64, 1, 1]));

	let vC:Vec<f64> = vec![ 1.4957  ,  1.7917  , -1.7917  ,  1.9376 ,  -1.8764  ,   0.7045  ,  0.2624];
	let C = arrayfire::Array::new(&vC, arrayfire::Dim4::new(&[1, vA.len() as u64, 1, 1]));

	let vD:Vec<f64> = vec![ -0.0685  , -0.1045  , -0.1405  , -0.0495  ,  0.3050  , -0.2759 ,  -0.2355];
	let D = arrayfire::Array::new(&vD, arrayfire::Dim4::new(&[1, vA.len() as u64, 1, 1]));

	let vE:Vec<f64> = vec![ -0.4353  ,  1.1616  , -1.3310  , -1.2005  , -1.6402 ,  -1.4096   , 1.2616];
	let E = arrayfire::Array::new(&vE, arrayfire::Dim4::new(&[1, vA.len() as u64, 1, 1]));

    let UAF_out = RayBNN_Neural::Network::Activation::UAF(&inx,&A,&B,&C,&D,&E);


    let mut UAF_act_cpu:Vec<f64> = vec![  -1.229361239592553 , -0.430689013312100 , -1.676232710717406 , -1.769939655881784 , -4.693361583872426 , -1.394708635225978  , -0.729895666552097      ];

    let mut UAF_out_cpu = vec!(f64::default();UAF_out.elements());

    UAF_out.host(&mut UAF_out_cpu);

    UAF_out_cpu = UAF_out_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    UAF_act_cpu = UAF_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();


    assert_eq!(UAF_act_cpu, UAF_out_cpu);






}
