#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;
use RayBNN_Graph;
use RayBNN_Neural;



use std::collections::HashMap;
use nohash_hasher;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;




#[test]
fn test_init() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


    let neuron_size = 53;

    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	
	let mut H = arrayfire::constant::<f64>(0.0,single_dims);
	let mut A = arrayfire::constant::<f64>(0.0,single_dims);
	let mut B = arrayfire::constant::<f64>(0.0,single_dims);
	let mut C = arrayfire::constant::<f64>(0.0,single_dims);
	let mut D = arrayfire::constant::<f64>(0.0,single_dims);
	let mut E = arrayfire::constant::<f64>(0.0,single_dims);

    let mut modeldata_float:  HashMap<String, f64> = HashMap::new();
    let mut modeldata_int:  HashMap<String, u64> = HashMap::new();

    modeldata_float.insert("neuron_std".to_string(), 0.1);
    modeldata_int.insert("neuron_size".to_string(), neuron_size);


	RayBNN_Neural::Network::Initialization::UAF_initial_as_identity(
        &modeldata_float,
        &modeldata_int,


        &mut A,
        &mut B,
        &mut C,
        &mut D,
        &mut E
    );

    //RayBNN_Neural::Network::Neurons::print_dims(&A);

    assert_eq!(A.dims()[0], neuron_size);
    assert_eq!(B.dims()[0], neuron_size);
    assert_eq!(C.dims()[0], neuron_size);
    assert_eq!(D.dims()[0], neuron_size);
    assert_eq!(E.dims()[0], neuron_size);

    for i in 1..4
    {
        assert_eq!(A.dims()[i], 1);
        assert_eq!(B.dims()[i], 1);
        assert_eq!(C.dims()[i], 1);
        assert_eq!(D.dims()[i], 1);
        assert_eq!(E.dims()[i], 1);
    }

    let mut inx_cpu:Vec<f64> = vec![   0.7888 ,  -0.9269  , -0.2022  ,  0.8652  ,  0.5039  , -0.6619  ,  0.4113 ];
    let inx = arrayfire::Array::new(&inx_cpu, arrayfire::Dim4::new(&[1, inx_cpu.len() as u64, 1, 1]));

    
    let UAF_out = RayBNN_Neural::Network::Activation::UAF(&inx,&A,&B,&C,&D,&E);


    let mut UAF_out_cpu = vec!(f64::default();UAF_out.elements());

    UAF_out.host(&mut UAF_out_cpu);

    UAF_out_cpu = UAF_out_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    inx_cpu = inx_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();


    assert_eq!(inx_cpu, UAF_out_cpu);




}
