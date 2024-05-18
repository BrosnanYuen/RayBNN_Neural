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
fn test_init() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

    let A_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	
	let mut H = arrayfire::constant::<f64>(0.0,A_dims);
	let mut A = arrayfire::constant::<f64>(0.0,A_dims);
	let mut B = arrayfire::constant::<f64>(0.0,A_dims);
	let mut C = arrayfire::constant::<f64>(0.0,A_dims);
	let mut D = arrayfire::constant::<f64>(0.0,A_dims);
	let mut E = arrayfire::constant::<f64>(0.0,A_dims);

	RayBNN_Neural::Network::Initialization::UAF_initial_as_identity(




        &mut A,
        &mut B,
        &mut C,
        &mut D,
        &mut E
    );



}
