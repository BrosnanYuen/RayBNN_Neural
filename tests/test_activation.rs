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







	let mut dX = arrayfire::constant::<f64>(0.0, A.dims());
	let mut dA = arrayfire::constant::<f64>(0.0, A.dims());
	let mut dB = arrayfire::constant::<f64>(0.0, A.dims());
	let mut dC = arrayfire::constant::<f64>(0.0, A.dims());
	let mut dD = arrayfire::constant::<f64>(0.0, A.dims());
	let mut dE = arrayfire::constant::<f64>(0.0, A.dims());

	RayBNN_Neural::Network::Activation::deriUAF(&inx,
		&A,
		&B,
		&C,
		&D,
		&E,
		&mut dX,
		&mut dA,
		&mut dB,
		&mut dC,
		&mut dD,
		&mut dE);



        let mut dX_act_cpu:Vec<f64> = vec![   -0.123406955300266 ,  0.320353422125326 ,  0.466063194985745 , -0.481715567525623 , -2.044655810183525 ,  0.028104792205827 ,  0.154207512685807    ];

        let mut dX_out_cpu = vec!(f64::default();dX.elements());

        dX.host(&mut dX_out_cpu);

        dX_act_cpu = dX_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        dX_out_cpu = dX_out_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        assert_eq!(dX_out_cpu, dX_act_cpu);












        let mut dA_act_cpu:Vec<f64> = vec![   0.369549809407306 ,  0.880593185595934  , 1.979063193134022 ,  0.460411899165353, -13.079421227583913 , -0.429394306561859 ,   1.189184803019221 ];

        let mut dA_out_cpu = vec!(f64::default();dA.elements());

        dA.host(&mut dA_out_cpu);

        dA_act_cpu = dA_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        dA_out_cpu = dA_out_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        assert_eq!(dA_out_cpu, dA_act_cpu);










        let mut dB_act_cpu:Vec<f64> = vec![ -0.057705065997712 , -0.090830090110622 , -0.080273580093522 , -0.040058571295469 ,  0.177357311082932 ,  0.001939312658922, -0.233167127800427  ];

        let mut dB_out_cpu = vec!(f64::default();dB.elements());

        dB.host(&mut dB_out_cpu);

        dB_act_cpu = dB_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        dB_out_cpu = dB_out_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        assert_eq!(dB_out_cpu, dB_act_cpu);









        let mut dC_act_cpu:Vec<f64> = vec![  -0.038121875085249 , -0.062496882277832  , 0.020554726788054 , -0.110041830838628  , 0.217011944096842 , -0.010109445684933  ,   -0.021584245797868   ];

        let mut dC_out_cpu = vec!(f64::default();dB.elements());

        dC.host(&mut dC_out_cpu);

        dC_act_cpu = dC_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        dC_out_cpu = dC_out_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        assert_eq!(dC_out_cpu, dC_act_cpu);






        let mut dD_act_cpu:Vec<f64> = vec![ 2.564116730779280 , 11.325751499026124 ,  2.812543689109861  , 0.725100724887516, -16.199482990886480 , -0.145511101613359  ,  7.485826928251551 ];

        let mut dD_out_cpu = vec!(f64::default();dB.elements());

        dD.host(&mut dD_out_cpu);

        dD_act_cpu = dD_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        dD_out_cpu = dD_out_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        assert_eq!(dD_out_cpu, dD_act_cpu);







        let mut dE_act_cpu:Vec<f64> = vec![1.0 ,  1.0,  1.0 ,  1.0  ,  1.0 ,  1.0,  1.0];

        let mut dE_out_cpu = vec!(f64::default();dE.elements());

        dE.host(&mut dE_out_cpu);

        dE_out_cpu = dE_out_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        dE_act_cpu = dE_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        assert_eq!(dE_act_cpu, dE_out_cpu);

}
