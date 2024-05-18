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
fn test_activation2() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);





    let inx_cpu:Vec<f64> = vec![   0.7888 ,  -0.9269  , -0.2022  ,  0.8652  ,  0.5039  , -0.6619  ,  0.4113,         
        
    0.9783 ,   0.7707  , -0.7861   , 0.3533  ,  0.3604  ,  0.4671 ,  -0.3071,       
          
     0.3677,   -0.5644,   -0.1787,    0.3248,    0.5010,    0.9923,   -0.5645,
    ];
    let inx = arrayfire::Array::new(&inx_cpu, arrayfire::Dim4::new(&[7, 3, 1, 1]));









	let vA:Vec<f64> = vec![-0.2982  , -0.0831  ,  0.0214  , -0.0983  , -0.1469  ,  0.1847  , -0.2015 ];
	let A = arrayfire::Array::new(&vA, arrayfire::Dim4::new(&[vA.len() as u64, 1, 1, 1]));

	let vB:Vec<f64> = vec![5.2428 ,  13.0324  ,  4.1387  ,  2.2668 , -15.8075 , -17.9467  ,  8.9090];
	let B = arrayfire::Array::new(&vB, arrayfire::Dim4::new(&[vA.len() as u64, 1, 1, 1]));

	let vC:Vec<f64> = vec![ 1.4957  ,  1.7917  , -1.7917  ,  1.9376 ,  -1.8764  ,   0.7045  ,  0.2624];
	let C = arrayfire::Array::new(&vC, arrayfire::Dim4::new(&[vA.len() as u64, 1, 1, 1]));

	let vD:Vec<f64> = vec![ -0.0685  , -0.1045  , -0.1405  , -0.0495  ,  0.3050  , -0.2759 ,  -0.2355];
	let D = arrayfire::Array::new(&vD, arrayfire::Dim4::new(&[vA.len() as u64, 1, 1, 1]));

	let vE:Vec<f64> = vec![ -0.4353  ,  1.1616  , -1.3310  , -1.2005  , -1.6402 ,  -1.4096   , 1.2616];
	let E = arrayfire::Array::new(&vE, arrayfire::Dim4::new(&[vA.len() as u64, 1, 1, 1]));

    let UAF_out = RayBNN_Neural::Network::Activation::UAF(&inx,&A,&B,&C,&D,&E);


    let mut UAF_act_cpu:Vec<f64> = vec![  -1.22936123959255283,-0.43068901331209997,-1.67623271071740554,-1.76993965588178370,-4.69336158387242630,-1.39470863522597788,-0.72989566655209714,

    -1.24843704959882174,-0.26081523602836415,-2.12495102798760449,-1.46782709755245144,-4.42921721614291553,-1.38238979106460258,-0.85782859692173963,

    -1.16665637163970914,-0.29283352663867290,-1.66574472382127725,-1.45317453386757922,-4.68744341700215550,-1.39338430513618938,-0.91342947151044573,

    ];



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



        let mut dX_act_cpu:Vec<f64> = vec![  -0.12340695530026649,0.32035342212532558,0.46606319498574528,-0.48171556752562289,-2.04465581018352482,0.02810479220582731,0.15420751268580674,

        -0.07697776012028157,-0.19911699729428844,0.83735302829026659,-0.52828888791611350,-1.62745846805266337,-0.01385095221721493,0.20544454958903371,

        -0.14594182421574761,0.40810303711914669,0.42635724608494324,-0.49945747859865652,-2.03683357130841891,-0.02442985946742689,0.22650515658107212,


        ];


        let mut dX_out_cpu = vec!(f64::default();dX.elements());

        dX.host(&mut dX_out_cpu);

        dX_act_cpu = dX_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        dX_out_cpu = dX_out_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        assert_eq!(dX_out_cpu, dX_act_cpu);












        let mut dA_act_cpu:Vec<f64> = vec![ 0.36954980940730592,0.88059318559593447,1.97906319313402168,0.46041189916535263,-13.07942122758391257,-0.42939430656185873,1.18918480301922136,


        0.22416466427523019,1.36296081745925957,0.87848375269466017,0.98955836248291940,-13.64664906149745249,-0.57432162082617455,1.26479974182664234,

        0.74581823549910597,2.08251631034869700,2.00725140261635371,1.00342803812288284,-13.09308294967461528,-0.36195950075708883,1.21964119134134408,

        ];

        let mut dA_out_cpu = vec!(f64::default();dA.elements());

        dA.host(&mut dA_out_cpu);

        dA_act_cpu = dA_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        dA_out_cpu = dA_out_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        assert_eq!(dA_out_cpu, dA_act_cpu);










        let mut dB_act_cpu:Vec<f64> = vec![ -0.05770506599771181,-0.09083009011062197,-0.08027358009352248,-0.04005857129546887,0.17735731108293212,0.00193931265892245,-0.23316712780042728,


        -0.04996228206711301,-0.08999551600673308,-0.08802133870713000,-0.06304716908008694,0.17303642071803771,0.00436378634089792,-0.24100314473439985,

        -0.07955673204306543,-0.09805214277950650,-0.08007922041388682,-0.06399894359681177,0.17724814426256338,0.00246705985343255,-0.24210791109388477,

        ];

        let mut dB_out_cpu = vec!(f64::default();dB.elements());

        dB.host(&mut dB_out_cpu);

        dB_act_cpu = dB_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        dB_out_cpu = dB_out_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        assert_eq!(dB_out_cpu, dB_act_cpu);









        let mut dC_act_cpu:Vec<f64> = vec![ -0.03812187508524918,-0.06249688227783166,0.02055472678805375,-0.11004183083862808,0.21701194409684171,-0.01010944568493328,-0.02158424579786774,

        -0.03448609968405037,-0.05865127458930360,0.16192264359318481,-0.04714230583262494,0.11474892612617456,-0.00716874959077788,-0.01386714356358070,

        -0.01797292205355564,-0.05320661475926517,0.01618660203111511,-0.04084607231783263,0.21470466229714683,-0.02102149206248700,-0.04657572855690940,

        ];

        let mut dC_out_cpu = vec!(f64::default();dB.elements());

        dC.host(&mut dC_out_cpu);

        dC_act_cpu = dC_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        dC_out_cpu = dC_out_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        assert_eq!(dC_out_cpu, dC_act_cpu);






        let mut dD_act_cpu:Vec<f64> = vec![ 2.56411673077928004,11.32575149902612388,2.81254368910986052,0.72510072488751620,-16.19948299088648014,-0.14551110161335884,7.48582692825155149,

        2.44148866664802355,9.59697618494431914,3.28187246478309413,1.00202699751201685,-16.05204009689910905,-0.11378262934087251,8.27199325986125267,

        2.84081236003903781,10.95186996163980098,2.79406419715320142,1.01763472324347770,-16.19650455178510384,-0.10132609454422944,8.55456857080037203,

        ];

        let mut dD_out_cpu = vec!(f64::default();dB.elements());

        dD.host(&mut dD_out_cpu);

        dD_act_cpu = dD_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        dD_out_cpu = dD_out_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        assert_eq!(dD_out_cpu, dD_act_cpu);







        let mut dE_act_cpu:Vec<f64> = vec![1.0 ,  1.0,  1.0 ,  1.0  ,  1.0 ,  1.0,  1.0,
        
        1.0 ,  1.0,  1.0 ,  1.0  ,  1.0 ,  1.0,  1.0,

        1.0 ,  1.0,  1.0 ,  1.0  ,  1.0 ,  1.0,  1.0,
        
        ];

        let mut dE_out_cpu = vec!(f64::default();dE.elements());

        dE.host(&mut dE_out_cpu);

        dE_out_cpu = dE_out_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        dE_act_cpu = dE_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

        assert_eq!(dE_act_cpu, dE_out_cpu);



}
