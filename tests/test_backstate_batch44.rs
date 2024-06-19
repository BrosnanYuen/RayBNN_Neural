#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;
use RayBNN_Graph;
use RayBNN_Neural;
use RayBNN_Optimizer;

use std::collections::HashMap;
use nohash_hasher;



const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;




#[test]
fn test_backstate_batch44() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

	let rand_seed = 4324;

	arrayfire::set_seed(rand_seed);

	let neuron_size: u64 = 20;
	let input_size: u64 = 6;
	let output_size: u64 = 5;
	let proc_num: u64 = 2;
	//let active_size: u64 = 25;
	//let space_dims: u64 = 3;
	//let sim_steps: u64 = 100;
	let mut batch_size: u64 = 2;


	//let dataset_size: u64 = 70000;
	//let train_size: u64 = 60000;
	//let test_size: u64 = 10000;

	//let mut netdata = clusterdiffeq::neural::network_f64::create_nullnetdata();

	
	let mut modeldata_string:  HashMap<String, String> = HashMap::new();
    let mut modeldata_int: HashMap<String,u64> = HashMap::new();

    modeldata_int.insert("neuron_size".to_string(), neuron_size);
    modeldata_int.insert("input_size".to_string(), input_size);
    modeldata_int.insert("output_size".to_string(), output_size);
    modeldata_int.insert("proc_num".to_string(), proc_num);
    modeldata_int.insert("active_size".to_string(), neuron_size);
    //modeldata_int.insert("space_dims".to_string(), space_dims);
    //modeldata_int.insert("step_num".to_string(), sim_steps);
    modeldata_int.insert("batch_size".to_string(), batch_size);

	let mut modeldata_float: HashMap<String,f64> = HashMap::new();
    modeldata_float.insert("time_step".to_string(), 0.1);
	modeldata_float.insert("nratio".to_string(), 0.5);
	modeldata_float.insert("neuron_std".to_string(), 0.06);
	modeldata_float.insert("sphere_rad".to_string(), 0.9);
	modeldata_float.insert("neuron_rad".to_string(), 0.1);
	modeldata_float.insert("con_rad".to_string(), 0.6);
	modeldata_float.insert("init_prob".to_string(), 0.5);
	modeldata_float.insert("add_neuron_rate".to_string(), 0.0);
	modeldata_float.insert("del_neuron_rate".to_string(), 0.0);


	let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);


	//let mut glia_pos = arrayfire::constant::<f64>(0.0,temp_dims);
	//let mut neuron_pos = arrayfire::constant::<f64>(0.0,temp_dims);


	//netdata.input_size = input_size;
	//netdata.output_size = output_size;
	//netdata.neuron_size = neuron_size;
	//netdata.proc_num = proc_num;
	//netdata.active_size = neuron_size;
	//netdata.batch_size = batch_size;

	
	//let mut H = arrayfire::constant::<f64>(0.0,temp_dims);
	//let mut A = arrayfire::constant::<f64>(0.0,temp_dims);
	//let mut B = arrayfire::constant::<f64>(0.0,temp_dims);
	//let mut C = arrayfire::constant::<f64>(0.0,temp_dims);
	//let mut D = arrayfire::constant::<f64>(0.0,temp_dims);
	//let mut E = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut neuron_idx = arrayfire::constant::<i32>(0,temp_dims);


	let N_dims = arrayfire::Dim4::new(&[neuron_size ,1,1,1]);
	let repeat_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    //let L_dims = arrayfire::Dim4::new(&[output_size,1,1,1]);

	neuron_idx = arrayfire::iota::<i32>(N_dims,repeat_dims);
    //let last_neurons = arrayfire::iota::<i32>(L_dims,repeat_dims) + ((neuron_size-output_size) as i32 );

    //neuron_idx = arrayfire::join(0, &neuron_idx, &last_neurons);




	//let mut WValues = arrayfire::constant::<f64>(0.0,temp_dims);
	//let mut WRowIdxCSR = arrayfire::constant::<i32>(0,temp_dims);
	//let mut WColIdx = arrayfire::constant::<i32>(0,temp_dims);



	let tile_dims = arrayfire::Dim4::new(&[1,batch_size,1,1]);
	



	let X_cpu: Vec<f64> = vec![0.263700,0.295200,-0.728400,0.834900,-0.619800,-0.310400,];
	let mut train_X = arrayfire::Array::new(&X_cpu, arrayfire::Dim4::new(&[X_cpu.len() as u64, 1, 1, 1]));

	train_X =  arrayfire::tile(&train_X, tile_dims);


	let Y_cpu: Vec<f64> = vec![-0.042500,-0.260800,0.417500,-0.972800,0.230200];
	let mut Y = arrayfire::Array::new(&Y_cpu, arrayfire::Dim4::new(&[Y_cpu.len() as u64, 1, 1, 1]));


	Y =  arrayfire::tile(&Y, tile_dims);



	let A_cpu: Vec<f64> = vec![-0.172100,-0.183100,-0.208500,-0.101300,-0.068700,-0.147900,-0.035200,-0.091000,-0.172000,-0.191700,-0.124300,-0.200900,-0.238200,-0.148800,-0.219900,-0.206900,-0.181000,-0.096800,-0.148600,-0.005400,	];
	let mut A = arrayfire::Array::new(&A_cpu, arrayfire::Dim4::new(&[A_cpu.len() as u64, 1, 1, 1]));


	let B_cpu: Vec<f64> = vec![2.433300,-10.346700,2.327200,6.903200,8.113400,-2.301700,-0.113800,-6.599900,2.394600,4.002600,-2.593300,-10.030200,-9.371200,-2.105800,10.471700,11.880500,2.829900,10.013500,0.572900,-10.936300,	];
	let mut B = arrayfire::Array::new(&B_cpu, arrayfire::Dim4::new(&[B_cpu.len() as u64, 1, 1, 1]));


	let C_cpu: Vec<f64> = vec![-1.871100,-1.439800,0.597400,-1.363600,-1.955700,0.398700,1.063500,-0.672400,1.574200,-0.089300,-1.493700,1.529400,0.764700,-0.250000,0.525600,1.107200,-0.969500,-1.222000,0.729700,1.815500,	];
	let mut C = arrayfire::Array::new(&C_cpu, arrayfire::Dim4::new(&[C_cpu.len() as u64, 1, 1, 1]));


	let D_cpu: Vec<f64> = vec![-0.171100,-0.206600,-0.172900,-0.073200,-0.107800,-0.176100,-0.112000,-0.174700,-0.230300,-0.016000,-0.015600,-0.030900,-0.113000,-0.019500,-0.029900,-0.051300,-0.103500,-0.196000,-0.013400,-0.122100,	];
	let mut D = arrayfire::Array::new(&D_cpu, arrayfire::Dim4::new(&[D_cpu.len() as u64, 1, 1, 1]));


	let E_cpu: Vec<f64> = vec![-0.721100,-0.434500,0.895100,0.375900,-0.322200,0.762100,-0.131100,0.762900,0.395100,-0.579600,-0.572600,0.091100,-0.678000,-0.283700,0.612900,-0.067200,0.282800,0.052900,-0.480400,0.662100,	];
	let mut E = arrayfire::Array::new(&E_cpu, arrayfire::Dim4::new(&[E_cpu.len() as u64, 1, 1, 1]));


	let H_cpu: Vec<f64> = vec![-0.635200,0.663500,-0.600000,0.084600,0.279600,-0.084200,-0.979000,0.279800,0.888600,0.205700,0.332300,-0.955100,-0.222700,0.006500,0.594000,0.703000,-0.816900,-0.361300,-0.328500,0.024600,	];
	let mut H = arrayfire::Array::new(&H_cpu, arrayfire::Dim4::new(&[H_cpu.len() as u64, 1, 1, 1]));







	




	let W_cpu: Vec<f64> = vec![0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.893800,0.000000,-0.304300,0.000000,0.404600,-0.000000,0.000000,-0.000000,0.000000,0.000000,0.177700,0.437700,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000000,-0.287300,0.640400,-0.988600,-0.000000,-0.000000,-0.852600,-0.000000,0.747900,-0.281300,-0.944700,-0.743700,0.000000,-0.449800,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.581500,0.008000,0.412600,-0.000000,0.853800,-0.000000,-0.611500,-0.000000,-0.000000,0.000000,0.000000,-0.354800,-0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000000,0.110500,-0.570200,-0.122800,-0.929400,0.158800,-0.000000,0.000000,0.000000,-0.155100,-0.000000,-0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.183800,-0.856500,-0.000000,0.501300,0.988100,-0.000000,0.000000,0.000000,-0.000000,-0.000000,-0.000000,0.000000,0.790200,-0.613700,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.513600,0.000000,0.000000,0.000000,-0.000000,-0.103200,0.000000,-0.000000,0.722500,-0.000000,0.622200,0.566800,0.141000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.089800,0.000000,-0.000000,-0.564100,0.906700,-0.469800,-0.000000,0.000000,-0.505700,-0.000000,0.023600,-0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.112600,0.000000,0.232800,-0.791000,-0.806400,0.276200,0.510700,0.000000,-0.000000,-0.000000,0.627900,0.926200,-0.826400,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.206600,0.431500,0.000000,-0.000000,0.000000,0.779000,0.823900,-0.000000,0.053600,0.000000,0.005400,-0.000000,0.000000,-0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.616600,-0.000000,-0.000000,0.647700,-0.299900,0.828400,0.142100,0.000000,0.699800,-0.000000,0.675000,0.000000,0.830000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000000,0.896600,0.000000,-0.630800,-0.000000,-0.323300,-0.097900,-0.604200,-0.676500,-0.000000,-0.000000,-0.000000,0.000000,-0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.603700,-0.656800,0.000000,-0.411300,0.000000,-0.000000,0.728700,0.000000,0.942000,-0.000000,0.077700,-0.209100,-0.157000,-0.057900,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.898700,0.218300,0.534700,-0.000000,-0.173900,-0.000000,-0.000000,0.000000,0.000000,-0.850800,0.000000,-0.000000,0.000000,-0.395200,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.517900,0.219100,-0.027500,-0.000000,-0.000000,0.000000,-0.053000,0.000000,0.748700,-0.000000,0.000000,0.714400,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.470500,-0.604000,0.000000,-0.532800,0.000000,-0.654700,0.161200,-0.300300,0.000000,-0.000000,0.209400,0.332400,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,];
	let mut W = arrayfire::Array::new(&W_cpu, arrayfire::Dim4::new(&[neuron_size, neuron_size, 1, 1]));

	let denseW = W.clone();
	
	W = arrayfire::sparse_from_dense(&W, arrayfire::SparseFormat::CSR);

	//arrayfire::print_gen("W".to_string(), &W, Some(6));


	let mut WValues = arrayfire::sparse_get_values(&W);
	let mut WRowIdxCSR = arrayfire::sparse_get_row_indices(&W);
	let mut WColIdx =  arrayfire::sparse_get_col_indices(&W);



	


    let Z_dims = arrayfire::Dim4::new(&[neuron_size,batch_size,proc_num,1]);
	let mut Z = arrayfire::constant::<f64>(0.0,Z_dims);
	let mut Q = arrayfire::constant::<f64>(0.0,Z_dims);



	//let mut alpha0: f64 =  0.0091;
	//let mut alpha1: f64 = alpha0;

	


	let active_size = neuron_idx.dims()[0];
	let idxsel = arrayfire::rows(&neuron_idx, (active_size-output_size)  as i64, (active_size-1)  as i64);
    let Qslices: u64 = Q.dims()[2];




	let mut WRowIdxCOO = RayBNN_Sparse::Util::Convert::CSR_to_COO(&WRowIdxCSR);






	let total_param_size = WValues.dims()[0]  +  H.dims()[0]  +  A.dims()[0]    +  B.dims()[0]    +  C.dims()[0]    +  D.dims()[0]   +  E.dims()[0]  ;
	let mt_dims = arrayfire::Dim4::new(&[total_param_size,1,1,1]);
	let mut mt = arrayfire::constant::<f64>(0.0,mt_dims);
	let mut vt = arrayfire::constant::<f64>(0.0,mt_dims);
	let mut grad = arrayfire::constant::<f64>(0.0,mt_dims);






	let X_dims = arrayfire::Dim4::new(&[input_size,batch_size,proc_num,1]);
    let mut X = arrayfire::constant::<f64>(0.0,X_dims);










	let mut idxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
	let mut valsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();

	let mut cvec_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
	let mut dXsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();

	let mut nrows_out: nohash_hasher::IntMap<i64, u64> = nohash_hasher::IntMap::default();
	let mut sparseval_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
	let mut sparsecol_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
	let mut sparserow_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();

	arrayfire::sync(DEVICE);




	let mut Wseqs = [arrayfire::Seq::default()];
	let mut Hseqs = [arrayfire::Seq::default()];
	let mut Aseqs = [arrayfire::Seq::default()];
	let mut Bseqs = [arrayfire::Seq::default()];
	let mut Cseqs = [arrayfire::Seq::default()];
	let mut Dseqs = [arrayfire::Seq::default()];
	let mut Eseqs = [arrayfire::Seq::default()];



	let mut Hidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
	let mut Aidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
	let mut Bidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
	let mut Cidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
	let mut Didxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
	let mut Eidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
	let mut combidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();




	let mut dAseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] > = nohash_hasher::IntMap::default();
	let mut dBseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] > = nohash_hasher::IntMap::default();
	let mut dCseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] > = nohash_hasher::IntMap::default();
	let mut dDseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] > = nohash_hasher::IntMap::default();
	let mut dEseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] > = nohash_hasher::IntMap::default();






	RayBNN_Neural::Network::Neurons::find_path_backward_group2(
		//&netdata,
		&modeldata_int,


		proc_num,
		1, 
		&WRowIdxCOO,
		&WColIdx,
		&neuron_idx,

	
	

		WValues.dims()[0],
		H.dims()[0],
		A.dims()[0],
		B.dims()[0],
		C.dims()[0],
		D.dims()[0],
		E.dims()[0],
	
	
	
		&mut idxsel_out,
		&mut valsel_out,

		&mut cvec_out,
		&mut dXsel_out,

		&mut nrows_out,
		&mut sparseval_out,
		&mut sparserow_out,
		&mut sparsecol_out,
	
	
	
		&mut Hidxsel_out,
		&mut Aidxsel_out,
		&mut Bidxsel_out,
		&mut Cidxsel_out,
		&mut Didxsel_out,
		&mut Eidxsel_out,
		&mut combidxsel_out,
	
	
	
	

		&mut dAseqs_out,
		&mut dBseqs_out,
		&mut dCseqs_out,
		&mut dDseqs_out,
		&mut dEseqs_out,
	
	
	
		
		&mut Wseqs,
		&mut Hseqs,
		&mut Aseqs,
		&mut Bseqs,
		&mut Cseqs,
		&mut Dseqs,
		&mut Eseqs
	
	);












	//let train_X_dims = arrayfire::Dim4::new(&[input_size,batch_size,1,1]);
	//let mut train_X =  arrayfire::constant::<f64>(0.0,temp_dims);
	//let Y_dims = arrayfire::Dim4::new(&[output_size,batch_size,1,1]);
	//let mut Y =  arrayfire::constant::<f64>(0.0,temp_dims);
	//let mut batch_idx = 0;
	//let epoch_num = (train_size/batch_size);








	let mut network_params = arrayfire::constant::<f64>(0.0,mt_dims);
	arrayfire::assign_seq(&mut network_params, &Wseqs, &WValues);
	arrayfire::assign_seq(&mut network_params, &Hseqs, &H);
	arrayfire::assign_seq(&mut network_params, &Aseqs, &A);
	arrayfire::assign_seq(&mut network_params, &Bseqs, &B);
	arrayfire::assign_seq(&mut network_params, &Cseqs, &C);
	arrayfire::assign_seq(&mut network_params, &Dseqs, &D);	
	arrayfire::assign_seq(&mut network_params, &Eseqs, &E);	






	arrayfire::device_gc();
	arrayfire::sync(DEVICE);

	//batch_idx = i % epoch_num;

	//train_X =  arrayfire::Array::new(&MNISTXX[&batch_idx], train_X_dims);

	//Y =  arrayfire::Array::new(&MNISTY[&batch_idx], Y_dims);
	arrayfire::set_slice(&mut X, &train_X, 0);




	//println!("a1");


	RayBNN_Neural::Network::Neurons::state_space_forward_batch(
		//&netdata,
		&modeldata_int,



		&X,
		
		&WRowIdxCSR,
		&WColIdx,
	
	
		&Wseqs,
		&Hseqs,
		&Aseqs,
		&Bseqs,
		&Cseqs,
		&Dseqs,
		&Eseqs,
		&network_params,
	
	
	
		&mut Z,
		&mut Q
	);


	//println!("a2");




	RayBNN_Neural::Network::Neurons::state_space_backward_group2(
		//&netdata,
		&modeldata_int,



		&X,
	
	

		&network_params,
	
	
	
	
		&Z,
		&Q,
		&Y,
		RayBNN_Optimizer::Continuous::Loss::MSE_grad,
		&neuron_idx,


	


		&idxsel_out,
		&valsel_out,

		&cvec_out,
		&dXsel_out,

		&nrows_out,
		&sparseval_out,
		&sparserow_out,
		&sparsecol_out,



	
	
		&Hidxsel_out,
		&Aidxsel_out,
		&Bidxsel_out,
		&Cidxsel_out,
		&Didxsel_out,
		&Eidxsel_out,
		&combidxsel_out,



		&dAseqs_out,
		&dBseqs_out,
		&dCseqs_out,
		&dDseqs_out,
		&dEseqs_out,
	


		
		&mut grad,
	);
	
	let gW = arrayfire::index(&grad, &Wseqs);

	let mut densegW = arrayfire::sparse(
		neuron_size, 
		neuron_size, 
		&gW, 
		&WRowIdxCOO, 
		&WColIdx, 
		arrayfire::SparseFormat::COO
	);
	
	densegW = arrayfire::sparse_to_dense(&densegW);
	
	//arrayfire::print_gen("densegW".to_string(), &densegW, Some(9));







	let gW_act_cpu: Vec<f64> = vec![0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.002210836591,0.042095616169,0.000001583667,-0.001442041826,0.000000000000,0.001258210933,0.002832970691,0.013912739964,0.001003553993,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.002474929699,0.047124102742,0.000001772842,-0.001614299382,0.000000000000,0.001408509168,0.003171380159,0.015574671359,0.001123432457,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,-0.006106838728,-0.116277765709,-0.000004374451,0.003983250914,0.000000000000,-0.003475467743,-0.007825316083,-0.038430185020,-0.002772046752,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.006999724951,0.133278839361,0.000005014044,-0.004565645508,0.000000000000,0.003983618916,0.008969462380,0.044049095927,0.003177350129,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,-0.005196346298,-0.098941459619,-0.000003722247,0.003389372483,0.000000000000,-0.002957296687,-0.006658609155,-0.032700478687,-0.002358751479,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,-0.002602365103,-0.049550547057,-0.000001864126,0.001697420488,0.000000000000,-0.001481034030,-0.003334676156,-0.016376619207,-0.001181278572,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.026261622081,0.006471526806,0.219652349093,0.028031665004,-0.001243964216,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,-0.040878516027,-0.010073498562,-0.341908129094,-0.043633742942,0.001936339309,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.017448600589,0.004299775775,0.145940189674,0.018624642643,-0.000826507772,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.040595514161,0.010003759756,0.339541099954,0.043331666647,-0.001922934037,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.021481270441,0.005293527455,0.179669462126,0.022929115917,-0.001017527846,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.021708742827,0.005349582394,0.181572042393,0.023171920025,-0.001028302790,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,-0.060391021741,-0.014881872675,-0.505110832398,-0.064461398670,0.002860610430,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.007060846077,0.001739970765,0.059056954766,0.007536749682,-0.000334459152,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.007577842919,0.001867371841,0.063381119153,0.008088592301,-0.000358948331,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.046934082726,0.011565743104,0.392556921711,0.050097457049,-0.002223180246,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.024751879335,0.006099488071,0.207024852598,0.026420165045,-0.001172450509,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.088517965555,0.021813061856,0.740364742548,0.094484108773,-0.004192931469,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.025956641866,0.006396371980,0.217101492926,0.027706129011,-0.001229517871,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,-0.049118537721,-0.012104047974,-0.410827715100,-0.052429145109,0.002326653818,];
	let mut gW_act = arrayfire::Array::new(&gW_act_cpu, arrayfire::Dim4::new(&[neuron_size, neuron_size, 1, 1]));

	
	//arrayfire::print_gen("gW_act".to_string(), &gW_act, Some(9));

	let zeros_dims = arrayfire::Dim4::new(&[neuron_size,neuron_size,1,1]);

	let mut zeros = arrayfire::constant::<f64>(0.0,zeros_dims);
	let mut ones = arrayfire::neq(&denseW,&zeros, true);


	gW_act = arrayfire::mul(&ones, &gW_act, false);


	//arrayfire::print_gen("gW_act".to_string(), &gW_act, Some(9));








	let mut densegW_cpu = vec!(f64::default();densegW.elements());
	densegW.host(&mut densegW_cpu);


	densegW_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&densegW_cpu, 9);


	let mut gW_cpu = vec!(f64::default();gW_act.elements());
	gW_act.host(&mut gW_cpu);


	gW_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gW_cpu, 9);


	assert_eq!(gW_cpu,densegW_cpu);














	let gH = arrayfire::index(&grad, &Hseqs);

	let mut gH_cpu = vec!(f64::default();gH.elements());
	gH.host(&mut gH_cpu);

	gH_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gH_cpu, 9);

	let mut gH_act_cpu: Vec<f64> = vec![0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.008383908193,0.159634494384,0.000006005562,-0.005468493841,0.000000000000,0.004771372519,0.010743157720,0.052759726826,0.003805665503,-0.045029402214,-0.011096381731,-0.376626163605,-0.048064400374,0.002132959071,];

	gH_act_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gH_act_cpu, 9);

	//println!("size  {}",gH_cpu.len());
	for qq in 0..gH_act_cpu.len()
	{
		if gH_act_cpu[qq] != 0.0
		{
			assert_eq!(gH_act_cpu[qq], gH_cpu[qq]);
		}
		
	}















	let gA = arrayfire::index(&grad, &Aseqs);

	let mut gA_cpu = vec!(f64::default();gA.elements());
	gA.host(&mut gA_cpu);

	gA_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gA_cpu, 9);

	let mut gA_act_cpu: Vec<f64> = vec![0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,-0.004017581107,0.452284378698,-0.000012870103,-0.264696663654,0.000000000000,-0.006729000557,-0.328411981905,0.244127959609,-0.071473739722,-0.228174545328,-0.012730189825,-2.034621093988,0.058446281706,-0.000664687881,];

	gA_act_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gA_act_cpu, 9);

	//println!("size  {}",gH_cpu.len());
	for qq in 0..gA_act_cpu.len()
	{
		if gA_act_cpu[qq] != 0.0
		{
			assert_eq!(gA_act_cpu[qq], gA_cpu[qq]);
		}
		
	}



	









	let gE = arrayfire::index(&grad, &Eseqs);

	let mut gE_cpu = vec!(f64::default();gE.elements());
	gE.host(&mut gE_cpu);

	gE_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gE_cpu, 9);

	let mut gE_act_cpu: Vec<f64> = vec![0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.013882991238,-0.282763493130,-0.000059920461,-0.283963880073,0.000000000000,0.085312954478,0.037468069979,-0.269061731279,-0.094842125343,-0.417501919990,-0.025225723322,-0.908414902605,0.129529749854,0.057588123224,];

	gE_act_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gE_act_cpu, 9);

	//println!("size  {}",gH_cpu.len());
	for qq in 0..gE_act_cpu.len()
	{
		if gE_act_cpu[qq] != 0.0
		{
			assert_eq!(gE_act_cpu[qq], gE_cpu[qq]);
		}
		
	}








}
