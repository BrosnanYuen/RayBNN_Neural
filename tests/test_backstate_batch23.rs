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



fn loss_grad(
	yhat: &arrayfire::Array<f32>,
	y: &arrayfire::Array<f32>) -> arrayfire::Array<f32> 
{ 
	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let TEN = arrayfire::constant::<f32>(10.0,single_dims);

	RayBNN_Optimizer::Continuous::Loss::weighted_sigmoid_cross_entropy_grad(yhat, y, &TEN)
	//clusterdiffeq::optimal::loss_f32::weighted_sigmoid_cross_entropy_grad(yhat, y, ten) 
}



#[test]
fn test_backstate_batch23() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

	let rand_seed = 4324;

	arrayfire::set_seed(rand_seed);

	let neuron_size: u64 = 15000;
	let input_size: u64 = 784;
	let output_size: u64 = 10;
	let proc_num: u64 = 4;
	let active_size: u64 = 25;
	let space_dims: u64 = 3;
	let sim_steps: u64 = 100;
	let mut batch_size: u64 = 1000;


	let dataset_size: u64 = 70000;
	let train_size: u64 = 60000;
	let test_size: u64 = 10000;

	//let mut netdata = clusterdiffeq::neural::network_f32::create_nullnetdata();


	let mut modeldata_string:  HashMap<String, String> = HashMap::new();
    let mut modeldata_int: HashMap<String,u64> = HashMap::new();

    modeldata_int.insert("neuron_size".to_string(), neuron_size);
    modeldata_int.insert("input_size".to_string(), input_size);
    modeldata_int.insert("output_size".to_string(), output_size);
    modeldata_int.insert("proc_num".to_string(), proc_num);
    modeldata_int.insert("active_size".to_string(), active_size);
    modeldata_int.insert("space_dims".to_string(), space_dims);
    modeldata_int.insert("step_num".to_string(), sim_steps);
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


	let mut glia_pos = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut neuron_pos = arrayfire::constant::<f32>(0.0,temp_dims);



	
	let mut H = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut A = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut B = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut C = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut D = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut E = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut neuron_idx = arrayfire::constant::<i32>(0,temp_dims);





	let mut WValues = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut WRowIdxCSR = arrayfire::constant::<i32>(0,temp_dims);
	let mut WColIdx = arrayfire::constant::<i32>(0,temp_dims);



	arrayfire::sync(DEVICE);
	/* 
	clusterdiffeq::export::dataloader_f32::load_network(
			"./test_data/network_MNIST.csv",
			&mut netdata,
			&mut WValues,
			&mut WRowIdxCSR,
			&mut WColIdx,
			&mut H,
			&mut A,
			&mut B,
			&mut C,
			&mut D,
			&mut E,
			&mut glia_pos,
			&mut neuron_pos,
			&mut neuron_idx
		);
	*/


	RayBNN_DataLoader::Model::Network::read_network_dir(
		"./test_data/network_batch23/", 
			&mut modeldata_string, 
			&mut modeldata_float, 
			&mut modeldata_int, 
			&mut WValues, 
			&mut WRowIdxCSR, 
			&mut WColIdx, 
			&mut H, 
			&mut A, 
			&mut B, 
			&mut C, 
			&mut D, 
			&mut E, 
			&mut glia_pos, 
			&mut neuron_pos, 
			&mut neuron_idx
	);
	
	arrayfire::sync(DEVICE);




	


	arrayfire::sync(DEVICE);
	let (MNISTX,_) = RayBNN_DataLoader::Dataset::CSV::file_to_hash_cpu::<f32>(
    	"./test_data/MNISTX.dat",
    	input_size,
		batch_size
    );


	let (MNISTY,_) = RayBNN_DataLoader::Dataset::CSV::file_to_hash_cpu::<f32>(
    	"./test_data/MNISTY.dat",
    	output_size,
		batch_size
    );
	arrayfire::sync(DEVICE);
	





    let Z_dims = arrayfire::Dim4::new(&[neuron_size,batch_size,proc_num,1]);
	let mut Z = arrayfire::constant::<f32>(0.0,Z_dims);
	let mut Q = arrayfire::constant::<f32>(0.0,Z_dims);



	let mut alpha0: f32 =  0.0091;
	//let mut alpha1: f32 = alpha0;

	


	let active_size = neuron_idx.dims()[0];
	let idxsel = arrayfire::rows(&neuron_idx, (active_size-output_size)  as i64, (active_size-1)  as i64);
    let Qslices: u64 = Q.dims()[2];




	let mut WRowIdxCOO = RayBNN_Sparse::Util::Convert::CSR_to_COO(&WRowIdxCSR);






	let total_param_size = WValues.dims()[0]  +  H.dims()[0]  +  A.dims()[0]    +  B.dims()[0]    +  C.dims()[0]    +  D.dims()[0]   +  E.dims()[0]  ;
	let mt_dims = arrayfire::Dim4::new(&[total_param_size,1,1,1]);
	let mut mt = arrayfire::constant::<f32>(0.0,mt_dims);
	let mut vt = arrayfire::constant::<f32>(0.0,mt_dims);
	let mut grad = arrayfire::constant::<f32>(0.0,mt_dims);






	//let X_dims = arrayfire::Dim4::new(&[input_size,batch_size,proc_num,1]);
    //let mut X = arrayfire::constant::<f32>(0.0,X_dims);










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












	let train_X_dims = arrayfire::Dim4::new(&[input_size,batch_size,1,1]);
	let mut train_X =  arrayfire::constant::<f32>(0.0,temp_dims);
	let Y_dims = arrayfire::Dim4::new(&[output_size,batch_size,1,1]);
	let mut Y =  arrayfire::constant::<f32>(0.0,temp_dims);
	let mut batch_idx = 0;
	let epoch_num = (train_size/batch_size);








	let mut network_params = arrayfire::constant::<f32>(0.0,mt_dims);
	arrayfire::assign_seq(&mut network_params, &Wseqs, &WValues);
	arrayfire::assign_seq(&mut network_params, &Hseqs, &H);
	arrayfire::assign_seq(&mut network_params, &Aseqs, &A);
	arrayfire::assign_seq(&mut network_params, &Bseqs, &B);
	arrayfire::assign_seq(&mut network_params, &Cseqs, &C);
	arrayfire::assign_seq(&mut network_params, &Dseqs, &D);	
	arrayfire::assign_seq(&mut network_params, &Eseqs, &E);	




	let beta0 = arrayfire::constant::<f32>(0.9,arrayfire::Dim4::new(&[1, 1, 1, 1]));
	let beta1 = arrayfire::constant::<f32>(0.999,arrayfire::Dim4::new(&[1, 1, 1, 1]));



	arrayfire::device_gc();
	arrayfire::sync(DEVICE);
	for i in 0..1300
	{
		batch_idx = i % epoch_num;

		train_X =  arrayfire::Array::new(&MNISTX[&batch_idx], train_X_dims);

		Y =  arrayfire::Array::new(&MNISTY[&batch_idx], Y_dims);
		//arrayfire::set_slice(&mut X, &train_X, 0);

		let X = train_X;





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



		RayBNN_Neural::Network::Neurons::state_space_backward_group2(
			//&netdata,
			&modeldata_int,



			&X,
		
		

			&network_params,
		
		
		
		
			&Z,
			&Q,
			&Y,
			loss_grad,
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
		

	
	

		RayBNN_Optimizer::Continuous::GD::adam(
			&beta0
			,&beta1
			,&mut grad
			,&mut mt
			,&mut vt);
	


		network_params = network_params + (alpha0*-grad.clone());


	}


	arrayfire::device_gc();
	arrayfire::sync(DEVICE);



	let mut yhatidx = arrayfire::constant::<u32>(0,temp_dims);
	let mut yidx = arrayfire::constant::<u32>(0,temp_dims);


	batch_idx = (train_size/batch_size);

	let testnum = test_size/batch_size;
	for i in 0..testnum
	{
		train_X =  arrayfire::Array::new(&MNISTX[&batch_idx], train_X_dims);

		Y =  arrayfire::Array::new(&MNISTY[&batch_idx], Y_dims);
		//arrayfire::set_slice(&mut X, &train_X, 0);

		let X = train_X;


		clusterdiffeq::neural::network_f32::state_space_forward_batch(
			&netdata,
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





		//Get Yhat
		let mut idxrs = arrayfire::Indexer::default();
		let seq1 = arrayfire::Seq::new(0.0f32, (batch_size-1) as f32, 1.0);
		let seq2 = arrayfire::Seq::new((proc_num-1) as f32, (Qslices-1) as f32, 1.0);
		idxrs.set_index(&idxsel, 0, None);
		idxrs.set_index(&seq1, 1, None);
		idxrs.set_index(&seq2, 2, None);
		let Yhat = arrayfire::index_gen(&Q, idxrs);


		let (_,mut yhatidx_temp) = arrayfire::imax(&Yhat,0);
		let (_,mut yidx_temp) = arrayfire::imax(&Y,0);
	
		yhatidx_temp = arrayfire::transpose(&yhatidx_temp, false);
		yidx_temp = arrayfire::transpose(&yidx_temp, false);


		yhatidx = arrayfire::join(0, &yhatidx, &yhatidx_temp);
		yidx = arrayfire::join(0, &yidx, &yidx_temp);

		batch_idx = batch_idx+1;
	}

	yhatidx = arrayfire::rows(&yhatidx,1,(yhatidx.dims()[0]-1) as i64);
	yidx = arrayfire::rows(&yidx,1,(yidx.dims()[0]-1) as i64);

	//println!("yhatidx.dims()[0] {}",yhatidx.dims()[0]);
	//println!("yidx.dims()[0] {}",yidx.dims()[0]);


    let confusion = clusterdiffeq::optimal::measure_u32::confusion_matrix(&yhatidx,&yidx,output_size);

    let diag = arrayfire::diag_extract(&confusion,0);
    let (correct,_) = arrayfire::sum_all(&diag);


	println!("correct :{}", correct);
	assert!(correct >=  9860);
	
	let (totalall,_) = arrayfire::sum_all(&confusion);
	assert_eq!(totalall as u64,  test_size);





}
