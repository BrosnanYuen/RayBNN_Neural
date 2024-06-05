#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;
use RayBNN_Graph;
use RayBNN_Neural;
use RayBNN_Optimizer;

use std::collections::HashMap;
use nohash_hasher;


use ndarray;
use ndarray_npy;



const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;




#[test]
fn test_backstate_batch11() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

	let rand_seed = 4324;

	arrayfire::set_seed(rand_seed);



	let W_nparr: ndarray::Array2<f64> = ndarray_npy::read_npy("./test_data/difftest/W.npy").unwrap();
	let X_nparr: ndarray::Array2<f64> = ndarray_npy::read_npy("./test_data/difftest/X.npy").unwrap();
	let Y_nparr: ndarray::Array2<f64> = ndarray_npy::read_npy("./test_data/difftest/Y.npy").unwrap();

	let A_nparr: ndarray::Array2<f64> = ndarray_npy::read_npy("./test_data/difftest/A.npy").unwrap();
	let B_nparr: ndarray::Array2<f64> = ndarray_npy::read_npy("./test_data/difftest/B.npy").unwrap();
	let C_nparr: ndarray::Array2<f64> = ndarray_npy::read_npy("./test_data/difftest/C.npy").unwrap();
	let D_nparr: ndarray::Array2<f64> = ndarray_npy::read_npy("./test_data/difftest/D.npy").unwrap();
	let E_nparr: ndarray::Array2<f64> = ndarray_npy::read_npy("./test_data/difftest/E.npy").unwrap();
	let H_nparr: ndarray::Array2<f64> = ndarray_npy::read_npy("./test_data/difftest/H.npy").unwrap();


	let gW_nparr: ndarray::Array2<f64> = ndarray_npy::read_npy("./test_data/difftest/gW.npy").unwrap();

	let gA_nparr: ndarray::Array2<f64> = ndarray_npy::read_npy("./test_data/difftest/gA.npy").unwrap();
	let gB_nparr: ndarray::Array2<f64> = ndarray_npy::read_npy("./test_data/difftest/gB.npy").unwrap();
	let gC_nparr: ndarray::Array2<f64> = ndarray_npy::read_npy("./test_data/difftest/gC.npy").unwrap();
	let gD_nparr: ndarray::Array2<f64> = ndarray_npy::read_npy("./test_data/difftest/gD.npy").unwrap();
	let gE_nparr: ndarray::Array2<f64> = ndarray_npy::read_npy("./test_data/difftest/gE.npy").unwrap();
	let gH_nparr: ndarray::Array2<f64> = ndarray_npy::read_npy("./test_data/difftest/gH.npy").unwrap();



	let neuron_size= W_nparr.dim().0 as u64;
	let input_size= X_nparr.dim().0 as u64;
	let output_size= Y_nparr.dim().0 as u64;
	let batch_size= Y_nparr.dim().1 as u64;

	//println!("X_nparr {:?}",X_nparr);
	println!("neuron_size {}",neuron_size);
	println!("input_size {}",input_size);
	println!("output_size {}",output_size);
	println!("batch_size {}",batch_size);

	


	let proc_num: u64 = 4;


	

	let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	/* 
	let mut netdata = clusterdiffeq::neural::network_f64::create_nullnetdata();
	netdata.input_size = input_size;
	netdata.output_size = output_size;
	netdata.neuron_size = neuron_size;
	netdata.proc_num = proc_num;
	netdata.active_size = neuron_size;
	netdata.batch_size = batch_size;
	*/


    let mut modeldata_int: HashMap<String,u64> = HashMap::new();

    modeldata_int.insert("neuron_size".to_string(), neuron_size);
    modeldata_int.insert("input_size".to_string(), input_size);
    modeldata_int.insert("output_size".to_string(), output_size);
    modeldata_int.insert("proc_num".to_string(), proc_num);
    modeldata_int.insert("active_size".to_string(), neuron_size);
    //modeldata_int.insert("space_dims".to_string(), space_dims);
    //modeldata_int.insert("step_num".to_string(), step_num);
    modeldata_int.insert("batch_size".to_string(), batch_size);



	

	let mut neuron_idx = arrayfire::constant::<i32>(0,temp_dims);


	let N_dims = arrayfire::Dim4::new(&[neuron_size ,1,1,1]);
	let repeat_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    //let L_dims = arrayfire::Dim4::new(&[output_size,1,1,1]);

	neuron_idx = arrayfire::iota::<i32>(N_dims,repeat_dims);



	let tile_dims = arrayfire::Dim4::new(&[1,batch_size,1,1]);
	



	//let X_cpu: Vec<f64> = vec![0.263700,0.295200,-0.728400,0.834900,-0.619800,-0.310400,            0.052100,0.370600,0.671300,0.307500,-0.926000,-0.639700,           0.503200,-0.781500,-0.027800,-0.000300,0.314200,-0.528200,            -0.645800,-0.899400,0.141500,-0.358300,-0.291600,-0.965600,];
	let X_cpu = X_nparr.into_raw_vec();
	let X = arrayfire::Array::new(&X_cpu, arrayfire::Dim4::new(&[input_size, batch_size, 1, 1]));
	
	
	//let tile_dims = arrayfire::Dim4::new(&[1,1,proc_num,1]);
	//let X =  arrayfire::tile(&train_X, tile_dims);


	//arrayfire::print_gen("X".to_string(), &train_X,Some(6));


	let Y_cpu = Y_nparr.into_raw_vec();
	let mut Y = arrayfire::Array::new(&Y_cpu, arrayfire::Dim4::new(&[output_size, batch_size, 1, 1]));




	let A_cpu = A_nparr.into_raw_vec();
	let mut A = arrayfire::Array::new(&A_cpu, arrayfire::Dim4::new(&[A_cpu.len() as u64, 1, 1, 1]));


	let B_cpu = B_nparr.into_raw_vec();
	let mut B = arrayfire::Array::new(&B_cpu, arrayfire::Dim4::new(&[B_cpu.len() as u64, 1, 1, 1]));


	let C_cpu = C_nparr.into_raw_vec();
	let mut C = arrayfire::Array::new(&C_cpu, arrayfire::Dim4::new(&[C_cpu.len() as u64, 1, 1, 1]));


	let D_cpu = D_nparr.into_raw_vec();
	let mut D = arrayfire::Array::new(&D_cpu, arrayfire::Dim4::new(&[D_cpu.len() as u64, 1, 1, 1]));


	let E_cpu = E_nparr.into_raw_vec();
	let mut E = arrayfire::Array::new(&E_cpu, arrayfire::Dim4::new(&[E_cpu.len() as u64, 1, 1, 1]));


	let H_cpu = H_nparr.into_raw_vec();
	let mut H = arrayfire::Array::new(&H_cpu, arrayfire::Dim4::new(&[H_cpu.len() as u64, 1, 1, 1]));







	




	let W_cpu = W_nparr.into_raw_vec();
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






	//let X_dims = arrayfire::Dim4::new(&[input_size,batch_size,proc_num,1]);
    //let mut X = arrayfire::constant::<f64>(0.0,X_dims);











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
	//arrayfire::set_slice(&mut X, &train_X, 0);




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


	println!("a2");
	
	
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


	



		&mut idxsel_out,
		&mut valsel_out,

		&mut cvec_out,
		&mut dXsel_out,


		&mut nrows_out,
		&mut sparseval_out,
		&mut sparserow_out,
		&mut sparsecol_out,


	
	
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
	
	println!("a3");


	let gW = arrayfire::index(&grad, &Wseqs);

	let mut densegW = arrayfire::sparse(
		neuron_size, 
		neuron_size, 
		&gW, 
		&WRowIdxCSR, 
		&WColIdx, 
		arrayfire::SparseFormat::CSR
	);
	
	densegW = arrayfire::sparse_to_dense(&densegW);
	
	//arrayfire::print_gen("densegW".to_string(), &densegW, Some(9));







	let gW_act_cpu = gW_nparr.into_raw_vec();
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

	let mut gH_act_cpu = gH_nparr.into_raw_vec();
	gH_act_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gH_act_cpu, 9);

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

	let mut gA_act_cpu = gA_nparr.into_raw_vec();
	gA_act_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gA_act_cpu, 9);

	for qq in 0..gA_act_cpu.len()
	{
		if gA_act_cpu[qq] != 0.0
		{
			assert_eq!(gA_act_cpu[qq], gA_cpu[qq]);
		}
		
	}











	let gB = arrayfire::index(&grad, &Bseqs);

	let mut gB_cpu = vec!(f64::default();gB.elements());
	gB.host(&mut gB_cpu);

	gB_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gB_cpu, 9);

	let mut gB_act_cpu = gB_nparr.into_raw_vec();
	gB_act_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gB_act_cpu, 9);

	for qq in 0..gB_act_cpu.len()
	{
		if gB_act_cpu[qq] != 0.0
		{
			assert_eq!(gB_act_cpu[qq], gB_cpu[qq]);
		}
		
	}





	let gC = arrayfire::index(&grad, &Cseqs);

	let mut gC_cpu = vec!(f64::default();gC.elements());
	gC.host(&mut gC_cpu);

	gC_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gC_cpu, 9);

	let mut gC_act_cpu = gC_nparr.into_raw_vec();
	gC_act_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gC_act_cpu, 9);

	for qq in 0..gC_act_cpu.len()
	{
		if gC_act_cpu[qq] != 0.0
		{
			assert_eq!(gC_act_cpu[qq], gC_cpu[qq]);
		}
		
	}




	let gD = arrayfire::index(&grad, &Dseqs);

	let mut gD_cpu = vec!(f64::default();gD.elements());
	gD.host(&mut gD_cpu);

	gD_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gD_cpu, 9);

	let mut gD_act_cpu = gD_nparr.into_raw_vec();
	gD_act_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gD_act_cpu, 9);

	for qq in 0..gD_act_cpu.len()
	{
		if gD_act_cpu[qq] != 0.0
		{
			assert_eq!(gD_act_cpu[qq], gD_cpu[qq]);
		}
	}




	let gE = arrayfire::index(&grad, &Eseqs);

	let mut gE_cpu = vec!(f64::default();gE.elements());
	gE.host(&mut gE_cpu);

	gE_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gE_cpu, 9);

	let mut gE_act_cpu = gE_nparr.into_raw_vec();
	gE_act_cpu = RayBNN_DataLoader::Dataset::Round::rvector(&gE_act_cpu, 9);

	for qq in 0..gE_act_cpu.len()
	{
		if gE_act_cpu[qq] != 0.0
		{
			assert_eq!(gE_act_cpu[qq], gE_cpu[qq]);
		}
	}


}
