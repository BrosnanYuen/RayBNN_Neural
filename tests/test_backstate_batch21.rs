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
fn test_backstate_batch21() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);



	let neuron_size: u64 = 600;
	let input_size: u64 = 4;
	let output_size: u64 = 3;
	let proc_num: u64 = 4;
	let active_size: u64 = 25;
	let space_dims: u64 = 3;
	let sim_steps: u64 = 10;
    let mut batch_size: u64 = 105;

	/*
	let mut netdata: clusterdiffeq::neural::network_f64::network_metadata_type = clusterdiffeq::neural::network_f64::network_metadata_type {
		neuron_size: neuron_size,
	    input_size: input_size,
		output_size: output_size,
		proc_num: proc_num,
		active_size: active_size,
		space_dims: space_dims,
		step_num: sim_steps,
        batch_size: batch_size,
		del_unused_neuron: true,

		time_step: 0.1,
		nratio: 0.5,
		neuron_std: 0.06,
		sphere_rad: 0.9,
		neuron_rad: 0.1,
		con_rad: 0.6,
        init_prob: 0.5,
		add_neuron_rate: 0.0,
		del_neuron_rate: 0.0,
		center_const: 0.005,
		spring_const: 0.01,
		repel_const: 10.0
	};
	 */

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


	let mut glia_pos = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut neuron_pos = arrayfire::constant::<f64>(0.0,temp_dims);



	
	let mut H = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut A = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut B = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut C = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut D = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut E = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut neuron_idx = arrayfire::constant::<i32>(0,temp_dims);





	let mut WValues = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut WRowIdxCSR = arrayfire::constant::<i32>(0,temp_dims);
	let mut WColIdx = arrayfire::constant::<i32>(0,temp_dims);




	/*
	clusterdiffeq::export::dataloader_f64::load_network(
			"./test_data/network_test.csv",
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
		"./test_data/network_batch21/", 
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
 

	







	let train_X_dims = arrayfire::Dim4::new(&[4, 105, 1, 1]);
    let mut X = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/flower_train_X.csv",
    	//train_X_dims
    );




	let train_Y_dims = arrayfire::Dim4::new(&[3, 105, 1, 1]);
    let mut train_Y = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/flower_train_Y.csv",
    	//train_Y_dims
    );






	let test_Y_dims = arrayfire::Dim4::new(&[3, 45, 1, 1]);
    let mut test_Y = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/flower_test_Y.csv",
    	//test_Y_dims
    );





	let test_X_dims = arrayfire::Dim4::new(&[4, 45, 1, 1]);
    let mut test_X = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/flower_test_X.csv",
    	//test_X_dims
    );





    let Z_dims = arrayfire::Dim4::new(&[neuron_size,batch_size,proc_num,1]);
	let mut Z = arrayfire::constant::<f64>(0.0,Z_dims);
	let mut Q = arrayfire::constant::<f64>(0.0,Z_dims);



	let mut alpha = 0.01;





	let total_param_size = WValues.dims()[0]  +  H.dims()[0]  +  A.dims()[0]    +  B.dims()[0]    +  C.dims()[0]    +  D.dims()[0]   +  E.dims()[0]  ;
	let mt_dims = arrayfire::Dim4::new(&[total_param_size,1,1,1]);
	let mut mt = arrayfire::constant::<f64>(0.0,mt_dims);
	let mut vt = arrayfire::constant::<f64>(0.0,mt_dims);
	let mut grad = arrayfire::constant::<f64>(0.0,mt_dims);




	let mut WRowIdxCOO = RayBNN_Sparse::Util::Convert::CSR_to_COO(&WRowIdxCSR);

    //let X_dims = arrayfire::Dim4::new(&[input_size,batch_size,proc_num,1]);
    //let mut X = arrayfire::constant::<f64>(0.0,X_dims);

    //arrayfire::set_slice(&mut X, &train_X, 0);



    let Y = train_Y;






	let mut idxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
	let mut valsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();

	let mut cvec_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
	let mut dXsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();

	let mut nrows_out: nohash_hasher::IntMap<i64, u64> = nohash_hasher::IntMap::default();
	let mut sparseval_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
	let mut sparsecol_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
	let mut sparserow_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();




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
		Y.dims()[2], 
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











	for i in 0..50
	{


		clusterdiffeq::neural::network_f64::state_space_forward_batch(
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





		clusterdiffeq::neural::network_f64::state_space_backward_group2(
			&netdata,
			&X,
		
		

			&network_params,
		
		
		
		
			&Z,
			&Q,
			&Y,
			clusterdiffeq::optimal::loss_f64::sigmoid_cross_entropy_grad,
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
		

	


		clusterdiffeq::optimal::gd_f64::adam(
			0.9
			,0.999
			,&mut grad
			,&mut mt
			,&mut vt);
	


		network_params = network_params + (alpha*-grad.clone());


		

	
	



		

	}





    batch_size = 45;

	let netdata: clusterdiffeq::neural::network_f64::network_metadata_type = clusterdiffeq::neural::network_f64::network_metadata_type {
		neuron_size: neuron_size,
	    input_size: input_size,
		output_size: output_size,
		proc_num: proc_num,
		active_size: active_size,
		space_dims: space_dims,
		step_num: sim_steps,
        batch_size: batch_size,
		del_unused_neuron: true,

		time_step: 0.1,
		nratio: 0.5,
		neuron_std: 0.06,
		sphere_rad: 0.9,
		neuron_rad: 0.1,
		con_rad: 0.6,
        init_prob: 0.5,
		add_neuron_rate: 0.0,
		del_neuron_rate: 0.0,
		center_const: 0.005,
		spring_const: 0.01,
		repel_const: 10.0
	};


    //let X_dims = arrayfire::Dim4::new(&[input_size,batch_size,proc_num,1]);
    //let mut X = arrayfire::constant::<f64>(0.0,X_dims);

    //arrayfire::set_slice(&mut X, &test_X, 0);
	X = test_X;

    let Y = test_Y;

    let active_size = neuron_idx.dims()[0];
	let idxsel = arrayfire::rows(&neuron_idx, (active_size-output_size)  as i64, (active_size-1)  as i64);
    let Qslices: u64 = Q.dims()[2];


    let Z_dims = arrayfire::Dim4::new(&[neuron_size,batch_size,proc_num,1]);
	let mut Z = arrayfire::constant::<f64>(0.0,Z_dims);
	let mut Q = arrayfire::constant::<f64>(0.0,Z_dims);



	clusterdiffeq::neural::network_f64::state_space_forward_batch(
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
    let seq1 = arrayfire::Seq::new(0.0f64, (batch_size-1) as f64, 1.0);
    let seq2 = arrayfire::Seq::new((proc_num-1) as f64, (Qslices-1) as f64, 1.0);
    idxrs.set_index(&idxsel, 0, None);
    idxrs.set_index(&seq1, 1, None);
    idxrs.set_index(&seq2, 2, None);
    let Yhat = arrayfire::index_gen(&Q, idxrs);

    let (_,mut yhatidx) = arrayfire::imax(&Yhat,0);
    let (_,mut yidx) = arrayfire::imax(&Y,0);

    yhatidx = arrayfire::transpose(&yhatidx, false);
    yidx = arrayfire::transpose(&yidx, false);

    let confusion = clusterdiffeq::optimal::measure_u32::confusion_matrix(&yhatidx,&yidx,output_size);

    let diag = arrayfire::diag_extract(&confusion,0);
    let (correct,_) = arrayfire::sum_all(&diag);



    assert_eq!(correct, 45);

}
