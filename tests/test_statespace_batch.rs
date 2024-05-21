#![allow(unused_parens)]
#![allow(non_snake_case)]

extern crate arrayfire;
extern crate clusterdiffeq;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;

use rayon::prelude::*;


#[test]
fn test_statespace_batch() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);



    let W_dims = arrayfire::Dim4::new(&[1,400,1,1]);
    let mut W = clusterdiffeq::export::dataloader_f64::file_to_matrix(
        "./test_data/batch_W.csv",
        W_dims
    );
    let W_dims = arrayfire::Dim4::new(&[20,20,1,1]);

    W = arrayfire::moddims(&W, W_dims);

    W = arrayfire::sparse_from_dense(&W, arrayfire::SparseFormat::CSR);







    let X_dims = arrayfire::Dim4::new(&[1,105,1,1]);
    let mut X = clusterdiffeq::export::dataloader_f64::file_to_matrix(
        "./test_data/batch_X.csv",
        X_dims
    );

    let X_dims = arrayfire::Dim4::new(&[5,3,7,1]);

    X = arrayfire::moddims(&X, X_dims);









    let H_dims = arrayfire::Dim4::new(&[1,20,1,1]);
    let mut H = clusterdiffeq::export::dataloader_f64::file_to_matrix(
        "./test_data/batch_H.csv",
        H_dims
    );

    H = arrayfire::transpose(&H, false);



    let mut A = clusterdiffeq::export::dataloader_f64::file_to_matrix(
        "./test_data/batch_A.csv",
        H_dims
    );

    A = arrayfire::transpose(&A, false);




    let mut B = clusterdiffeq::export::dataloader_f64::file_to_matrix(
        "./test_data/batch_B.csv",
        H_dims
    );

    B = arrayfire::transpose(&B, false);




    let mut C = clusterdiffeq::export::dataloader_f64::file_to_matrix(
        "./test_data/batch_C.csv",
        H_dims
    );

    C = arrayfire::transpose(&C, false);




    let mut D = clusterdiffeq::export::dataloader_f64::file_to_matrix(
        "./test_data/batch_D.csv",
        H_dims
    );

    D = arrayfire::transpose(&D, false);




    let mut E = clusterdiffeq::export::dataloader_f64::file_to_matrix(
        "./test_data/batch_E.csv",
        H_dims
    );

    E = arrayfire::transpose(&E, false);




    let Z_dims = arrayfire::Dim4::new(&[20,3,7,1]);

    let mut Z = arrayfire::constant::<f64>(0.0,Z_dims);
	let mut Q = arrayfire::constant::<f64>(0.0,Z_dims);



    let netdata: clusterdiffeq::neural::network_f64::network_metadata_type = clusterdiffeq::neural::network_f64::network_metadata_type {
		neuron_size: 20,
	    input_size: 5,
		output_size: 2,
		proc_num: 7,
		active_size: 30,
		space_dims: 3,
		step_num: 100,
        batch_size: 3,
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








    let WValues = arrayfire::sparse_get_values(&W);
    let WRowIdxCSR = arrayfire::sparse_get_row_indices(&W);
	let WColIdx = arrayfire::sparse_get_col_indices(&W);




    
    let Wdims0 = WValues.dims()[0];
    let Hdims0 = H.dims()[0];
    let Adims0 = A.dims()[0];
    let Bdims0 = B.dims()[0];
    let Cdims0 = C.dims()[0];
    let Ddims0 = D.dims()[0];
    let Edims0 = E.dims()[0];



    let Wstart = 0;
    let Wend = (Wdims0 as i64) - 1;

    let Hstart = Wend + 1; 
    let Hend = Hstart + (Hdims0 as i64) - 1;

    let Astart = Hend + 1; 
    let Aend = Astart + (Adims0 as i64) - 1;

    let Bstart = Aend + 1; 
    let Bend = Bstart + (Bdims0 as i64) - 1;

    let Cstart = Bend + 1; 
    let Cend = Cstart + (Cdims0 as i64) - 1;

    let Dstart = Cend + 1; 
    let Dend = Dstart + (Ddims0 as i64) - 1;

    let Estart = Dend + 1; 
    let Eend = Estart + (Edims0 as i64) - 1;


    let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
    let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
    let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
    let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
    let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
    let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
    let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];







	let total_param_size = WValues.dims()[0]  +  H.dims()[0]  +  A.dims()[0]    +  B.dims()[0]    +  C.dims()[0]    +  D.dims()[0]   +  E.dims()[0]  ;
	let mt_dims = arrayfire::Dim4::new(&[total_param_size,1,1,1]);


	let mut network_params = arrayfire::constant::<f64>(0.0,mt_dims);
	arrayfire::assign_seq(&mut network_params, &Wseqs, &WValues);
	arrayfire::assign_seq(&mut network_params, &Hseqs, &H);
	arrayfire::assign_seq(&mut network_params, &Aseqs, &A);
	arrayfire::assign_seq(&mut network_params, &Bseqs, &B);
	arrayfire::assign_seq(&mut network_params, &Cseqs, &C);
	arrayfire::assign_seq(&mut network_params, &Dseqs, &D);	
	arrayfire::assign_seq(&mut network_params, &Eseqs, &E);	



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







    let Z_dims = arrayfire::Dim4::new(&[1,420,1,1]);
    let mut Z_act = clusterdiffeq::export::dataloader_f64::file_to_matrix(
        "./test_data/batch_Z.csv",
        Z_dims
    );

    let Z_dims = arrayfire::Dim4::new(&[20,3,7,1]);

    Z_act = arrayfire::moddims(&Z_act, Z_dims);

    let mut Z_act_cpu = vec!(f64::default();Z.elements());

    Z_act.host(&mut Z_act_cpu);







    let mut Z_pred_cpu = vec!(f64::default();Z.elements());

    Z.host(&mut Z_pred_cpu);

    Z_pred_cpu = Z_pred_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    Z_act_cpu = Z_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();


    assert_eq!(Z_pred_cpu, Z_act_cpu);








    let Q_dims = arrayfire::Dim4::new(&[1,420,1,1]);
    let mut Q_act = clusterdiffeq::export::dataloader_f64::file_to_matrix(
        "./test_data/batch_Q.csv",
        Q_dims
    );

    let Q_dims = arrayfire::Dim4::new(&[20,3,7,1]);

    Q_act = arrayfire::moddims(&Q_act, Q_dims);

    let mut Q_act_cpu = vec!(f64::default();Q.elements());

    Q_act.host(&mut Q_act_cpu);




    let mut Q_pred_cpu = vec!(f64::default();Q.elements());

    Q.host(&mut Q_pred_cpu);


    Q_pred_cpu = Q_pred_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    Q_act_cpu = Q_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();


    assert_eq!(Q_pred_cpu, Q_act_cpu);
}
