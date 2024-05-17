#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;
use RayBNN_Graph;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;




#[test]
fn test_graph() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);







    let neuron_size: u64 = 16;

    let matrix_dims = arrayfire::Dim4::new(&[neuron_size,neuron_size,1,1]);
    let mut W: arrayfire::Array<f64> = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire(
    	"./test_data/sparse_test.csv",
    );



    W = arrayfire::sparse_from_dense(&W, arrayfire::SparseFormat::COO);

    let mut WValues = arrayfire::sparse_get_values(&W);
	let mut WRowIdxCOO = arrayfire::sparse_get_row_indices(&W);
	let mut WColIdx = arrayfire::sparse_get_col_indices(&W);






    let in_idx_cpu:Vec<i32> = vec![0,1,2];
	let mut in_idx = arrayfire::Array::new(&in_idx_cpu, arrayfire::Dim4::new(&[3, 1, 1, 1]));


    let mut out_idx = in_idx.clone();
    let mut total_idx = in_idx.clone();
    
    RayBNN_Graph::COO::Traversal::traverse_forward(
        &in_idx,
        &WRowIdxCOO,
        &WColIdx,
        neuron_size,
        1,
        &mut out_idx
    );


    let out_idx_act:Vec<i32> = vec![3,4,5,6];

    let mut out_idx_cpu = vec!(i32::default();out_idx.elements());
    out_idx.host(&mut out_idx_cpu);

    assert_eq!(out_idx_act, out_idx_cpu);

















    let in_idx_cpu:Vec<i32> = vec![0,1,2];
	let mut in_idx = arrayfire::Array::new(&in_idx_cpu, arrayfire::Dim4::new(&[3, 1, 1, 1]));


    let mut out_idx = in_idx.clone();
    let mut total_idx = in_idx.clone();
    RayBNN_Graph::COO::Traversal::traverse_forward(
        &in_idx,
        &WRowIdxCOO,
        &WColIdx,
        neuron_size,
        2,
        &mut out_idx
    );

    let out_idx_act:Vec<i32> = vec![9,10,11,12];

    let mut out_idx_cpu = vec!(i32::default();out_idx.elements());
    out_idx.host(&mut out_idx_cpu);

    assert_eq!(out_idx_act, out_idx_cpu);



















    let in_idx_cpu:Vec<i32> = vec![0,1,2];
	let mut in_idx = arrayfire::Array::new(&in_idx_cpu, arrayfire::Dim4::new(&[3, 1, 1, 1]));


    let mut out_idx = in_idx.clone();
    let mut total_idx = in_idx.clone();
    RayBNN_Graph::COO::Traversal::traverse_forward(
        &in_idx,
        &WRowIdxCOO,
        &WColIdx,
        neuron_size,
        3,
        &mut out_idx
    );

    let out_idx_act:Vec<i32> = vec![14,15];

    let mut out_idx_cpu = vec!(i32::default();out_idx.elements());
    out_idx.host(&mut out_idx_cpu);

    assert_eq!(out_idx_act, out_idx_cpu);




















    let in_idx_cpu:Vec<i32> = vec![14,15];
	let mut in_idx = arrayfire::Array::new(&in_idx_cpu, arrayfire::Dim4::new(&[3, 1, 1, 1]));


    let mut out_idx = in_idx.clone();
    let mut total_idx = in_idx.clone();
    RayBNN_Graph::COO::Traversal::traverse_backward(
        &in_idx,
        &WRowIdxCOO,
        &WColIdx,
        neuron_size,
        1,
        &mut out_idx
    );

    let out_idx_act:Vec<i32> = vec![10,11];

    let mut out_idx_cpu = vec!(i32::default();out_idx.elements());
    out_idx.host(&mut out_idx_cpu);

    assert_eq!(out_idx_act, out_idx_cpu);














    let in_idx_cpu:Vec<i32> = vec![14,15];
	let mut in_idx = arrayfire::Array::new(&in_idx_cpu, arrayfire::Dim4::new(&[3, 1, 1, 1]));


    let mut out_idx = in_idx.clone();
    let mut total_idx = in_idx.clone();
    RayBNN_Graph::COO::Traversal::traverse_backward(
        &in_idx,
        &WRowIdxCOO,
        &WColIdx,
        neuron_size,
        2,
        &mut out_idx
    );

    let out_idx_act:Vec<i32> = vec![3,5,6];

    let mut out_idx_cpu = vec!(i32::default();out_idx.elements());
    out_idx.host(&mut out_idx_cpu);

    assert_eq!(out_idx_act, out_idx_cpu);




















    let in_idx_cpu:Vec<i32> = vec![14,15];
	let mut in_idx = arrayfire::Array::new(&in_idx_cpu, arrayfire::Dim4::new(&[3, 1, 1, 1]));


    let mut out_idx = in_idx.clone();
    let mut total_idx = in_idx.clone();
    RayBNN_Graph::COO::Traversal::traverse_backward(
        &in_idx,
        &WRowIdxCOO,
        &WColIdx,
        neuron_size,
        3,
        &mut out_idx
    );

    let out_idx_act:Vec<i32> = vec![0,1,2];

    let mut out_idx_cpu = vec!(i32::default();out_idx.elements());
    out_idx.host(&mut out_idx_cpu);

    assert_eq!(out_idx_act, out_idx_cpu);



}
