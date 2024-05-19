
use arrayfire;



use std::collections::HashMap;
use nohash_hasher;

use crate::Network::Activation::UAF;
use crate::Network::Activation::deriUAF;


use RayBNN_Sparse::Util::Search::COO_batch_find;

use RayBNN_Sparse::Util::Search::find_unique;


const COO_FIND_LIMIT: u64 = 1500000000;



const ZERO_F64: f64 = 0.0;
const ONE_F64: f64 = 1.0;
const HIGH_F64: f64 = 1000000.0;
const TWO_F64: f64 = 2.0;
const ONEHALF_F64: f64 = 0.5f64;


const EPS1_F64: f64 = 0.0001;
const EPS2_F64: f64 = 0.00001;
const EPS3_F64: f64 = 0.0000001;

const CONST1_F64: f64 = 2.12616013f64;
const CONST2_F64: f64 = (1.0f64/2.12616013f64);




pub fn UAF_initial_as_identity<Z: arrayfire::FloatingPoint>(
    modeldata_float:  &HashMap<String, f64>,
    modeldata_int:  &HashMap<String, u64>,


    A: &mut arrayfire::Array<Z>,
    B: &mut arrayfire::Array<Z>,
    C: &mut arrayfire::Array<Z>,
    D: &mut arrayfire::Array<Z>,
    E: &mut arrayfire::Array<Z>
)
{
    /* 
    let neuron_size: u64 = modeldata_int["neuron_size"].clone();
    let input_size: u64 = modeldata_int["input_size"].clone();
    let output_size: u64 = modeldata_int["output_size"].clone();
    let proc_num: u64 = modeldata_int["proc_num"].clone();
    let active_size: u64 = modeldata_int["active_size"].clone();
    let space_dims: u64 = modeldata_int["space_dims"].clone();
    let step_num: u64 = modeldata_int["step_num"].clone();


    let time_step: f64 = modeldata_float["time_step"].clone();
    let nratio: f64 = modeldata_float["nratio"].clone();
    let neuron_std: f64 = modeldata_float["neuron_std"].clone();
    let sphere_rad: f64 = modeldata_float["sphere_rad"].clone();
    let neuron_rad: f64 = modeldata_float["neuron_rad"].clone();
    let con_rad: f64 = modeldata_float["con_rad"].clone();
    let center_const: f64 = modeldata_float["center_const"].clone();
    let spring_const: f64 = modeldata_float["spring_const"].clone();
    let repel_const: f64 = modeldata_float["repel_const"].clone();
    */
    let neuron_size: u64 = modeldata_int["neuron_size"].clone();
    let neuron_std: f64 = modeldata_float["neuron_std"].clone();

    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let ONE = arrayfire::constant::<f64>(ONE_F64,single_dims).cast::<Z>();
    let EPS1 = arrayfire::constant::<f64>(EPS1_F64,single_dims).cast::<Z>();
    let EPS2 = arrayfire::constant::<f64>(EPS2_F64,single_dims).cast::<Z>();
    let NEURONSTD = arrayfire::constant::<f64>(neuron_std,single_dims).cast::<Z>();




    let H_dims = arrayfire::Dim4::new(&[neuron_size,1,1,1]);
    
    *A = ONE.clone() + EPS1.clone()*NEURONSTD.clone()*arrayfire::randn::<Z>(H_dims);
    *B = EPS1*NEURONSTD.clone()*arrayfire::randn::<Z>(H_dims);
    *C = EPS2.clone()*NEURONSTD.clone()*arrayfire::randn::<Z>(H_dims);
    *D = EPS2.clone()*NEURONSTD.clone()*arrayfire::randn::<Z>(H_dims) -ONE;
    *E = EPS2*NEURONSTD*arrayfire::randn::<Z>(H_dims);


}






pub fn UAF_initial_as_tanh<Z: arrayfire::FloatingPoint>(
    modeldata_float:  &HashMap<String, f64>,
    modeldata_int:  &HashMap<String, u64>,



    A: &mut arrayfire::Array<Z>,
    B: &mut arrayfire::Array<Z>,
    C: &mut arrayfire::Array<Z>,
    D: &mut arrayfire::Array<Z>,
    E: &mut arrayfire::Array<Z>)
{
    
    let neuron_size: u64 = modeldata_int["neuron_size"].clone();
    let neuron_std: f64 = modeldata_float["neuron_std"].clone();

    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let ONE = arrayfire::constant::<f64>(ONE_F64,single_dims).cast::<Z>();
    let EPS1 = arrayfire::constant::<f64>(EPS1_F64,single_dims).cast::<Z>();
    let EPS2 = arrayfire::constant::<f64>(EPS2_F64,single_dims).cast::<Z>();
    let NEURONSTD = arrayfire::constant::<f64>(neuron_std,single_dims).cast::<Z>();
    let EPS3 = arrayfire::constant::<f64>(EPS3_F64,single_dims).cast::<Z>();
    let CONST1 = arrayfire::constant::<f64>(CONST1_F64,single_dims).cast::<Z>();
    let CONST2 = arrayfire::constant::<f64>(CONST2_F64,single_dims).cast::<Z>();


    

    let H_dims = arrayfire::Dim4::new(&[neuron_size,1,1,1]);
    
    *A = CONST1.clone() + EPS1.clone()*NEURONSTD.clone()*arrayfire::randn::<Z>(H_dims);
    *B = CONST2 + EPS1.clone()*NEURONSTD.clone()*arrayfire::randn::<Z>(H_dims);
    *C = EPS3*NEURONSTD.clone()*arrayfire::randn::<Z>(H_dims);
    *D = CONST1 + EPS1.clone()*NEURONSTD.clone()*arrayfire::randn::<Z>(H_dims);
    *E = EPS1*NEURONSTD*arrayfire::randn::<Z>(H_dims) - ONE;


}






pub fn xavier_init(
    in_idx: &arrayfire::Array<i32>,
    WRowIdxCOO: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,
    neuron_size: u64,
    depth: u64,

    WValues: &mut arrayfire::Array<f64>,
    H: &mut arrayfire::Array<f64>,
)
{



        
    *WValues = 0.000001f64*arrayfire::randn::<f64>(WValues.dims());


    let H_dims = arrayfire::Dim4::new(&[neuron_size,1,1,1]);
    *H = 0.000001f64*arrayfire::randn::<f64>(H_dims);



    let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);

    let mut out_idx = in_idx.clone();
    let mut valsel = arrayfire::constant::<i32>(0,temp_dims);

    let COO_batch_size = 1 + ((COO_find_limit/WColIdx.dims()[0]) as u64);


    let mut input_degree = 0;
    let mut output_degree = 0;

    for i in 0..depth
    {
        valsel = COO_batch_find(WColIdx, &out_idx, COO_batch_size);
        if valsel.dims()[0] == 0
        {
            break;
        }
        out_idx = arrayfire::lookup(WRowIdxCOO, &valsel, 0);

        out_idx = find_unique(&out_idx, neuron_size);


        output_degree = valsel.dims()[0];

        let mulitiplier = (6.0f64/((input_degree + output_degree) as f64) ).sqrt()*2.0f64;
        let mut newWValues = arrayfire::randu::<f64>(valsel.dims());
        newWValues = (newWValues - 0.5f64)*( mulitiplier );

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&valsel, 0, None);
        arrayfire::assign_gen(WValues, &idxrs, &newWValues);

        drop(newWValues);
        drop(idxrs);


        let mut newH = arrayfire::randu::<f64>(out_idx.dims());
        newH = (newH - 0.5f64)*( mulitiplier );

        let mut idxrs2 = arrayfire::Indexer::default();
        idxrs2.set_index(&out_idx, 0, None);
        arrayfire::assign_gen(H, &idxrs2, &newH);



        if out_idx.dims()[0] == 0
        {
            break;
        }

        input_degree = valsel.dims()[0];
    }


}








