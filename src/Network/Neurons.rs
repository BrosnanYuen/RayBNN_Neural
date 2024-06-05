

use arrayfire;



use std::collections::HashMap;
use nohash_hasher;

use crate::Network::Activation::UAF;
use crate::Network::Activation::deriUAF;


use RayBNN_Sparse::Util::Search::COO_batch_find;

use RayBNN_Sparse::Util::Search::find_unique;

use RayBNN_Sparse::Util::Convert::remap_rows;

use RayBNN_Sparse::Util::Convert::get_global_weight_idx2;


const COO_FIND_LIMIT: u64 = 1500000000;



const ZERO_F64: f64 = 0.0;
const ONE_F64: f64 = 1.0;
const HIGH_F64: f64 = 1000000.0;
const TWO_F64: f64 = 2.0;
const ONEHALF_F64: f64 = 0.5f64;



pub fn print_dims<Z: arrayfire::HasAfEnum>(
    arr:  &arrayfire::Array<Z>,
)
{
    for i in 0..4
    {
        println!("dims{}:{}",i,arr.dims()[i]);
    }
}


pub fn print_netdata(
    modeldata_string:  &HashMap<String, String>,
    modeldata_float:  &HashMap<String, f64>,
    modeldata_int:  &HashMap<String, u64>,
)
{
    println!("\n\n******Network Information******");
    for (key, value) in modeldata_string {
        println!("{} : {}", key.clone(), value.clone());
    }

    for (key, value) in modeldata_float {
        println!("{} : {}", key.clone(), value.clone());
    }

    for (key, value) in modeldata_int {
        println!("{} : {}", key.clone(), value.clone());
    }

}


















pub fn find_path_backward_group2(
    //netdata: &network_metadata_type,
    modeldata_int:  &HashMap<String, u64>,



    Xslices: u64,
    Yslices: u64,
    WRowIdxCOO: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,
    neuron_idx: &arrayfire::Array<i32>,


    Wdims0: u64,
    Hdims0: u64,
    Adims0: u64,
    Bdims0: u64,
    Cdims0: u64,
    Ddims0: u64,
    Edims0: u64,



    idxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    valsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    
    cvec_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    dXsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    

    nrows_out: &mut nohash_hasher::IntMap<i64, u64 >,
    sparseval_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    sparserow_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    sparsecol_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,



    Hidxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Aidxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Bidxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Cidxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Didxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Eidxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    combidxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,





    dAseqs_out: &mut nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,
    dBseqs_out: &mut nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,
    dCseqs_out: &mut nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,
    dDseqs_out: &mut nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,
    dEseqs_out: &mut nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,



    
    Wseqs: &mut [arrayfire::Seq<i32>; 1],
    Hseqs: &mut [arrayfire::Seq<i32>; 1],
    Aseqs: &mut [arrayfire::Seq<i32>; 1],
    Bseqs: &mut [arrayfire::Seq<i32>; 1],
    Cseqs: &mut [arrayfire::Seq<i32>; 1],
    Dseqs: &mut [arrayfire::Seq<i32>; 1],
    Eseqs: &mut [arrayfire::Seq<i32>; 1]

) {
    /* 
    let neuron_size: u64 = netdata.neuron_size.clone();
	let input_size: u64 = netdata.input_size.clone();
	let output_size: u64 = netdata.output_size.clone();
	let proc_num: u64 = netdata.proc_num.clone();
	let active_size: u64 = netdata.active_size.clone();
	let space_dims: u64 = netdata.space_dims.clone();
	let step_num: u64 = netdata.step_num.clone();
    let batch_size: u64 = netdata.batch_size.clone();
    */

    let neuron_size: u64 = modeldata_int["neuron_size"].clone();
    //let input_size: u64 = modeldata_int["input_size"].clone();
    let output_size: u64 = modeldata_int["output_size"].clone();
    //let proc_num: u64 = modeldata_int["proc_num"].clone();
    //let active_size: u64 = modeldata_int["active_size"].clone();
    //let space_dims: u64 = modeldata_int["space_dims"].clone();
    //let step_num: u64 = modeldata_int["step_num"].clone();
    //let batch_size: u64 = modeldata_int["batch_size"].clone();




    let COO_batch_size = 1 + ((COO_FIND_LIMIT/WRowIdxCOO.dims()[0]) as u64);



    //Get current selection of neurons
    let active_size = neuron_idx.dims()[0];
    let mut newidxsel = arrayfire::rows(neuron_idx, (active_size-output_size) as i64, (active_size-1)   as i64);
    let mut idxsel = newidxsel.clone();
    let mut output_idxsel = newidxsel.clone();


    
    let mut yslicidx: i64 = (Yslices-1) as i64;



    let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);

    let mut valsel = arrayfire::constant::<i32>(0,temp_dims);
    
    let mut rvec = arrayfire::constant::<i32>(0,temp_dims);

    let mut cvec = arrayfire::constant::<i32>(0,temp_dims);

    let mut dXsel = arrayfire::constant::<i32>(0,temp_dims);


    let mut sparseval = arrayfire::constant::<i32>(0,temp_dims);
    let mut sparserow = arrayfire::constant::<i32>(0,temp_dims);
    let mut sparsecol = arrayfire::constant::<i32>(0,temp_dims);
    let mut gidx1 = arrayfire::constant::<u64>(0,temp_dims);




    let Hoffset = Wdims0 as i32;
    let Aoffset = (Wdims0 + Hdims0) as i32;
    let Boffset = ((Aoffset as u64) + Adims0) as i32;
    let Coffset = ((Boffset as u64) + Bdims0) as i32;
    let Doffset = ((Coffset as u64) + Cdims0) as i32;
    let Eoffset = ((Doffset as u64) + Ddims0) as i32;




    let mut Hidxsel = idxsel.clone();
    let mut Aidxsel = idxsel.clone();
    let mut Bidxsel = idxsel.clone();
    let mut Cidxsel = idxsel.clone();
    let mut Didxsel = idxsel.clone();
    let mut Eidxsel = idxsel.clone();
    let mut combidxsel = idxsel.clone();




    //Main loop
    for i in (0i64..(Xslices as i64)).rev() {
        idxsel = newidxsel.clone();
        idxsel_out.insert(i, idxsel.clone());



        Hidxsel = Hoffset + idxsel.clone();
        Aidxsel = Aoffset + idxsel.clone();
        Bidxsel = Boffset + idxsel.clone();
        Cidxsel = Coffset + idxsel.clone();
        Didxsel = Doffset + idxsel.clone();
        Eidxsel = Eoffset + idxsel.clone();

        Hidxsel_out.insert(i, Hidxsel.clone());
        Aidxsel_out.insert(i, Aidxsel.clone());
        Bidxsel_out.insert(i, Bidxsel.clone());
        Cidxsel_out.insert(i, Cidxsel.clone());
        Didxsel_out.insert(i, Didxsel.clone());
        Eidxsel_out.insert(i, Eidxsel.clone());




        


        let dAsize = idxsel.dims()[0];

        let dAstart = 0;
        let dAend = dAstart + dAsize - 1;

        let dBstart = dAend + 1;
        let dBend = dBstart + dAsize - 1;

        let dCstart = dBend + 1;
        let dCend = dCstart + dAsize - 1;

        let dDstart = dCend + 1;
        let dDend = dDstart + dAsize - 1;

        let dEstart = dDend + 1;
        let dEend = dEstart + dAsize - 1;

        dAseqs_out.insert(i,[arrayfire::Seq::new(dAstart as i32, dAend as i32, 1i32), arrayfire::Seq::default() ] );
        dBseqs_out.insert(i,[arrayfire::Seq::new(dBstart as i32, dBend as i32, 1i32), arrayfire::Seq::default() ] );
        dCseqs_out.insert(i,[arrayfire::Seq::new(dCstart as i32, dCend as i32, 1i32), arrayfire::Seq::default() ] );
        dDseqs_out.insert(i,[arrayfire::Seq::new(dDstart as i32, dDend as i32, 1i32), arrayfire::Seq::default() ] );
        dEseqs_out.insert(i,[arrayfire::Seq::new(dEstart as i32, dEend as i32, 1i32), arrayfire::Seq::default() ] );




        combidxsel = arrayfire::constant::<i32>(0,arrayfire::Dim4::new(&[idxsel.dims()[0]*5 , 1,1,1]));
        arrayfire::assign_seq(&mut combidxsel, &[arrayfire::Seq::new(dAstart as i32, dAend as i32, 1i32)], &Aidxsel);
        arrayfire::assign_seq(&mut combidxsel, &[arrayfire::Seq::new(dBstart as i32, dBend as i32, 1i32)], &Bidxsel);
        arrayfire::assign_seq(&mut combidxsel, &[arrayfire::Seq::new(dCstart as i32, dCend as i32, 1i32)], &Cidxsel);
        arrayfire::assign_seq(&mut combidxsel, &[arrayfire::Seq::new(dDstart as i32, dDend as i32, 1i32)], &Didxsel);
        arrayfire::assign_seq(&mut combidxsel, &[arrayfire::Seq::new(dEstart as i32, dEend as i32, 1i32)], &Eidxsel);

        combidxsel_out.insert(i, combidxsel.clone());










        //Get indexes of WValues
        valsel = COO_batch_find(WRowIdxCOO,&idxsel, COO_batch_size);
        valsel_out.insert(i, valsel.clone());



        //Get rows of WRowIdx
        rvec = arrayfire::lookup(WRowIdxCOO, &valsel, 0);
        


        //Get cols
        cvec = arrayfire::lookup(WColIdx, &valsel, 0);
        cvec_out.insert(i, cvec.clone());


        //Find idx of dX
        dXsel = remap_rows(&rvec, &idxsel, neuron_size);
        dXsel_out.insert(i, dXsel);


        //Compute global index
        gidx1 = get_global_weight_idx2(
            neuron_size,
            &rvec,
            &cvec,
        );

        //Sort array
        let (_,idx) = arrayfire::sort_index(
            &gidx1, 
            0, 
            true
        );

        //Sparse value
        sparseval = arrayfire::lookup(&valsel, &idx, 0);

        //Sparse Col vector
        sparsecol = arrayfire::lookup(&rvec, &idx, 0);

        let mut temparr = arrayfire::constant::<i32>(0,arrayfire::Dim4::new(&[neuron_size,1,1,1]));

        let repeat_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    
        let mut counts = arrayfire::iota::<i32>(idxsel.dims(),repeat_dims);
    
        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&idxsel, 0, None);
        arrayfire::assign_gen(&mut temparr, &idxrs, &counts);

        sparsecol = arrayfire::lookup(&temparr, &sparsecol, 0);










        //Sparse Row
        sparserow = arrayfire::lookup(&cvec, &idx, 0);



        let ones = arrayfire::constant::<i32>(1,sparserow.dims());
        let  (_,mut sumarr) = arrayfire::sum_by_key(&sparserow, &ones, 0);


        nrows_out.insert(i, sumarr.dims()[0].clone());

        sparserow = arrayfire::accum(&sumarr, 0);


        let constarr = arrayfire::constant::<i32>(0,arrayfire::Dim4::new(&[1,1,1,1]));
        sparserow = arrayfire::join(0, &constarr, &sparserow);
        

        sparseval_out.insert(i, sparseval.clone());
        sparserow_out.insert(i, sparserow.clone());
        sparsecol_out.insert(i, sparsecol.clone());





        //Next idxsel
        newidxsel = find_unique(&cvec,neuron_size);

        //Add new Y error
        if (yslicidx > 0)
        {
            yslicidx = yslicidx - 1;

            newidxsel = arrayfire::join(0, &newidxsel, &output_idxsel);
        }


    }





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


    *Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
    *Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
    *Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
    *Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
    *Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
    *Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
    *Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];



}






















/*
Forward pass using CSR weighted adjacency sparse matrices and UAF. 
Generates all internal states and the neural network output


Inputs
netdata:             Neural Network Metadata
X:                   Input Matrix
WRowIdxCSR:          Row sparse matrix of the weighted adjacency matrix
WColIdx:             Column sparse matrix of the weighted adjacency matrix
Wseqs:               Indexes of the weight parameters
Hseqs:               Indexes of the bias parameters
Aseqs:               Indexes of the UAF A parameters
Bseqs:               Indexes of the UAF B parameters
Cseqs:               Indexes of the UAF C parameters
Dseqs:               Indexes of the UAF D parameters
Eseqs:               Indexes of the UAF E parameters
network_params:      All trainable parameters in the network


Outputs:
Z:                   Internal State Matrix Z
Q:                   Internal State Matrix Q

*/

pub fn state_space_forward_batch<Z: arrayfire::FloatingPoint<UnaryOutType = Z,AbsOutType = Z>   >(
    modeldata_int:  &HashMap<String, u64>,



    X: &arrayfire::Array<Z>,
    
    WRowIdxCSR: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,


    Wseqs: &[arrayfire::Seq<i32>; 1],
    Hseqs: &[arrayfire::Seq<i32>; 1],
    Aseqs: &[arrayfire::Seq<i32>; 1],
    Bseqs: &[arrayfire::Seq<i32>; 1],
    Cseqs: &[arrayfire::Seq<i32>; 1],
    Dseqs: &[arrayfire::Seq<i32>; 1],
    Eseqs: &[arrayfire::Seq<i32>; 1],
    network_params: &arrayfire::Array<Z>,



    Z: &mut arrayfire::Array<Z>,
    Q: &mut arrayfire::Array<Z>
) {

    let neuron_size: u64 = modeldata_int["neuron_size"].clone();
    let input_size: u64 = modeldata_int["input_size"].clone();
    let batch_size: u64 = modeldata_int["batch_size"].clone();

    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let ZERO = arrayfire::constant::<f64>(ZERO_F64,single_dims).cast::<Z>();
    



    let Zslices:i64 = Z.dims()[2] as i64;

    let X_slices:i64 = X.dims()[2] as i64;

    let S_dims = arrayfire::Dim4::new(&[neuron_size,batch_size,1,1]);
    //let mut S = arrayfire::constant::<f64>(0.0, S_dims);
    let mut S = arrayfire::tile(&ZERO, S_dims);


    let mut tempx =  arrayfire::slice(X, 0);
    let seqs = &[arrayfire::Seq::new(0.0f64, (input_size-1) as f64, 1.0f64),  arrayfire::Seq::default()];






    let H = arrayfire::index(&network_params, Hseqs);
    let A = arrayfire::index(&network_params, Aseqs);
    let B = arrayfire::index(&network_params, Bseqs);
    let C = arrayfire::index(&network_params, Cseqs);
    let D = arrayfire::index(&network_params, Dseqs);
    let E = arrayfire::index(&network_params, Eseqs);



    let WValues = arrayfire::index(network_params, Wseqs);


    let W = arrayfire::sparse::<Z>(
        neuron_size,
        neuron_size,
        &WValues,
        WRowIdxCSR,
        WColIdx,
        arrayfire::SparseFormat::CSR
    );
    

    for i in 0i64..Zslices
    {
        if X_slices > 1
        {
            tempx =  arrayfire::slice(X, i);
            arrayfire::assign_seq(&mut S, seqs, &tempx);
            drop(tempx);
        }
        else
        {
            arrayfire::assign_seq(&mut S, seqs, &X);
        }
        


        S = arrayfire::matmul(&W, &S, arrayfire::MatProp::NONE, arrayfire::MatProp::NONE);
        S = arrayfire::add(&S, &H, true);
        arrayfire::set_slice(Z, &S, i);

        S = UAF(&S,&A,&B,&C,&D,&E);
        arrayfire::set_slice(Q, &S, i);
    }


    
}





















/*
Backward pass using CSR weighted adjacency sparse matrices and UAF.
Generates the gradients of the sparse weighted adjacency matrix


Inputs
netdata:             Neural Network Metadata
X:                   Input Matrix
network_params:      All trainable parameters in the network
Z:                   Internal State matrix Z
Q:                   Internal State matrix Q
Y:                   Target Matrix to fit
loss_grad:           Loss function
neuron_idx:          Indexes of the neurons
idxsel_out:          Indexes of the UAF parameters
valsel_out:          Indexes of the UAF values
cvec_out:            Indexes of the UAF column sparse vector
dXsel_out:           Indexes of the dX values
nrows_out:           Number of rows in UAF
sparseval_out:       Indexes of the values in the sparse matrix
sparserow_out:       Indexes of the rows in the sparse matrix
sparsecol_out:       Indexes of the columns in the sparse matrix
Hidxsel_out:         Indexes of the bias vector H
Aidxsel_out:         Indexes of the UAF vector A
Bidxsel_out:         Indexes of the UAF vector B
Cidxsel_out:         Indexes of the UAF vector C
Didxsel_out:         Indexes of the UAF vector D
Eidxsel_out:         Indexes of the UAF vector E
combidxsel_out:      Indexes of the all vectors
dAseqs_out:          Indexes of the dA
dBseqs_out:          Indexes of the dB
dCseqs_out:          Indexes of the dC
dDseqs_out:          Indexes of the dD
dEseqs_out:          Indexes of the dE


Outputs:
grad:                   Gradient of all trainable parameters


*/




pub fn state_space_backward_group2<Z: arrayfire::FloatingPoint<UnaryOutType = Z,AbsOutType = Z,AggregateOutType=Z>  >(
    //netdata: &network_metadata_type,
    modeldata_int:  &HashMap<String, u64>,


    X: &arrayfire::Array<Z>,



    network_params: &arrayfire::Array<Z>,




    Z: &arrayfire::Array<Z>,
    Q: &arrayfire::Array<Z>,
    Y: &arrayfire::Array<Z>,
    loss_grad: impl Fn(&arrayfire::Array<Z>, &arrayfire::Array<Z>) -> arrayfire::Array<Z>,
    neuron_idx: &arrayfire::Array<i32>,



    idxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    valsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    
    cvec_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    dXsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    
    nrows_out: &nohash_hasher::IntMap<i64, u64 >,
    sparseval_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    sparserow_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    sparsecol_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,



    Hidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Aidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Bidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Cidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Didxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Eidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    combidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,




    dAseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,
    dBseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,
    dCseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,
    dDseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,
    dEseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,




    grad: &mut arrayfire::Array<Z>,
) {
    /* 
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();

    let batch_size: u64 = netdata.batch_size.clone();
    */


    let neuron_size: u64 = modeldata_int["neuron_size"].clone();
    let input_size: u64 = modeldata_int["input_size"].clone();
    let output_size: u64 = modeldata_int["output_size"].clone();
    let proc_num: u64 = modeldata_int["proc_num"].clone();
    //let active_size: u64 = modeldata_int["active_size"].clone();
    //let space_dims: u64 = modeldata_int["space_dims"].clone();
    //let step_num: u64 = modeldata_int["step_num"].clone();
    let batch_size: u64 = modeldata_int["batch_size"].clone();



    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let ZERO = arrayfire::constant::<f64>(ZERO_F64,single_dims).cast::<Z>();
    





    //Set output to zero
    //    *grad = arrayfire::constant::<f64>(0.0,network_params.dims());
    *grad = arrayfire::tile(&ZERO, network_params.dims());





    //Get current selection of neurons
    let active_size = neuron_idx.dims()[0];
    let idxsel = arrayfire::rows(neuron_idx, (active_size-output_size) as i64, (active_size-1)   as i64);


    let Qslices: u64 = Q.dims()[2];
    let Yslices: u64 = Y.dims()[2];


    //Get Yhat
    let mut idxrs = arrayfire::Indexer::default();
    let seq1 = arrayfire::Seq::new(0.0f64, (batch_size-1) as f64, 1.0);
    let seq2 = arrayfire::Seq::new((proc_num-1) as f64, (Qslices-1) as f64, 1.0);
    idxrs.set_index(&idxsel, 0, None);
    idxrs.set_index(&seq1, 1, None);
    idxrs.set_index(&seq2, 2, None);
    let Yhat = arrayfire::index_gen(Q, idxrs);
    drop(idxsel);

    let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let S_dims = arrayfire::Dim4::new(&[neuron_size,batch_size,1,1]);
    

    //Calculate error
    let total_error = loss_grad(&Yhat,Y);
    let mut yslicidx: i64 = (Yslices-1) as i64;
    let mut error = arrayfire::slice(&total_error, yslicidx);




    let Zslices: i64 = Z.dims()[2] as i64;

    let X_slices: i64 = X.dims()[2] as i64;

    



    //let mut inx = arrayfire::constant::<f64>(0.0,S_dims);
    let mut inx = arrayfire::tile(&ZERO, S_dims);




    let mut tempx =  arrayfire::slice(X, 0);


    let seqs = &[arrayfire::Seq::new(0.0f64, (input_size-1) as f64, 1.0f64),  arrayfire::Seq::default()];


    let mut Xtemp = arrayfire::slice(Z, 0);


    let mut sA = ZERO.clone();
    let mut sB = ZERO.clone();
    let mut sC = ZERO.clone();
    let mut sD = ZERO.clone();
    let mut sE = ZERO.clone();




    let mut dX = ZERO.clone();
    let mut dA = ZERO.clone();
    let mut dB = ZERO.clone();
    let mut dC = ZERO.clone();
    let mut dD = ZERO.clone();
    let mut dE = ZERO.clone();




    let mut tempW = ZERO.clone();


    
    let mut tempinx = ZERO.clone();



    let mut tempdX = ZERO.clone();
    let mut tempgW = ZERO.clone();
    let mut tempgH = ZERO.clone();




    let mut derror = arrayfire::slice(&total_error,  0);






    let batchseq = arrayfire::Seq::new(0.0f64, (batch_size-1) as f64, 1.0);
    let mut sliceseq = arrayfire::Seq::new((proc_num-1) as f64, (Qslices-1) as f64, 1.0);








    let mut UAFgroup = ZERO.clone();
    let mut tileerror = ZERO.clone();
    let tileerror_dims = arrayfire::Dim4::new(&[5,1,1,1]);



    //Main loop
    for i in (0i64..Zslices).rev() {




        //Select X value
        let mut idxrs = arrayfire::Indexer::default();
        sliceseq = arrayfire::Seq::new(i as f64, i as f64, 1.0);
        idxrs.set_index(&idxsel_out[&i], 0, None);
        idxrs.set_index(&batchseq, 1, None);
        idxrs.set_index(&sliceseq, 2, None);
        Xtemp = arrayfire::index_gen(Z, idxrs);


        //Get current UAF parameters
        sA = arrayfire::lookup(&network_params, &Aidxsel_out[&i], 0);

        sB = arrayfire::lookup(&network_params, &Bidxsel_out[&i], 0);

        sC = arrayfire::lookup(&network_params, &Cidxsel_out[&i], 0);

        sD = arrayfire::lookup(&network_params, &Didxsel_out[&i], 0);

        sE = arrayfire::lookup(&network_params, &Eidxsel_out[&i], 0);









        //Compute derivative of UAF
        deriUAF(&Xtemp,
            &sA,
            &sB,
            &sC,
            &sD,
            &sE,
            &mut dX,
            &mut dA,
            &mut dB,
            &mut dC,
            &mut dD,
            &mut dE);
        drop(Xtemp);
        drop(sA);
        drop(sB);
        drop(sC);
        drop(sD);
        drop(sE);

        //Compute dX
        dX = arrayfire::mul(&dX, &error, false);




        //Update H
        tempgH = arrayfire::lookup(grad, &Hidxsel_out[&i], 0) + (arrayfire::sum(&dX, 1) );


        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&Hidxsel_out[&i], 0, None);
        arrayfire::assign_gen(grad, &idxrs, &tempgH);
        drop(tempgH);



        //Join all
        
        //UAFgroup = arrayfire::constant::<f64>(0.0,arrayfire::Dim4::new(&[dA.dims()[0]*5 , dA.dims()[1],1,1]));
        UAFgroup = arrayfire::tile(&ZERO, arrayfire::Dim4::new(&[dA.dims()[0]*5 , dA.dims()[1],1,1]));

        arrayfire::assign_seq(&mut UAFgroup, &dAseqs_out[&i], &dA);
        arrayfire::assign_seq(&mut UAFgroup, &dBseqs_out[&i], &dB);
        arrayfire::assign_seq(&mut UAFgroup, &dCseqs_out[&i], &dC);
        arrayfire::assign_seq(&mut UAFgroup, &dDseqs_out[&i], &dD);
        arrayfire::assign_seq(&mut UAFgroup, &dEseqs_out[&i], &dE);



        tileerror =  arrayfire::tile(&error, tileerror_dims);

        UAFgroup = arrayfire::mul(&tileerror, &UAFgroup, false);
        drop(tileerror);

        UAFgroup = arrayfire::sum(&UAFgroup, 1) + arrayfire::lookup(grad, &combidxsel_out[&i],  0);

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&combidxsel_out[&i], 0, None);
        arrayfire::assign_gen(grad, &idxrs, &UAFgroup);
        drop(UAFgroup);










        //Get dX of each row
        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&dXsel_out[&i], 0, None);
        idxrs.set_index(&batchseq, 1, None);
        tempdX = arrayfire::index_gen(&dX, idxrs);
        












        //Get input values
        //inx = arrayfire::constant::<f64>(0.0,S_dims);
        inx = arrayfire::tile(&ZERO, S_dims);

        if (i > 0)
        {
            inx = arrayfire::slice(Q, (i-1) );
        }


        if X_slices > 1
        {
            tempx =  arrayfire::slice(X, i);
            arrayfire::assign_seq(&mut inx, seqs, &tempx);
            drop(tempx);
        }
        else
        {
            arrayfire::assign_seq(&mut inx, seqs, &X);
        }


        //Upadate gW
        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&cvec_out[&i], 0, None);
        idxrs.set_index(&batchseq, 1, None);
        tempinx = arrayfire::index_gen(&inx, idxrs);
        drop(inx);

        tempgW = arrayfire::mul(&tempdX, &tempinx, false);
        drop(tempinx);
        tempgW = (arrayfire::sum(&tempgW, 1) )+ arrayfire::lookup(grad, &valsel_out[&i], 0);



        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&valsel_out[&i], 0, None);
        arrayfire::assign_gen(grad, &idxrs, &tempgW);
        drop(tempgW);


        


        //Propagate Errors
        tempW = arrayfire::lookup(network_params, &sparseval_out[&i], 0);

        tempW = arrayfire::sparse(
            nrows_out[&i],
            dX.dims()[0],
            &tempW,
            &sparserow_out[&i],
            &sparsecol_out[&i],
            arrayfire::SparseFormat::CSR
        );
        
        
        error = arrayfire::matmul(&tempW,
            &dX,
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::NONE
        );
        
        drop(tempW);
        



        //Add new Y error
        if (yslicidx > 0)
        {
            yslicidx = yslicidx - 1;
            derror = arrayfire::slice(&total_error,  yslicidx);


            error = arrayfire::join(0, &error, &derror);
        }



}





}

























