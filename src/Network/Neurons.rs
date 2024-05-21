

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

pub fn state_space_forward_batch(
    netdata: &network_metadata_type,
    X: &arrayfire::Array<f64>,
    
    WRowIdxCSR: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,


    Wseqs: &[arrayfire::Seq<i32>; 1],
    Hseqs: &[arrayfire::Seq<i32>; 1],
    Aseqs: &[arrayfire::Seq<i32>; 1],
    Bseqs: &[arrayfire::Seq<i32>; 1],
    Cseqs: &[arrayfire::Seq<i32>; 1],
    Dseqs: &[arrayfire::Seq<i32>; 1],
    Eseqs: &[arrayfire::Seq<i32>; 1],
    network_params: &arrayfire::Array<f64>,



    Z: &mut arrayfire::Array<f64>,
    Q: &mut arrayfire::Array<f64>
) {

    let neuron_size: u64 = netdata.neuron_size.clone();
    //let proc_num: u64 = netdata.proc_num.clone();
    let input_size: u64 = netdata.input_size.clone();
    let batch_size: u64 = netdata.batch_size.clone();

    let Zslices:i64 = Z.dims()[2] as i64;

    let X_slices:i64 = X.dims()[2] as i64;

    let S_dims = arrayfire::Dim4::new(&[neuron_size,batch_size,1,1]);
    let mut S = arrayfire::constant::<f64>(0.0, S_dims);


    let mut tempx =  arrayfire::slice(X, 0);
    let seqs = &[arrayfire::Seq::new(0.0f64, (input_size-1) as f64, 1.0f64),  arrayfire::Seq::default()];






    let H = arrayfire::index(&network_params, Hseqs);
    let A = arrayfire::index(&network_params, Aseqs);
    let B = arrayfire::index(&network_params, Bseqs);
    let C = arrayfire::index(&network_params, Cseqs);
    let D = arrayfire::index(&network_params, Dseqs);
    let E = arrayfire::index(&network_params, Eseqs);



    let WValues = arrayfire::index(network_params, Wseqs);


    let W = arrayfire::sparse::<f64>(
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

pub fn state_space_backward_group2(
    netdata: &network_metadata_type,
    X: &arrayfire::Array<f64>,



    network_params: &arrayfire::Array<f64>,




    Z: &arrayfire::Array<f64>,
    Q: &arrayfire::Array<f64>,
    Y: &arrayfire::Array<f64>,
    loss_grad: impl Fn(&arrayfire::Array<f64>, &arrayfire::Array<f64>) -> arrayfire::Array<f64>,
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




    grad: &mut arrayfire::Array<f64>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();



    let batch_size: u64 = netdata.batch_size.clone();



    //Set output to zero
    *grad = arrayfire::constant::<f64>(0.0,network_params.dims());








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

    



    let mut inx = arrayfire::constant::<f64>(0.0,S_dims);

    let mut tempx =  arrayfire::slice(X, 0);


    let seqs = &[arrayfire::Seq::new(0.0f64, (input_size-1) as f64, 1.0f64),  arrayfire::Seq::default()];


    let mut Xtemp = arrayfire::slice(Z, 0);


    let mut sA = arrayfire::constant::<f64>(0.0,temp_dims);
    let mut sB = arrayfire::constant::<f64>(0.0,temp_dims);
    let mut sC = arrayfire::constant::<f64>(0.0,temp_dims);
    let mut sD = arrayfire::constant::<f64>(0.0,temp_dims);
    let mut sE = arrayfire::constant::<f64>(0.0,temp_dims);




    let mut dX = arrayfire::constant::<f64>(0.0,temp_dims);
    let mut dA = arrayfire::constant::<f64>(0.0,temp_dims);
    let mut dB = arrayfire::constant::<f64>(0.0,temp_dims);
    let mut dC = arrayfire::constant::<f64>(0.0,temp_dims);
    let mut dD = arrayfire::constant::<f64>(0.0,temp_dims);
    let mut dE = arrayfire::constant::<f64>(0.0,temp_dims);




    let mut tempW = arrayfire::constant::<f64>(0.0,temp_dims);


    let mut gtemperr = arrayfire::constant::<f64>(0.0,temp_dims);
    let mut tempinx = arrayfire::constant::<f64>(0.0,temp_dims);



    let mut tempdX = arrayfire::constant::<f64>(0.0,temp_dims);
    let mut tempgW = arrayfire::constant::<f64>(0.0,temp_dims);
    let mut tempgH = arrayfire::constant::<f64>(0.0,temp_dims);




    let mut derror = arrayfire::slice(&total_error,  0);






    let batchseq = arrayfire::Seq::new(0.0f64, (batch_size-1) as f64, 1.0);
    let mut sliceseq = arrayfire::Seq::new((proc_num-1) as f64, (Qslices-1) as f64, 1.0);




    let mut keys = arrayfire::constant::<i32>(0,temp_dims);
    let mut vals = arrayfire::constant::<f64>(0.0,temp_dims);




    let mut UAFgroup = arrayfire::constant::<f64>(0.0,temp_dims);
    let mut tileerror = arrayfire::constant::<f64>(0.0,temp_dims);
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

        UAFgroup = arrayfire::constant::<f64>(0.0,arrayfire::Dim4::new(&[dA.dims()[0]*5 , dA.dims()[1],1,1]));
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
        inx = arrayfire::constant::<f64>(0.0,S_dims);

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

        tempW = arrayfire::sparse::<f64>(
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

























