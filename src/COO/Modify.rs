use arrayfire;
use RayBNN_Sparse::Util::Search::COO_batch_find;
use RayBNN_Sparse::Util::Search::find_unique;


use RayBNN_Sparse::Util::Convert::get_global_weight_idx;


use crate::COO::Traversal::traverse_backward;

use nohash_hasher;
use rayon::prelude::*;


const COO_FIND_LIMIT: u64 = 1500000000;








pub fn delete_loops<Z: arrayfire::FloatingPoint>(
    last_idx: &arrayfire::Array<i32>,
    first_idx: &arrayfire::Array<i32>,
    neuron_size: u64,
    depth: u64,

    WValues: &mut arrayfire::Array<Z>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>
)
{
    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	


    let mut cur_idx = last_idx.clone();
    let mut cur_num = cur_idx.dims()[0] as i64;
    let mut filter_idx = arrayfire::join(0, &first_idx, &last_idx);



    let mut input_idx = arrayfire::constant::<i32>(0,single_dims);
    let mut temp_first_idx = arrayfire::constant::<i32>(0,single_dims);
    let mut detect_first_idx = arrayfire::constant::<i32>(0,single_dims);
    let mut next_idx = arrayfire::constant::<i32>(0,single_dims);
    let mut con_first_idx = arrayfire::constant::<i32>(0,single_dims);


    let mut delWRowIdxCOO = arrayfire::constant::<i32>(0,single_dims);
    let mut delWColIdx = arrayfire::constant::<i32>(0,single_dims);



    let mut COO_batch_size = 1 + ((COO_FIND_LIMIT/filter_idx.dims()[0]) as u64);




    let mut table = arrayfire::constant::<bool>(true,single_dims);

    let mut inarr = arrayfire::constant::<bool>(false, single_dims);

    let mut tempidx = arrayfire::locate(&table);



    for j in 0..depth
    {
        cur_num = cur_idx.dims()[0] as i64;


        if j == (depth-1)
        {
            filter_idx = arrayfire::rows(&filter_idx, first_idx.dims()[0] as i64, (filter_idx.dims()[0]-1) as i64 );
        }


        input_idx = arrayfire::row(&cur_idx, 0);

        traverse_backward(
            &input_idx,
            WRowIdxCOO,
            WColIdx,
            neuron_size,
            1,

            &mut temp_first_idx
        );

        if detect_first_idx.dims()[0] > 0
        {

            COO_batch_size = 1 + ((COO_FIND_LIMIT/temp_first_idx.dims()[0]) as u64);
            detect_first_idx = COO_batch_find( &temp_first_idx,&filter_idx, COO_batch_size);
        
            if detect_first_idx.dims()[0] > 0
            {
                con_first_idx = arrayfire::lookup(&temp_first_idx, &detect_first_idx, 0);

                input_idx = arrayfire::tile(&input_idx, con_first_idx.dims());

                delWRowIdxCOO = arrayfire::join(0, &delWRowIdxCOO, &input_idx);
                delWColIdx = arrayfire::join(0, &delWColIdx, &con_first_idx);





                table = arrayfire::constant::<bool>(true,temp_first_idx.dims());
                inarr = arrayfire::constant::<bool>(false, detect_first_idx.dims());

                let mut idxrs = arrayfire::Indexer::default();
                idxrs.set_index(&detect_first_idx, 0, None);
                arrayfire::assign_gen(&mut table, &idxrs, &inarr);
            
                tempidx = arrayfire::locate(&table);

                if (tempidx.dims()[0] > 0)
                {
                    temp_first_idx = arrayfire::lookup(&temp_first_idx, &tempidx, 0);
                }
                
            }
        }

        next_idx = temp_first_idx.clone();



        for i in 1..cur_num
        {
            input_idx = arrayfire::row(&cur_idx, i);

            traverse_backward(
                &input_idx,
                WRowIdxCOO,
                WColIdx,
                neuron_size,
                1,

                &mut temp_first_idx
            );

            if (temp_first_idx.dims()[0] == 0)
            {
                continue;
            }

            COO_batch_size = 1 + ((COO_FIND_LIMIT/temp_first_idx.dims()[0]) as u64);
            detect_first_idx = COO_batch_find( &temp_first_idx,&filter_idx, COO_batch_size);
        
            if detect_first_idx.dims()[0] > 0
            {
                con_first_idx = arrayfire::lookup(&temp_first_idx, &detect_first_idx, 0);

                input_idx = arrayfire::tile(&input_idx, con_first_idx.dims());

                delWRowIdxCOO = arrayfire::join(0, &delWRowIdxCOO, &input_idx);
                delWColIdx = arrayfire::join(0, &delWColIdx, &con_first_idx);





                table = arrayfire::constant::<bool>(true,temp_first_idx.dims());
                inarr = arrayfire::constant::<bool>(false, detect_first_idx.dims());

                let mut idxrs = arrayfire::Indexer::default();
                idxrs.set_index(&detect_first_idx, 0, None);
                arrayfire::assign_gen(&mut table, &idxrs, &inarr);
            
                tempidx = arrayfire::locate(&table);

                if (tempidx.dims()[0] == 0)
                {
                    continue;
                }

                temp_first_idx = arrayfire::lookup(&temp_first_idx, &tempidx, 0);

            }

            next_idx = arrayfire::join(0, &next_idx, &temp_first_idx);
            next_idx = find_unique(&next_idx, neuron_size);

        }
        cur_idx = next_idx.clone();

        filter_idx =  arrayfire::join(0, &next_idx, &filter_idx);
        filter_idx = find_unique(&filter_idx, neuron_size);


    }

    drop(cur_idx);
    drop(filter_idx);
    drop(input_idx);
    drop(temp_first_idx);
    drop(detect_first_idx);


    delWRowIdxCOO = arrayfire::rows(&delWRowIdxCOO, 1, (delWRowIdxCOO.dims()[0]-1) as i64 );
    delWColIdx = arrayfire::rows(&delWColIdx, 1, (delWColIdx.dims()[0]-1) as i64 );



	//Compute global index
	let gidx1 = get_global_weight_idx(
		neuron_size,
		WRowIdxCOO,
		WColIdx,
	);


	let gidx2 = get_global_weight_idx(
		neuron_size,
		&delWRowIdxCOO,
		&delWColIdx,
	);

    //TO CPU
	let mut gidx1_cpu = vec!(u64::default();gidx1.elements());
    gidx1.host(&mut gidx1_cpu);

	let mut gidx2_cpu = vec!(u64::default();gidx2.elements());
    gidx2.host(&mut gidx2_cpu);





	let mut WValues_cpu = vec!(Z::default();WValues.elements());
    WValues.host(&mut WValues_cpu);

	let mut WRowIdxCOO_cpu = vec!(i32::default();WRowIdxCOO.elements());
    WRowIdxCOO.host(&mut WRowIdxCOO_cpu);

	let mut WColIdx_cpu = vec!(i32::default();WColIdx.elements());
    WColIdx.host(&mut WColIdx_cpu);






	let mut join_WValues = nohash_hasher::IntMap::default();
	let mut join_WColIdx = nohash_hasher::IntMap::default();
	let mut join_WRowIdxCOO = nohash_hasher::IntMap::default();

    //Place old values
	for qq in 0..gidx1.elements()
	{
		let cur_gidx = gidx1_cpu[qq].clone();

		join_WValues.insert(cur_gidx, WValues_cpu[qq].clone());
		join_WColIdx.insert(cur_gidx, WColIdx_cpu[qq].clone());
		join_WRowIdxCOO.insert(cur_gidx, WRowIdxCOO_cpu[qq].clone());
	}

    //Remove values
    for qq in 0..gidx2.elements()
	{
		let cur_gidx = gidx2_cpu[qq].clone();

		join_WValues.remove(&cur_gidx);
		join_WColIdx.remove(&cur_gidx);
		join_WRowIdxCOO.remove(&cur_gidx);
	}

    let mut gidx3:Vec<u64> = join_WValues.clone().into_keys().collect();
	gidx3.par_sort_unstable();


	WValues_cpu = Vec::new();
	WRowIdxCOO_cpu = Vec::new();
	WColIdx_cpu = Vec::new();

	for qq in gidx3
	{
		WValues_cpu.push( join_WValues[&qq].clone() );
		WColIdx_cpu.push( join_WColIdx[&qq].clone() );
		WRowIdxCOO_cpu.push( join_WRowIdxCOO[&qq].clone() );
	}



	*WValues = arrayfire::Array::new(&WValues_cpu, arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]));
	*WColIdx = arrayfire::Array::new(&WColIdx_cpu, arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]));
	*WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]));
	



}

















