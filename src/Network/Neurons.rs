

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







pub fn print_netdata(
    modeldata_string: &HashMap<String, String>,
    modeldata_float: &HashMap<String, String>,
    modeldata_int: &HashMap<String, String>,
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







