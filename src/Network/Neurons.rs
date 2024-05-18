

use arrayfire;



use std::collections::HashMap;
use nohash_hasher;

use crate::Network::Activation::UAF;
use crate::Network::Activation::deriUAF;


use crate::graph::large_sparse_i32::COO_batch_find;

use crate::graph::tree_i32::find_unique;


const COO_find_limit: u64 = 1500000000;



const ZERO_F64: f64 = 0.0;
const ONE_F64: f64 = 1.0;
const HIGH_F64: f64 = 1000000.0;
const TWO_F64: f64 = 2.0;
const ONEHALF_F64: f64 = 0.5f64;
















