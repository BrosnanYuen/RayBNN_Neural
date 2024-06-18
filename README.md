# RayBNN_Neural

Neural Networks with Sparse Weights in Rust using GPUs, CPUs, and FPGAs via CUDA, OpenCL, and oneAPI


# Install Arrayfire

Install the Arrayfire 3.9.0 binaries at [https://arrayfire.com/binaries/](https://arrayfire.com/binaries/)

or build from source
[https://github.com/arrayfire/arrayfire/wiki/Getting-ArrayFire](https://github.com/arrayfire/arrayfire/wiki/Getting-ArrayFire)




# Add to Cargo.toml
```
arrayfire = { version = "3.8.1", package = "arrayfire_fork" }
rayon = "1.10.0"
num = "0.4.3"
num-traits = "0.2.19"
half = { version = "2.4.1" , features = ["num-traits"] }
RayBNN_DataLoader = "2.0.3"
RayBNN_Graph = "2.0.3"
RayBNN_Sparse = "2.0.2"
RayBNN_Optimizer = "2.0.1"
RayBNN_Neural = "2.0.3"
```

# List of Examples


