# RayBNN_Graph

Neural Networks with Sparse Weights in Rust using GPUs, CPUs, and FPGAs via CUDA, OpenCL, and oneAPI


# Install Arrayfire

Install the Arrayfire 3.9.0 binaries at [https://arrayfire.com/binaries/](https://arrayfire.com/binaries/)

or build from source
[https://github.com/arrayfire/arrayfire/wiki/Getting-ArrayFire](https://github.com/arrayfire/arrayfire/wiki/Getting-ArrayFire)




# Add to Cargo.toml
```
arrayfire = { version = "3.8.1", package = "arrayfire_fork" }
rayon = "1.7.0"
num = "0.4.1"
num-traits = "0.2.16"
half = { version = "2.3.1" , features = ["num-traits"] }
RayBNN_Sparse = "0.1.5"
RayBNN_DataLoader = "0.1.3"
RayBNN_Graph = "0.1.0"
RayBNN_Neural = "0.1.0"
```

# List of Examples


