```
                                                                            ,,  
  mm                                                                      `7MM  
  MM                                                                        MM  
mmMMmm .gP"Ya   .gP"Ya `7MMpMMMb.`7M'   `MF'.P"Ybmmm `7Mb,od8 ,6"Yb.   ,M""bMM  
  MM  ,M'   Yb ,M'   Yb  MM    MM  VA   ,V :MI  I8     MM' "'8)   MM ,AP    MM  
  MM  8M"""""" 8M""""""  MM    MM   VA ,V   WmmmP"     MM     ,pm9MM 8MI    MM  
  MM  YM.    , YM.    ,  MM    MM    VVV   8M          MM    8M   MM `Mb    MM  
  `Mbmo`Mbmmd'  `Mbmmd'.JMML  JMML.  ,V     YMMMMMb  .JMML.  `Moo9^Yo.`Wbmd"MML.
                                    ,V     6'     dP                            
                                 OOb"      Ybmmmd'                              
```

*a teaching deep learning framework: the bridge from [micrograd](https://github.com/karpathy/micrograd) to [tinygrad](https://github.com/tinygrad/tinygrad)*

Take a whirlwind tour with the [SITP lectures and textbook](https://j4orz.ai/sitp/)
and build your own deep learning framework from scratch (a *pedagogically-omakase'd* tinygrad fork, sharing 90% of abstractions)
in order to run training and inference for [nanogpt](https://github.com/karpathy/nanoGPT).

**Installation**

`teenygrad` has a mixed source of Python, Rust, and CUDA Rust,
where the Python to Rust interop is implemented using CPython Extension Modules via [`PyO3`](https://pyo3.rs/),
with the shared object files compiled by driving `cargo` via PyO3's build tool [`maturin`](https://www.maturin.rs/).
Because the project uses [CUDA Rust](https://github.com/Rust-GPU/rust-cuda),
there is a specific version matrix required (notably an old version of LLVM),
and so `teenygrad` uses CUDA Rust's provided docker containers and shell scripts.

Before starting, please ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed on your machine.

```sh
> sudo nvidia-ctk runtime configure --runtime=docker # configure container runtime to docker
> sudo systemctl restart docker # restart docker
> ./dcr.sh # create the docker container with old version of llvm for cuda rust
> ./dex.sh "cd rust && maturin develop" # build the shared object for cpython's extension modules
> ./dex.sh "cd rust && cargo run" # run rust binary crate (cpu/gpu kernel hello world)
> ./dex.sh "uv run examples/abstractions.py" # run python
```
