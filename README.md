```
                                                                            ,,  
  mm         Rest in Pure Land Dr. Thomas Zhang ND., R.TCMP, R.Ac.        `7MM  
  MM                                                                        MM  
mmMMmm .gP"Ya   .gP"Ya `7MMpMMMb.`7M'   `MF'.P"Ybmmm `7Mb,od8 ,6"Yb.   ,M""bMM  
  MM  ,M'   Yb ,M'   Yb  MM    MM  VA   ,V :MI  I8     MM' "'8)   MM ,AP    MM  
  MM  8M"""""" 8M""""""  MM    MM   VA ,V   WmmmP"     MM     ,pm9MM 8MI    MM  
  MM  YM.    , YM.    ,  MM    MM    VVV   8M          MM    8M   MM `Mb    MM  
  `Mbmo`Mbmmd'  `Mbmmd'.JMML  JMML.  ,V     YMMMMMb  .JMML.  `Moo9^Yo.`Wbmd"MML.
                                    ,V     6'     dP                            
                                 OOb"      Ybmmmd'                              
```


The teaching deep learning framework for the [SITP](https://book.j4orz.ai) textbook.</br>
*train [nanogpt](https://github.com/karpathy/nanoGPT) by building teenygrad in Python and Rust — the bridge from [micrograd](https://github.com/karpathy/micrograd) to [tinygrad](https://github.com/tinygrad/tinygrad)*

---

# Installation

## Graph Mode
`teenygrad` graph mode (developed in part [3](https://book.j4orz.ai/3.html) of the book) is a pure Python Tensor compiler.

## Eager Mode
`teenygrad` eager mode (developed in part [1](https://book.j4orz.ai/1.html) and [2](https://book.j4orz.ai/2.html) of the book) has a mixed source of Python, Rust, and CUDA Rust
in order to support CPU and GPU acceleration.
The Python to Rust interop is implemented using CPython Extension Modules via [`PyO3`](https://pyo3.rs/),
with the shared object files compiled by driving `cargo` via PyO3's build tool [`maturin`](https://www.maturin.rs/).

**CPU kernels (x86/ARM)**
1. CPU kernels do not use the docker container (for now).
    ```sh
    uv pip install maturin                             # install maturin (which drives pyo3)
    cd rust && cargo run                               # run cpu acccelerated gemm kernel
    maturin develop                                    # build shared object for cpython's extension modules
    uv run examples/abstractions.py                    # run cpu accelerated gemm kernel from python
    ```
**GPU kernels (PTX)**

To enable GPU acceleration, teenygrad uses [CUDA Rust](https://github.com/Rust-GPU/rust-cuda), which in turn requires a specific version matrix required (notably an old version of LLVM) and so CUDA Rust's provided docker containers and shell scripts are used.
1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on your machine
2. Run the following in your shell:
    ```sh
    sudo nvidia-ctk runtime configure --runtime=docker # configure container runtime to docker
    sudo systemctl restart docker                      # restart docker
    ./dcr.sh                                           # create container with old version of llvm for cuda rust
    ./dex.sh "cd rust && cargo run --features cuda"    # run gpu accelerated gemm kernel
    ./dex.sh "maturin develop"                         # build the shared object for cpython's extension modules
    ./dex.sh "uv run examples/abstractions.py"         # run gpu accelerated gemm kernel from python
    ```
3. Point `rustanalyzer` to the Rust and CUDA Rust source:
    ```json
    {
      <!-- other fields in settings.json -->
      "rust-analyzer.linkedProjects": ["rust/Cargo.toml"]
    }
    ```