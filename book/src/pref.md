# Preface

As a compiler writer for domain specific cloud languages,
I became interested in compiler implementations for domain specific tensor languages
(such as PyTorch 2) after the "software 3.0" unlock (programming with natural language) from
large language models like ChatGPT.
However, I became frustrated with the *non-constructiveness* and *disjointedness* of
my learning experience in the discipline of machine learning systems —
the book that you are currently reading is my personal answer[^0] to these frustrations. It
1. is inspired by the introductory computer science canon created by Schemers. [SICP](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/index.html) and it's [dual](https://cs.brown.edu/~sk/Publications/Papers/Published/fffk-htdp-vs-sicp-journal/paper.pdf), [HTDP](https://htdp.org/), took you from counting to compilers in an unbroken, logical sequence, which, although has an informal [flânnerie](https://cs.uwaterloo.ca/~plragde/flaneries/) like feel, provides strong foundational edifice for computation. The recent addition of [DCIC](https://dcic-world.org/), spawning from it's phylogonetic cousin [PAPL](https://papl.cs.brown.edu/2020/), was created to cater to the recent [shift in data science](https://cs.brown.edu/~sk/Publications/Papers/Published/kf-data-centric/paper.pdf) with the *table* data structure. This book follows suit, (*aspirationally* titled SITP),
and is an experimental CS2 for "software 2.0" which places a heavier focus on
the statistical inference, numerical linear algebra, low level and parallel systems programming required for deep learning,
taking the reader from counting, to compilers *for* calculus.
2. concerns itself with the low-level programming of deep learning systems. So the capstone project `teenygrad` involves programming against language, platform, and architecture specifics with a `Python` core for *user productivty*, and CPU/GPU accelerated kernels implemented with `Rust` and `CUDA Rust` for it's amenability towards *native acceleration*. However, you are more than welcome to follow along with your own choice of host language implementation — for instance, feel free to swap out `Python` for `Javascript`[^1]/`Lua`, `Rust` for `C++`, etc[^2].

With that said, the book has three parts:
- in [part 1](./1.md) you implement a multidimensional `Tensor` and accelerated `BLAS` kernels
- in [part 2](./2.md) you implement `.backward()` and accelerated `cuBLAS` kernels for the *age of research*
- in [part 3](./3.md) you implement a fusion compiler with `OpNode` graph IR for the *age of scaling*

If you empathize with some of my frustrations, you may benefit from the course too[^3].</br>
Good luck on your journey.</br>
Are you ready to begin?</br>

---
[^0]: *As of the time writing in 2026, I am writing this book with myself as the target audience. For a good critique on constructionist learning, refer to [(Ames 2018)](https://dl.acm.org/doi/epdf/10.1145/3274287).*
[^1]: *For instance, Andrej Karpathy's [convnetjs](https://github.com/karpathy/convnetjs)*
[^2]: *Or combine the productivity and performance with `Mojo` for instance. The world is your oyster.*
[^3]: *And if not, I hope this book poses as a good counterexample for what you have in mind — for instance, perhaps Kevin Murphy's excellent [encyclopedic treatment](https://probml.github.io/pml-book/) of the subject*