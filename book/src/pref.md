# Preface

As a compiler writer for domain specific cloud languages (Terraform HCL),
I became interested in compiler implementations for domain specific tensor languages
such as PyTorch 2 after the software 3.0 unlock of natural language programming from large language models such as ChatGPT.
However, I became frustrated with the *non-constructiveness* and *disjointedness* of
my learning experience in the discipline of machine learning systems —
the book that you are currently reading is my personal answer to these frustrations.

It's inspired by the introductory computer science canon created by Schemers,
which consists of two books secretly masquerading as one.
[SICP](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/index.html) and it's [dual](https://cs.brown.edu/~sk/Publications/Papers/Published/fffk-htdp-vs-sicp-journal/paper.pdf), [HTDP](https://htdp.org/), teach programming
and programming languages by taking readers through an unbroken logical sequence in a [flânnerie](https://cs.uwaterloo.ca/~plragde/flaneries/)-like style.
The recent addition of [DCIC](https://dcic-world.org/), spawning from it's phylogenetic cousin [PAPL](https://papl.cs.brown.edu/2020/), was created to adjust the curriculum to the recent [shift in data science](https://cs.brown.edu/~sk/Publications/Papers/Published/kf-data-centric/paper.pdf) by emphasizing the *tabular/table* data structure.
This book follows suit, (*aspirationally* titled SITP), and is an experimental CS2 for software 2.0 whose only prereqisite is a familiarity of CS1 programming from, well, software 1.0.

If you are more experienced, you may benefit in jumping straight to part three of the book
which develops a "graph mode" fusion compiler and inference engine with tinygrad's RISC-y IR,
which borrows ideas from ThunderKitten's tile registers, MegaKernels, and Halide/TVM schedules.
Beyond helping those like myself interested in the systems of deep learning,
developing the low level performance primitives of deep neural networks will shed light on the open research question of
how domain specific tensor languages of deep learning frameworks can best support the development and compilation of accelerated kernels for novel network architectures (inductive biases) beyond the attention mechanism of transformers?
In the [afterword](./after.md) there is a table which maps out the correspondance of the tinygrad subset that exists in teenygrad.

<!-- The SITP book takes readers from training models to developing their own deep learning frameworks.
This book has been creatively handwritten for humans, and for now, achieves such a goal better
than prompting a state of the art LLM to *"write me the SICP for software 2.0"*
— the number of books that comprise this style is not enough for the sample efficiency of today's SOTA LLMs.

understanding.
Agents, what a system should platonically do.
link to tinygrad devlog. -->

If you empathize with some of my frustrations, you may benefit from the book too[^0].</br>
Good luck on your journey.</br>
Are you ready to begin?</br>

---
[^0]: *And if not, I hope this book poses as a good counterexample for what you have in mind — for instance, perhaps Kevin Murphy's excellent [formal encyclopedic treatment](https://probml.github.io/pml-book/) of the subject*