# Preface

As a compiler writer for domain specific cloud languages,
I became interested in compiler implementations for domain specific tensor languages
such as PyTorch 2 after the software 3.0 unlock of natural language programming from large language models such as ChatGPT.
However, I became frustrated with the *non-constructiveness* and *disjointedness* of
my learning experience in the discipline of machine learning systems —
the book that you are currently reading is my personal answer[^0] to these frustrations.

It's inspired by the introductory computer science canon created by Schemers. [SICP](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/index.html) and it's [dual](https://cs.brown.edu/~sk/Publications/Papers/Published/fffk-htdp-vs-sicp-journal/paper.pdf), [HTDP](https://htdp.org/), took you from counting to compilers in an unbroken, logical sequence, which, although has an informal [flânnerie](https://cs.uwaterloo.ca/~plragde/flaneries/) like feel, provides strong foundational edifice for computation. The recent addition of [DCIC](https://dcic-world.org/), spawning from it's phylogonetic cousin [PAPL](https://papl.cs.brown.edu/2020/), was created to cater to the recent [shift in data science](https://cs.brown.edu/~sk/Publications/Papers/Published/kf-data-centric/paper.pdf) with the *table* data structure. This book follows suit, (*aspirationally* titled SITP),
and is an experimental CS2 for software 2.0 which places a heavier focus on
the statistical inference, numerical linear algebra, low level and parallel systems programming required for deep learning, taking the reader from counting, to compilers *for* calculus.
And for now, the book you are reading is still higher quality than prompting a state of the art LLM to *"write me the SICP for software 2.0"*.

If you empathize with some of my frustrations, you may benefit from the course too[^1].</br>
Good luck on your journey.</br>
Are you ready to begin?</br>

---
[^0]: *I am writing this book with myself as the target audience in which I construct my own truth. For a good critique on constructionist learning however, refer to [(Ames 2018)](https://dl.acm.org/doi/epdf/10.1145/3274287).*
[^1]: *And if not, I hope this book poses as a good counterexample for what you have in mind — for instance, perhaps Kevin Murphy's excellent [encyclopedic treatment](https://probml.github.io/pml-book/) of the subject*