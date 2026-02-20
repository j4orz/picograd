# Afterword

To continue deepening your knowledge on deep learning and it's systems,
the following resources are a good next step.
You might find this book complementary to your reading, since the three streams outlined below were woven into a single narrative for the book.
Once you feel comfortable, you should graduate towards contributing to larger deep learning systems.

Good luck on your journey.</br>
I'll see you at work.

## Futher Reading

### 1. Deep Learning

*Mathematics for Deep Learning*
##### Recommended Books
- Introduction to Probability for Data Science by Stanley Chan
- Introduction to Linear Algebra by Gilbert Strang
- Foundations of Linear Algebra for Data Science by Wanmo Kang and Kyunghyun Cho

##### Recommended Lectures
- Sanford CS109: Probability for Computer Scientists by Chris Piech
- UPenn STAT 4830: Numerical Optimization for Machine Learning by Damek Davis
- MIT 18.06 Linear Algebra by Gilbert Strang
- MIT 18.S096: Matrix Calculus by Alan Edelman and Steven Johnson

*Deep Learning*
##### Recommended Books
- Speech and Language Processing by Jurafsky and Martin
- The Elements of Statistical Learning by Friedman, Tibshirani, and Hastie
- Probabilistic Machine Learning by Kevin Murphy
- Deep Learning by Goodfellow, Bengio and Courville
- Reinforcement Learning by Sutton and Barto

##### Recommended Lectures
- Stanford CS124: From Languages to Information by Dan Jurafsky
- Stanford CS229: Machine Learning by Andrew Ng
- Stanford CS224N: NLP with Deep Learning by Christopher Manning
- Stanford CS336: Language Modeling from Scratch by Percy Liang
- Eureka LLM101N: Neural Networks Zero to Hero by Andrej Karpathy
- HuggingFace: Ultra-Scale Playbook: Training LLMs on GPU Clusters

### 2. Deep Learning Systems

*Performance Engineering for Deep Learning*
##### Recommended Books
- Computer Systems: A Programmer's Perspective by Randal Bryant and David O'Hallaron
- Computer Organization and Design RISC-V Edition by David A. Patterson and John L. Hennessy
- Computer Architecture: A Quantitative Approach by Hennessy and Patterson
- Programming Massively Parallel Processors by Hwu, Kirk, and Hajj
- The CUDA Handbook by Nicholas Wilt

##### Recommended Lectures
- CMU 414/714 Deep Learning Systems
- UTA Linear Algebra: Foundations to Frontiers by Robert van de Geijn and Margaret Meyers
- MIT 6.172: Performance Engineering by Saman Amarasinghe, Charles Leiserson and Julian Shun
- MIT 6.S894: Accelerated Computing by Jonathan Ragan-Kelley
- Stanford CS149: Parallel Computing by Kayvon Fatahalian
- Berkeley CS267: Applications of Parallel Computers by Katthie Yellick
- UIUC ECE408: Programming Massively Parallel Processors by Wen-mei Hwu

*Compiler Engineering for Deep Learning*
##### Recommended Books
- Machine Learning Compiler by Tianqi Chen


<!-- ### 4. Classical Compiler Construction

##### Recommended Books
- Programming Languages by Shriram Krishnamurthi
- Optimizing Compilers by Muchnick
- SSA book by Fabrice Rastello and Florent Bouchez Tichadou
- Register Allocation for Programs in SSA Form by Sebastian Hack
- Static Program Analysis by Anders Møller and Michael I Schwartzbach

##### Recommended Lectures
- Berkeley CS265: Compiler Optimization by Max Willsey
- Cornell CS4120: Compilers by Andrew Myers
- Cornell CS6120: Advanced Compilers by Adrian Sampson
- Carnegie Mellon 15-411: Compiler Design by Frank Pfenning
- Carnegie Mellon 15-745: Optimizing Compilers by Phil Gibbons
- Rice COMP412: Compiler Construction by Keith Cooper
- Rice COMP512: Advanced Compiler Construction by Keith Cooper -->

## Tinygrad Teenygrad Abstraction Correspondance

| Teenygrad | Tinygrad | Notes |
|-------------------|----------|-------|
| `OpNode` | `UOp` | Expression graph vertices |
| `OpCode` | `Ops` (enum) | Operation types |
| `Buffer` | `Buffer` | Device memory handles |
| `Runtime` | `Compiled` (Device class) | Memory + compute management |
| `Allocator` | `Allocator` | Buffer allocation/free |
| `Compiler` | `Compiler` | Source → binary compilation |
| `Generator` | `Renderer` | IR → source code generation |
| `Kernel` | `Program` (CPUProgram, CUDAProgram) | Executable kernel wrapper |