# Afterword

To continue deepening your knowledge, the following courses are a good next step.
You might find this book complementary to your reading, since the three streams outlined below were woven into a single narrative for the book.
Once you feel comfortable, you should graduate towards contributing to larger deep learning systems.

Good luck on your journey.</br>
I'll see you at work.

## Recommend Resources

### 1. Mathematics for Deep Learning
##### Recommended Books
- Introduction to Probability for Data Science by Stanley Chan
- Introduction to Linear Algebra by Gilbert Strang
- Foundations of Linear Algebra for Data Science by Wanmo Kang and Kyunghyun Cho
- Numerical Linear Algebra by Nick Trefethen and David Bau III
- Numerical Linear Algebra by Yuji Nakatsukasa
- Numerical Linear Algebra by Eric Darve

##### Recommended Lectures
- Sanford CS109: Probability for Computer Scientists by Chris Piech
- MIT 18.06 Linear Algebra by Gilbert Strang
- MIT 18.S096: Matrix Calculus by Alan Edelman and Steven Johnson
- UPenn STAT 4830: Numerical Optimization for Machine Learning by Damek Davis

### 2. Deep Learning

##### Recommended Books
- Speech and Language Processing by Jurafsky and Martin
- The Elements of Statistical Learning by Friedman, Tibshirani, and Hastie
- Deep Learning by Goodfellow, Bengio and Courville
- Reinforcement Learning by Sutton and Barto
- Probabilistic Machine Learning by Kevin Murphy

##### Recommended Lectures
- Stanford CS124: From Languages to Information by Dan Jurafsky
- Stanford CS229: Machine Learning by Andrew Ng
- Stanford CS230: Deep Learning by Andrew Ng
- Stanford CS224N: NLP with Deep Learning by Christopher Manning
- Eureka LLM101N: Neural Networks Zero to Hero by Andrej Karpathy
- Stanford CS336: Language Modeling from Scratch by Percy Liang
- HuggingFace: Ultra-Scale Playbook: Training LLMs on GPU Clusters

### 3. Performance Engineering for Deep Learning

##### Recommended Books
- Computer Architecture: A Quantitative Approach by Hennessy and Patterson
- Computer Systems: A Programmer's Perspective by Randal Bryant and David O'Hallaron
- Performance Analysis And Tuning on Modern CPUs by Denis Bakhvalov
- Optimization Manuals by Agner Fog
- Programming Massively Parallel Processors by Hwu, Kirk, and Hajj
- The CUDA Handbook by Nicholas Wilt

##### Recommended Lectures
- MIT 6.172: Performance Engineering by Saman Amarasinghe, Charles Leiserson and Julian Shun
- MIT 6.S894: Accelerated Computing by Jonathan Ragan-Kelley
- Berkeley CS267: Applications of Parallel Computers by Katthie Yellick
- Berkeley EECS151: Introduction to Digital Design and Integrated Circuits by Sophia Shao
- Berkeley EECS152: Computer Architecture by Sophia Shao
- UIUC ECE408: Programming Massively Parallel Processors by Wen-mei Hwu
- Stanford CS149: Parallel Computing by Kayvon Fatahalian

### 4. Compiler Engineering for Deep Learning

##### Recommended Books
Programming Languages by Shriram Krishnamurthi
Optimizing Compilers by Muchnick
SSA book by Fabrice Rastello and Florent Bouchez Tichadou
Register Allocation for Programs in SSA Form by Sebastian Hack
Static Program Analysis by Anders Møller and Michael I Schwartzbach 

##### Recommended Lectures
- Berkeley CS265: Compiler Optimization by Max Willsey
- Cornell CS4120: Compilers by Andrew Myers
- Cornell CS6120: Advanced Compilers by Adrian Sampson
- Carnegie Mellon 15-411: Compiler Design by Frank Pfenning
- Carnegie Mellon 15-745: Optimizing Compilers by Phil Gibbons
- Rice COMP412: Compiler Construction by Keith Cooper
- Rice COMP512: Advanced Compiler Construction by Keith Cooper

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