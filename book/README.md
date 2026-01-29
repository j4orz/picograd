```
 .M"""bgd `7MMF'MMP""MM""YMM `7MM"""Mq. 
,MI    "Y   MM  P'   MM   `7   MM   `MM.
`MMb.       MM       MM        MM   ,M9 
  `YMMNq.   MM       MM        MMmmdM9  
.     `MM   MM       MM        MM       
Mb     dM   MM       MM        MM       
P"Ybmmd"  .JMML.   .JMML.    .JMML.     
```

SITP uses Rust's [mdbook](https://rust-lang.github.io/mdBook/) infrastructure.

**Installation**
1. Globally install the following [mdbook preprocessors](https://rust-lang.github.io/mdBook/for_developers/preprocessors.html)
(binary crates) with [`cargo install`](https://doc.rust-lang.org/cargo/commands/cargo-install.html) in your shell:
    ```sh
    cargo install mdbook --version 0.4.52
    cargo install mdbook-katex mdbook-repl mdbook-embedify mdbook-svgbob
    ```
2. Run mdbook's build server at http://localhost:3000/, run the following:
    ```sh
    mdbook serve
    ```