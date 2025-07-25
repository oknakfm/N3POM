# Overview
This repository provides R Source codes to reproduce numerical experiments in the following manuscript:

```
@article{okuno2024N3POM,
    year      = {2024},
    publisher = {Taylor \& Francis},
    volume    = {33},
    number    = {},
    issue     = {4},
    pages     = {1454-1463},
    author    = {Akifumi Okuno and Kazuharu Harada},
    title     = {An Interpretable Neural Network-based Nonproportional Odds Model for Ordinal Regression},
    journal   = {Journal of Computational and Graphical Statistics}
}
```

## Main scripts
### <a href="https://github.com/oknakfm/N3POM/blob/main/0-demonstration.R">0-demonstration.R</a>
You can train N3POM and plot the estimated coefficient functions (for one instance). Please replace the dataset if you want (default: real-estate). 
Our computer, equipped with an AMD Ryzen 7 5700X 8-Core Processor operating at 3.4GHz and 32 GB of RAM, took approximately 75 seconds to train the N3POM in this script.

### <a href="https://github.com/oknakfm/N3POM/blob/main/1-synthetic.R">1-synthetic.R</a>
This script computes the synthetic dataset experiments.

### <a href="https://github.com/oknakfm/N3POM/blob/main/2-real.R">2-real.R</a>
This script computes the real dataset experiments.

# Contact info.
Akifumi Okuno (okuno@ism.ac.jp)
