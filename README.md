# Overview
This repository provides R Source codes to reproduce numerical experiments in the following arXiv preprint:

```
@article{okuno2023N3POM,
    year      = {2023},
    publisher = {CoRR},
    volume    = {},
    number    = {},
    pages     = {},
    author    = {Akifumi Okuno and Kazuharu Harada},
    title     = {An interpretable neural network-based non-proportional odds model for ordinal regression with continuous response},
    journal   = {arXiv preprint arXiv:2303.17823}
}
```

## Main scripts
### <a href="https://github.com/oknakfm/N3POM/blob/main/0-illustration.R">0-illustration.R</a>
You can train N3POM and plot the estimated coefficient functions (for one instance). Please replace the dataset if you want (default: real-estate).

### <a href="https://github.com/oknakfm/N3POM/blob/main/1-synthetic.R">1-synthetic.R</a>
This script computes the synthetic dataset experiments.

### <a href="https://github.com/oknakfm/N3POM/blob/main/2-real.R">2-real.R</a>
This script computes the real dataset experiments.

# Contact info.
Akifumi Okuno (okuno@ism.ac.jp)
