# nesy-suffix-prediction-dfa

**Neurosymbolic suffix prediction using Deterministic Finite Automata (DFA).**

*Axel Mezini, Elena Umili, Ivan Donadello, Fabrizio Maria Maggi, Matteo Mancanelli, Fabio Patrizi*

This repository contains the source code accompanying the paper: [Neuro-Symbolic Predictive Process Monitoring](https://authors.elsevier.com/c/1nGKE15hGZkPXv). 
The implementation provides the experimental framework used in the paper, including model execution, DFA-based symbolic components, and evaluation pipelines.

---

## Overview

This project implements a neurosymbolic approach to suffix prediction that integrates neural sequence modeling with symbolic reasoning via deterministic finite automata (DFA).

The repository is intended primarily for:

- Reproducing the experiments reported in the paper  
- Running the provided suffix prediction pipeline  
- Extending the framework for further research  

For theoretical details, methodology, and evaluation discussion, please refer to the paper.

---

## Project Structure

```
nesy-suffix-prediction-dfa/
│
├── runner.py              # Main entry point
├── requirements.txt       # Python dependencies
├── config_example.yaml/   # Configuration template
├── src/                   # Core implementation
├── data/                  # Datasets
├── results/               # Results
├── README.md
├── .gitignore
└── LICENCE
```

---

## Prerequisites

- Python x
- CUDA
- [Mona](https://www.brics.dk/mona/)

---

## Setup

1. Clone the Repository
    ```bash
    git clone https://github.com/axelmezini/nesy-suffix-prediction-dfa.git
    cd nesy-suffix-prediction-dfa
    ```

2. Install dependencies
    ```bash
    conda env create -f environment.yml
    ```

3. Create configuration file
    ```bash
    cp config_example.yaml config.yaml
    ```
    Edit `config.yaml` to set output directories

---

## Running the Code

To reproduce a specific experiment:
1. Ensure MONA is correctly installed.
2. Ensure the correct dataset is available in the `data/` directory.
3. Set experiment parameters in `config.yaml` to match the paper setup. For exact parameter settings, refer to the experimental section of the paper.
4. Activate the environment:
    ```bash
    conda activate myenv
    ```
5. Run:
    ```bash
    python -m src/runner.py
    ```

Results will be written to the configured output directory.

---

## Citation

If you use this code in your research, please cite:
```bibtex
@article{MEZINI2026102762,
    title = {Neuro-Symbolic Predictive Process Monitoring},
    journal = {Information Systems},
    volume = {141},
    pages = {102762},
    year = {2026},
    issn = {0306-4379},
    doi = {https://doi.org/10.1016/j.is.2026.102762},
    url = {https://www.sciencedirect.com/science/article/pii/S0306437926000761},
    author = {Axel Mezini and Elena Umili and Ivan Donadello and Fabrizio Maria Maggi and Matteo Mancanelli and Fabio Patrizi},
    keywords = {Suffix prediction, Neuro-Symbolic AI, Deep learning with logical knowledge, Linear Temporal Logic, Differentiable automata},
    abstract = {This paper addresses the problem of suffix prediction in Business Process Management (BPM) by proposing a Neuro-Symbolic Predictive Process Monitoring (PPM) approach that integrates data-driven learning with temporal logic-based prior knowledge. While recent approaches leverage deep learning models for suffix prediction, they often fail to satisfy even basic logical constraints due to the lack of explicit integration of domain knowledge during training. We propose a novel method to incorporate Linear Temporal Logic over finite traces (LTLf ) into the training process of autoregressive sequence predictors. Our approach introduces a differentiable logical loss function, defined using a soft approximation of LTLf semantics and the Gumbel-Softmax trick, which can be combined with standard predictive losses. This ensures that the model learns to generate suffixes that are both accurate and logically consistent. Experimental evaluation on three real-world datasets shows that our method improves suffix prediction accuracy and compliance with temporal constraints. We also introduce two variants of the logic loss (local and global) and demonstrate their effectiveness under noisy and realistic settings. While developed in the context of BPM, our framework is applicable to any symbolic sequence generation task and contributes to advancing Neuro-Symbolic AI.}
}
```


## License

This project is licensed under the MIT License.  
See the `LICENSE` file for details.
