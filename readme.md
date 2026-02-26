# nesy-suffix-prediction-dfa

**Neurosymbolic suffix prediction using Deterministic Finite Automata (DFA).**

*Axel Mezini, Elena Umili, Ivan Donadello, Fabrizio Maria Maggi, Matteo Mancanelli, Fabio Patrizi*

This repository contains the source code accompanying the paper: [Neuro-Symbolic Predictive Process Monitoring](https://arxiv.org/abs/2509.00834). 
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
├── config.yaml/           # Configuration template
├── src/                   # Core implementation
├── data/                  # Datasets
├── README.md
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
    pip install -r requirements.txt
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
4. Run:
    ```bash
      python -m runner.py
    ```

Results will be written to the configured output directory.

---

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{mezini2025neurosymbolicpredictiveprocessmonitoring,
      title={Neuro-Symbolic Predictive Process Monitoring}, 
      author={Axel Mezini and Elena Umili and Ivan Donadello and Fabrizio Maria Maggi and Matteo Mancanelli and Fabio Patrizi},
      year={2025},
      eprint={2509.00834},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.00834}, 
}
```


## License

This project is licensed under the MIT License.  
See the `LICENSE` file for details.