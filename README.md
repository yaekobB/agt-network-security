# Network Security Sets via Algorithmic Game Theory

> A Python framework for studying the **Network Security Set (NSS)** problem through four complementary Algorithmic Game Theory formulations.

## Overview

This repository implements the **Network Security Set (NSS)** problem from four perspectives:

1. **Strategic Games** – learning dynamics for decentralized security decisions.
2. **Coalitional Games** – Shapley-value based node importance and security-set construction.
3. **Planner Optimization** – vendor assignment under capacity constraints.
4. **Mechanism Design** – truthful secure-path procurement using VCG payments.

The project was developed for the **Algorithmic Game Theory** course at the **Department of Mathematics and Computer Science (DeMaCS), University of Calabria**. It includes an interactive Streamlit interface, command-line experiment runners, and reproducible evaluation.

---

## Features

- Streamlit visualization and inspection tools
- Command-line runners for reproducible experiments
- Support for Regular, Erdős–Rényi, and Barabási–Albert graphs
- Network Security Set (NSS) validation
- BRD, Regret Matching, and Fictitious Play
- Monte Carlo Shapley-value approximation
- Build–Prune algorithm for minimal NSS construction
- Planner assignment with Min-Cost Flow
- Truthful secure-path auction with VCG payments

---

## Project Structure

```text
app/            Streamlit application
experiments/    CLI experiment runners
src/            Core algorithms and utilities
outputs/        Generated results
docs/           Figures and project report
```

---

## Project Components

| Task | Summary |
|------|---------|
| **Task 1** | Strategic game using Best Response Dynamics, Regret Matching, and Fictitious Play. |
| **Task 2** | Coalitional game with characteristic functions, Monte Carlo Shapley values, and Build–Prune construction. |
| **Task 3** | Welfare-maximizing vendor assignment with infinite and limited capacities. |
| **Task 4** | Truthful secure-path auction using VCG payments. |

---

## Installation

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Running the Project

### Streamlit

```bash
streamlit run app/ui_streamlit.py
```

### CLI

```bash
python experiments/run_task1_cli.py
python experiments/run_task2_cli.py
python experiments/run_task12_compare_cli.py
```

---

## Experimental Evaluation

Experiments are conducted on Regular, Erdős–Rényi, and Barabási–Albert graphs and evaluate:

- convergence
- validity
- inclusion-wise minimality
- solution size
- runtime
- welfare
- VCG payments

---

## Reproducibility

Use fixed random seeds (default `42`) and identical graph parameters when comparing different methods.

---

## Outputs

Generated files are stored under `outputs/`, including experiment summaries, Shapley rankings, planner results, comparison tables, and VCG payment reports.

---

## Documentation

The full implementation details, methodology, and experimental results are available in:

`docs/report/AGT_Network_Security_Set_Report.pdf`

---

## License

MIT License

---

## Author

**Yaekob Beyene Yowhanns**  
M.Sc. Artificial Intelligence and Computer Science  
Department of Mathematics and Computer Science (DeMaCS)  
University of Calabria
