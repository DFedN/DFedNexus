# Asynchronous Decentralized Federated Learning over Lossy Wireless Links via Reception- and Age-Aware Aggregation

Welcome to the **DFL-AA Simulator** codebase.

## DFL-AA in a nutshell

**DFL-AA (Decentralized Federated Learning with Adaptive AoI-weighted Aggregation)** is a lightweight framework designed for **decentralized FL over wireless links** where communication is **unreliable** (packet loss).Operating over lossy wireless links under constraints, these systems cannot rely on retransmissions, so model parameters must be accepted as partial chunks, leading to two key failure modes, which are selection bias, where poor-quality links are systematically under-represented in gossip aggregation, and update staleness, where asynchronous nodes contribute outdated models. We prove that classical gossip aggregation introduces irreducible selection bias proportional to the link-loss rate.

### Key idea: partial + stale updates done right
DFL-AA handles **partial** and **stale** neighbor updates using two simple mechanisms:

- **Spatial: IPW corrects selection bias ->** Under partial reception, low-quality links contribute fewer messages, leading to systematic under-representation. This creates selection bias, where the received models are not a uniform sample of neighbors. DFL-AA uses the Horvitz–Thompson correction by weighting each received update with inverce reception estimation, so weaker links are up-weighted to compensate for their lower visibility.
- **Temporal: AoI decay discounts staleness ->** DFL-AA addresses the staleness issue by applying an exponential decay, reducing the influence of outdated updates relative to fresh ones. This operates independently of IPW, where the first corrects spatial sampling bias, while the second accounts for temporal staleness.


## System Diagram

<p align="center">
  <img src="https://github.com/DFedN/DFedNexus/blob/srs/main_dflaa.png" alt="DFL-AA System Diagram" width="85%">
</p>

##### Here is the promising performances of our method compared to other baselines (1. on MNIST and 2. on Fashion MNIST) 

<p align="center">
  <img src="https://github.com/DFedN/DFL-AA/blob/main/icdcs_paper_results/mnist_time_mean_1x3_zoom.png" alt="DFL-AA System Diagram" width="85%">
</p>

<p align="center">
  <img src="https://github.com/DFedN/DFL-AA/blob/main/icdcs_paper_results/fmnist_time_mean_1x3_zoom.png" alt="DFL-AA System Diagram" width="85%">
</p>

<br>
<br>

* * *
* * *
* * *

<br>

This repository provides the simulator code and config scripts required to reproduce the experiments and generate figures reported in our work. The typical workflow is:


> **Important:** The scripts assume specific default folder structures and output paths.  
> Please use the same locations as in the code, or update paths carefully if you change them.

<br>


## Installation

Install dependencies from:

```bash
pip install -r requirements.txt
```

Install DFedNexus itself:
```bash
pip install -e src/
```

#### (b) Ablation / component-wise results

``` bash
python main_results_comp_abl.py \
    --root ablation_results \
    --dataset mnist \
    --alpha 0.1,0.5 \
    --aggregations dflaa,dflaa_s,dflaa_c,softSGD,softGSD_c \
    --out-dir icdcs_paper_results
```

> **Tip:** For results generated under custom modifications (e.g., altered core parameters), reuse the logic in `main_results.py` and pass the correct results directory.


<br>

* * *

#### Anonymity Note

*This repository was created **solely for anonymous sharing** of the simulator code and results for **reproducibility**. It was created **after all paper experiments were completed**, and the code/results were then **copied and organized** here.*

*As a result, this repository **does not include the full development history or commit record** from the original private repository. However, it contains **all simulator code used to produce the reported results**—only the full commit history is missing.*






