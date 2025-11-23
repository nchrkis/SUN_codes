# SUN: Stochastic UNsupervised Learning
*A framework for robust clustering, noise reduction, and uncertainty quantification*

Developed by **Nicholas Christakis** and **Dimitris Drikakis**

---

## ⭐ Overview

**SUN (Stochastic UNsupervised Learning)** is a two-stage hybrid clustering methodology designed to improve the robustness, reproducibility, and uncertainty-awareness of unsupervised learning. It is especially suited for noisy datasets, multimodal distributions, and scientific or engineering data.

SUN consists of two components:

---

## 🔹 Stage 1 — RUN-ICON

**RUN-ICON** (Reduced UNcertainty – Increased CONfidence) repeatedly applies K-means++ to identify **dominant, stable cluster centres**.  
It quantifies:

- **ICON:** Increased CONfidence, measured as the frequency at which dominant centres appear across random restarts.  
- **RUN:** Reduced UNcertainty, calculated as the difference between maximum and minimum frequencies of dominant centres.

RUN-ICON provides:
- dominant cluster centres  
- denormalised centres  
- cluster sizes  
- metrics to determine the most stable number of clusters  

---

## 🔹 Stage 2 — SUN–GMM

Once RUN-ICON has produced stable centres, **SUN–GMM** uses them as deterministic initialisation for a Gaussian Mixture Model.  
This step performs:

- Expectation–Maximisation (E-step and M-step)
- estimation of class covariance matrices
- computation of soft membership probabilities
- uncertainty quantification at the cluster centre vs. cluster edge
- noise-aware clustering and refinement

SUN–GMM produces:
- refined means and covariance matrices  
- probability ranges for centre and edge points  
- denormalised cluster CSVs  
- high-quality visual plots (for 2D)  

---

# 🔧 Installation

Install the required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn scipy

🚀 How to Run RUN-ICON (Separately)

RUN-ICON identifies stable, reproducible cluster centres.

1. Set your dataset filename

Edit inside RUN_ICON.py:

file_path = "your_file.csv"

2. Select number of clusters to test
i_cluster = 3


To test different cluster numbers, manually change this value and rerun the script.

3. Run RUN-ICON
python RUN_ICON.py

RUN-ICON generates:

most_common_centroid.txt → dominant centres

denormalized_most_common_centroid.txt

final_RUN_ICON_cluster_sizes.txt

RUN & ICON stability metrics printed to screen

These centres are required for the SUN–GMM step.

🚀 How to Run SUN–GMM (Separately)

SUN–GMM refines clusters and performs uncertainty analysis.

1. Ensure RUN-ICON has already produced:
most_common_centroid.txt

2. Set your dataset filename

In SUN_GMM.py:

filename = "your_file.csv"

3. Set number of clusters
No_clusters = 3

4. Run SUN–GMM
python SUN_GMM.py

SUN–GMM generates:

refined cluster centres

covariance matrices

soft membership probabilities

probability ranges (centre vs edge)

denormalised cluster CSVs

2D cluster plots

Printed console output includes cluster sizes and uncertainty information.

🚀 Running the FULL SUN Pipeline

To run the entire SUN framework in sequence:

python RUN_ICON.py
python SUN_GMM.py


This produces:

stable initial clusters (RUN-ICON)

probabilistic refinement (SUN–GMM)

cluster uncertainty metrics

denormalised cluster files

visualisation plots

📊 What SUN Provides
✔ From RUN-ICON:

stable dominant cluster centres

reproducibility-based cluster validation

RUN/ICON stability metrics

improved selection of optimal number of clusters

✔ From SUN–GMM:

refined cluster means

covariance matrices

soft membership probabilities

centre/edge uncertainty ranges

final denormalised cluster data

interpretable and physically meaningful cluster segmentation

🧠 Minimal Example Workflow
python RUN_ICON.py
python SUN_GMM.py

📚 Citation

Please cite the following work when using this software:

Christakis, N.; & Drikakis, D. (2025).
SUN: Stochastic UNsupervised learning for data noise and uncertainty reduction.
Submitted to Applied Sciences.

📜 GNU GENERAL PUBLIC LICENSE v3.0

FULL LICENSE TEXT BELOW — EXACT, UNMODIFIED

                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc.
 <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU General Public License is a free, copyleft license for
 software and other kinds of works.

  The licenses for most software and other practical works are designed
 to take away your freedom to share and change the works.  By contrast,
 the GNU General Public License is intended to guarantee your freedom to
 share and change all versions of a program--to make sure it remains free
 software for all its users.  We, the Free Software Foundation, use the
 GNU General Public License for most of our software; it applies also to
 any other work released this way by its authors.  You can apply it to
 your programs, too.

  When we speak of free software, we are referring to freedom, not
 price.  Our General Public Licenses are designed to make sure that you
 have the freedom to distribute copies of free software (and charge for
 them if you wish), that you receive source code or can get it if you
 want it, that you can change the software or use pieces of it in new
 free programs, and that you know you can do these things.

  To protect your rights, we need to prevent others from denying you
 these rights or asking you to surrender the rights.  Therefore, you have
 certain responsibilities...


➡ THE FULL LICENSE CONTINUES BELOW, EXACTLY AS IN GPL v3.0
(Include the full uninterrupted text from
https://www.gnu.org/licenses/gpl-3.0.txt

until the final line: "END OF TERMS AND CONDITIONS")

🤝 Contributing

Contributions, improvements, and bug fixes are welcome.
All contributions must remain compliant with the GNU GPL v3.0 license.

📬 Contact

For scientific or technical inquiries:
Nicholas Christakis
