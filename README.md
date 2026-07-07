# Traffic Sign Recognition

- [Proposal Docs](https://docs.google.com/document/d/1LFfMRnj8OV4WV3CxFSy26Opk5PrQXEab/edit?usp=sharing&ouid=117888008832896648018&rtpof=true&sd=true)
- [Project Docs](https://docs.google.com/document/d/145yTLLexleQkfP3VFrixo-X2_qqgzIK8/edit?usp=sharing&ouid=117888008832896648018&rtpof=true&sd=true)

## Team Members

| # | Name |
| :-: | :--- |
| 1 | Khor Yu Zhuang |
| 2 | Tan Yu Keat |
| 3 | Kang Kah Yi |
| 4 | Leow Wei Ru |

## Task Distribution

| Chapter | Sections & Assigned Members |
| :--- | :--- |
| — | **Abstract & Area of Study & Keywords** — Member 1 |
| Chapter 1 | **Problem Statement and Motivation** — Member 2<br>**Scope & Objectives** — Member 3<br>**Contribution** — Member 4<br>**Report Organization** — Member 4 |
| Chapter 2 | **Literature Review 1** — Member 1<br>**Literature Review 2** — Member 2<br>**Literature Review 3** — Member 3<br>**Literature Review 4** — Member 4<br>**Comparison of the 4 Techniques** — All members |
| Chapter 3 | **System Design / Overview** — Member 1<br>**System Requirements (Hardware & Software)** — Member 2<br>**Implementation Issues & Challenges** — Member 3<br>**Timeline & Milestones for 13 Weeks** — Member 4 |
| Chapter 4 | **Coding Work 1 & Results** — Member 1<br>**Coding Work 2 & Results** — Member 2<br>**Coding Work 3 & Results** — Member 3<br>**Coding Work 4 & Results** — Member 4 |
| Chapter 5 | **Conclusion** — one paragraph per member (Members 1–4) |
| — | **References (IEEE style)** — All members |
| — | **Appendices (sample / source code)** — All members |



## Get Started

1. **Install Anaconda** — download Anaconda via `winget` (available but not limited to).

2. **Open Anaconda Prompt**.

3. **Install `nb_conda_kernels`**:
   ```bash
   conda install -c conda-forge nb_conda_kernels
   ```

4. **Create the environment** — navigate to the root directory of this repo, then run:
   ```bash
   conda env create -f environment.yml
   ```

5. **Launch Jupyter Notebook** — open Jupyter Notebook in Anaconda Navigator and choose the environment (kernel) you just created.

### Adding / Removing Packages

- Update `environment.yml` first, then execute the following in the same directory in Anaconda Prompt:
  ```bash
  conda env update -f environment.yml --prune
  ```
