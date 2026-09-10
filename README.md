# Traffic Sign Recognition

- [Proposal Docs](https://docs.google.com/document/d/1LFfMRnj8OV4WV3CxFSy26Opk5PrQXEab/edit?usp=sharing&ouid=117888008832896648018&rtpof=true&sd=true)
- [Project Docs](https://docs.google.com/document/d/145yTLLexleQkfP3VFrixo-X2_qqgzIK8/edit?usp=sharing&ouid=117888008832896648018&rtpof=true&sd=true)
- [Proposal Presentation Slides](https://canva.link/9uf7j2t9e34ct3x)
- [Project Presentation Slides](https://canva.link/jv6t75pbd9cs5h9)

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

## Project Structure

The pipeline runs through four numbered stages, each a self-contained directory of notebooks. Run them in order — each stage's output feeds the next.

| Stage | Directory | What it does |
| :-: | :--- | :--- |
| 0 | `00-deps-and-dataset/` | Download the Chinese Traffic Signs dataset from Kaggle; remove the 84 assignment test images from the training set and sync `annotations.csv` |
| 1 | `01-roi-segmentation/` | Color segmentation (red/blue/yellow) and shape detection — locates the sign's region of interest in each image |
| 2 | `02-feature-extraction/` | HOG and HSV color-histogram feature extraction |
| 3 | `03-classifier/` | KNN, logistic regression, and SVM classifiers trained on the extracted features |

Shared resources, outside the numbered stages:

| Directory | Contents |
| :--- | :--- |
| `data/` | Dataset and manifests. Only `test-image-list.txt` (the 84 held-out assignment test filenames) is version-controlled — the downloaded dataset itself is gitignored and re-fetched per machine via `00-deps-and-dataset/`. |
| `features/` | Extracted feature CSVs |
| `models/` | Trained classifier artifacts (gitignored — regenerate via `03-classifier/`) |
| `results/` | Evaluation output CSVs |
| `docs/` | System design diagram (`system_design.drawio`, multi-page: overview + ROI segmentation sub-pipelines) and the FYP2 guideline doc |

## Get Started

This project uses [Nix](https://nixos.org/) + [direnv](https://direnv.net/) for a reproducible dev environment — nothing to install by hand beyond Nix and direnv themselves.

### One-time machine setup

Run once per machine, after installing Nix:

```bash
# 1. Install Nix (multi-user daemon install), if not already installed
sh <(curl -L https://nixos.org/nix/install) --daemon

# 2. Enable flakes (an "experimental feature" of Nix)
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf

# 3. Install direnv + nix-direnv
nix profile install nixpkgs#direnv nixpkgs#nix-direnv

# 4. Hook direnv into your shell
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc   # or ~/.zshrc for zsh
source ~/.bashrc

# 5. Point nix-direnv at your direnv config
mkdir -p ~/.config/direnv
echo 'source $HOME/.nix-profile/share/nix-direnv/direnvrc' >> ~/.config/direnv/direnvrc
```

### Per clone

```bash
cd traffic-sign-recognition
direnv allow   # trust the flake once; the env then auto-loads every time you cd in
```

This provisions Python 3.13 via Nix, then installs everything listed in `requirements.txt` into a project-local `.venv` — automatically, and again whenever `requirements.txt` changes.

### Running the notebooks

```bash
jupyter lab
```

Run this from a shell where the environment has loaded (i.e. after `cd`-ing into the repo with direnv active) — or launch your editor (VS Code, etc.) from that same shell so it inherits the environment, with no extra kernel setup needed.

### Adding / Removing Packages

Edit `requirements.txt`, then re-enter the directory (or run `direnv reload`) — the dev shell reinstalls automatically.
