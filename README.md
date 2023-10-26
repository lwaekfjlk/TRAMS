<div align="center">
  <h1> TRAMS: Training-free Memory Selection for Long-range Langauge Modeling </h1>

  [![License: Apache-2.0](https://img.shields.io/crates/l/Ap?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
</div>

  # 📌 Table of Contents
- [Introduction](#-introduction)
- [Development](#-development)
- [Data](#-data)
- [License](#-license)
- [Citation](#-citation)
- [Contact](#-contact)
# 🚀 Introduction
This repository is for the EMNLP-2023 findings paper [TRAMS: Training-free Memory Selection for Long-range Language Modeling](http://arxiv.org/abs/2310.15494).

# 💻 Development
## Setup
Clone the repository from GitHub and install:
```
git clone https://github.com/lwaekfjlk/TRAMS.git
cd TRAMS/
pip install -r requirements.txt
```
## Train Script
```bash
sh enwik8_xl_train.sh # for enwik8 XL train
sh wt103_xl_train.sh # for wt103 XL train
```
## Test Script
```bash
sh enwik8_xl.sh # for enwik8 baseline
sh enwik8_trams.sh # for enwik8 trams
sh wt103_xl.sh # for wt103 baseline
sh wt103_trams.sh # for wt103 trams
```
## Checkpoints

The Transformer-XL checkpoints for wikitext-103 and enwik8 are provided [here](https://drive.google.com/drive/folders/1XpN9nmTxE8l-FZdXhrHvxuf_hxH4Dmjd?usp=sharing).


# 📝 Data
We follow the instructions mentioned in Transformer-XL to collect enwik8 and wiki text-103 data. Data and its tokenized cache are provided [here](https://drive.google.com/drive/folders/1Bdc4l3nYG6q3JXDvydFTHNAZniiJFlBY?usp=sharing).

# 📜 License

This repository is released under the [Apache-2.0 License](LICENSE).

# 📚 Citation

If you find this repository useful, please cite it as follows:
```bibtex
@misc{yu2023trams,
      title={TRAMS: Training-free Memory Selection for Long-range Language Modeling}, 
      author={Haofei Yu and Cunxiang wang and Yue Zhang and Wei Bi},
      year={2023},
      eprint={2310.15494},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
## 📮 Contact
If you have any questions or feedback, please feel free to reach out at haofeiy@cs.cmu.edu.