## PCLL 
This repository gives the implementation for self-training with prototype-based pcll.

## Abstract
Partial label learning assumes that each training instance is labeled with a candidate label set, where the true label is concealed. In some real-world scenario, however, it is laborious to collect candidate labels from a long list of classes for the whole huge-size dataset. Instead, a common case is that each training instance in part of the dataset is specified with one incorrect class termed as complementary label while others are partially labeled, which well suits more real-world data annotation. For this article, we formulate such task as a novel learning framework dubbed partial-complementary label learning (PCLL) and propose prototype-based PCLL method by adopting self-training technique to handle this particular scenario. The working principle is to learn a classifier from partially labeled instances with prototype-based label disambiguation, which is then used to assign pseudo-labels to complementarily labeled instances based on the model’s output. When the confidence-over-threshold pseudo-label is not equal to the corresponding complementary label, instance with the produced label is used to train a new classifier coupling with the partially labeled instances. In this manner, the pseudo-labeled instances are gradually used to enrich the training set. As the number of iteration increases, they become more important and the label assignment becomes more reliable in identifying the true label. Experiments on benchmark datasets distinctly valid the efficiency of our proposed method

## Prerequisite
The requirements are in requirements.txt. 

## Usage
Train the model by running the following command directly. 
python -u train_pll_com_cifar_single.py --exp-dir experiment --prot_start 1 --lr 0.001 --wd 1e-5 --cosine --epochs 250 --proto_m 0.99 --partial_rate 0.1 --gpu 1

## Baseline
1.CAVL: Zhang F, Feng L, Han B, et al. Exploiting class activation value for partial-label learning[C]//International conference on learning representations. 2021. [code] https://github.com/SII-Ferenas/CAVL

2.PRODEN: Lv J, Xu M, Feng L, et al. Progressive identification of true labels for partial-label learning[C]//international conference on machine learning. PMLR, 2020: 6500-6510.
 [code] https://github.com/lvjiaqi77/PRODEN
 
3.RC CC: Feng L, Lv J, Han B, et al. Provably consistent partial-label learning[J]. Advances in neural information processing systems, 2020, 33: 10948-10960.
 [code] https://lfeng1995.github.io/Code/RCCC.zip
 
4.PLSP: Li X, Jiang Y, Li C, et al. Learning with partial labels from semi-supervised perspective[C]//Proceedings of the AAAI conference on artificial intelligence. 2023, 37(7): 8666-8674.  [code] https://github.com/changchunli/PLSP

5.MSE EXP: Feng L, Kaneko T, Han B, et al. Learning with multiple complementary labels[C]//International conference on machine learning. PMLR, 2020: 3072-3081.
 [code] https://lfeng1995.github.io/Code/LMCL.zip
 
6.NN: Ishida T, Niu G, Menon A, et al. Complementary-label learning for arbitrary losses and models[C]//International conference on machine learning. PMLR, 2019: 2971-2980.
 [code] https://github.com/takashiishida/comp
 
7.SELF-CL:  Liu J, Li B, Lei M, et al. Self-supervised knowledge distillation for complementary label learning[J]. Neural Networks, 2022, 155: 318-327.

8.SCL-NL: Chou Y T, Niu G, Lin H T, et al. Unbiased risk estimators can mislead: A case study of learning with complementary labels[C]//International conference on machine learning. PMLR, 2020: 1929-1938.

Kim Y, Yim J, Yun J, et al. Nlnl: Negative learning for noisy labels[C]//2019 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2019: 101-110.
[code] https://github.com/ydkim1293/NLNL-Negative-Learning-for-Noisy-Labels

9.IRNet: Lian Z, Xu M, Chen L, et al. IRNet: Iterative refinement network for noisy partial label learning[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
https://github.com/zeroQiaoba/IRNet
