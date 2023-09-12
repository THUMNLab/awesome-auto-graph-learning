# awesome-auto-graph-learning
This is a paper collection about **automated graph learning**, i.e., fusing AutoML and graph learning. Two special focuses are **graph hyper-parameter optimization (HPO)** and **graph neural architecture search (NAS)**.

**Please submit a pull request if you want to add new papers or have any suggestions!**


## Survey
* [IJCAI 2021] **Automated Machine Learning on Graphs: A Survey** [(Paper)](https://arxiv.org/abs/2103.00742)
* [Extension] **Automated Graph Machine Learning: Approaches, Libraries and Directions** [(Paper)](https://arxiv.org/abs/2201.01288)
* (In Chinese)[计算机学报 2023] **图神经架构搜索综述** [(Paper)](http://cjc.ict.ac.cn/online/onlinepaper/zzw-202375101026.pdf)

## Tool
* [ICLR 2021 GTRL workshop] **AutoGL: A Library for Automated Graph Learning** [(Code)](https://github.com/THUMNLab/AutoGL) [(Homepage)](https://mn.cs.tsinghua.edu.cn/AutoGL) [(Paper)](https://openreview.net/pdf?id=0yHwpLeInDn) 

## Graph NAS
### 2023
* [ICML 2023] **Do Not Train It A Linear Neural Architecture Search of Graph Neural Networks** [(Paper)](https://arxiv.org/abs/2305.14065) 
* [ICLR 2023] **AutoGT Automated Graph Transformer Architecture Search** [(Paper)](https://openreview.net/forum?id=GcM7qfl5zY)
* [WWW 2023] **Auto-HeG Automated Graph Neural Network on Heterophilic Graphs** [(Paper)](https://arxiv.org/abs/2302.12357)
* [WWW 2023] **Search to Capture Long-range Dependency with Stacking GNNs for Graph Classification** [(Paper)](https://arxiv.org/pdf/2302.08671.pdf)
* [AAAI 2023] **Dynamic Heterogeneous Graph Attention Neural Architecture Search** [(Paper)](https://zw-zhang.github.io/files/2023_AAAI_DHGAS.pdf)
* [AAAI 2023] **Differentiable Meta Multigraph Search with Partial Message Propagation on Heterogeneous Information Networks** [(Paper)](https://arxiv.org/abs/2211.14752)
* [ICDE 2023] **PSP Progressive Space Pruning for Efficient Graph Neural Architecture Search** [(Paper)](https://ieeexplore.ieee.org/document/9835246/)
* [TKDE 2023] **Automated Graph Neural Network Search under Federated Learning Framework** [(Paper)](https://ieeexplore.ieee.org/document/10056291/)
* [TKDE 2023] **HGNAS++ Efficient Architecture Search for Heterogeneous Graph Neural Networks** [(Paper)](https://ieeexplore.ieee.org/document/10040227)
* [TOIS 2023] **Neural Architecture Search for GNN-based Graph Classification** [(Paper)](https://dl.acm.org/doi/10.1145/3584945)
* [AI 2023] **AutoSTG+: An automatic framework to discover the optimal network for spatio-temporal graph prediction** [(Paper)](https://www.sciencedirect.com/science/article/pii/S0004370223000450)
* [TETC 2023] **CommGNAS: Unsupervised Graph Neural Architecture Search for Community Detection** [(Paper)](https://ieeexplore.ieee.org/document/10112632)
* [EngAppAI 2023] **Meta-GNAS Meta-reinforcement learning for graph neural architecture search** [(Paper)](https://www.sciencedirect.com/science/article/pii/S0952197623004840)
* [KAIS 2023] **GM2NAS: multitask multiview graph neural architecture** [(Paper)](https://link.springer.com/article/10.1007/s10115-023-01886-7)
* [ASOC 2023] **A surrogate evolutionary neural architecture search algorithm for graph neural networks search** [(Paper)](https://www.sciencedirect.com/science/article/abs/pii/S1568494623005033) [(Code)](https://github.com/chnyliu/CTFGNAS)
* [arXiv 2023] **Efficient and Explainable Graph Neural Architecture Search via Monte-Carlo Tree Search** [(Paper)](https://arxiv.org/abs/2308.15734) [(Code)](https://github.com/OnizukaLab/AutoGNN_mcts)
### 2022
* [TKDD 2022] **Auto-STGCN: Autonomous Spatial-Temporal Graph Convolutional Network Search** [(Paper)](https://arxiv.org/abs/2010.07474)
* [ICDM 2022] **Multi-Relational Graph Neural Architecture Search with Fine-grained Message Passing** [(Paper)](https://ieeexplore.ieee.org/document/10027750/)
* [NeurIPS 2022] **NAS-Bench-Graph: Benchmarking Graph Neural Architecture Search** [(Paper)](https://arxiv.org/abs/2206.09166)[(Code)](https://github.com/THUMNLab/NAS-Bench-Graph)
* [CIKM 2022] **GraTO: Graph Neural Network Framework Tackling Over-smoothing with Neural Architecture Search** [(Paper)](https://arxiv.org/pdf/2208.09027.pdf) [(Code)](https://github.com/fxsxjtu/GraTO)
* [ICML 2022] **Large-Scale Graph Neural Architecture Search** [(Paper)](https://zw-zhang.github.io/files/2022_ICML_GAUSS.pdf) [(Code)](https://github.com/THUMNLab/GAUSS)
* [ICML 2022] **Graph Neural Architecture Search Under Distribution Shifts** [(Paper)](https://zw-zhang.github.io/files/2022_ICML_GRACES.pdf)
* [ICML 2022] **DFG-NAS: Deep and Flexible Graph Neural Architecture Search** [(Paper)](https://arxiv.org/abs/2206.08582) [(Code)](https://github.com/PKU-DAIR/DFG-NAS)
* [KDD 2022] **Graph Neural Networks with Node-wise Architecture** [(Paper)](https://www.bolin-ding.com/papers/kdd22nwgnn.pdf)
* [KDDDLG 2022] **Graph Property Prediction on Open Graph Benchmark: A Winning Solution by Graph Neural Architecture Search** [(Paper)](https://arxiv.org/abs/2207.06027) [(Code)](https://github.com/AutoML-Research/PAS-OGB)
* [SIGIR 2022] **AutoGSR: Neural Architecture Search for Graph-based Session Recommendation** [(Paper)](https://dl.acm.org/doi/10.1145/3477495.3531940)
* [TKDE 2022] **GraphNAS++: Distributed Architecture Search for Graph Neural Network** [(Paper)](https://ieeexplore.ieee.org/document/9782531)
* [CVPR 2022] **Automatic Relation-aware Graph Network Proliferation** [(Paper)](https://arxiv.org/abs/2205.15678) [(Code)](https://github.com/phython96/ARGNP)
* [WWW 2022] **PaSca a Graph Neural Architecture Search System under the Scalable Paradigm** [(Paper)](https://arxiv.org/abs/2203.00638) [(Code)](https://github.com/PKU-DAIR/SGL)
* [WWW 2022] **Designing the Topology of Graph Neural Networks A Novel Feature Fusion Perspective** [(Paper)](https://arxiv.org/abs/2112.14531) [(Code)](https://github.com/AutoML-Research/F2GNN)
* [ICDE 2022] **AutoHEnsGNN Winning Solution to AutoGraph Challenge for KDD Cup 2020** [(Paper)](https://arxiv.org/abs/2111.12952) [(Code)](https://github.com/aister2020/KDDCUP_2020_AutoGraph_1st_Place)
* [TPDS 2022] **Auto-GNAS A Parallel Graph Neural Architecture Search Framework** [(Paper)](https://ieeexplore.ieee.org/document/9714826) 
* [WSDM 2022] **Profiling the Design Space for Graph Neural Networks based Collaborative Filtering** [(Paper)](http://www.shichuan.org/doc/125.pdf) [(Code)](https://github.com/BUPT-GAMMA/Design-Space-for-GNN-based-CF)
* [ESWA 2022] **Efficient graph neural architecture search using Monte Carlo Tree search and prediction network** [(Paper)](https://www.sciencedirect.com/science/article/pii/S0957417422019340)
* [Applied Intelligence] **Automatic search of architecture and hyperparameters of graph convolutional networks for node classification** [(Paper)](https://link.springer.com/article/10.1007/s10489-022-04096-w)
* [arXiv 2022] **AutoKE: An automatic knowledge embedding framework for scientific machine learning** [(Paper)](https://arxiv.org/abs/2205.05390)
* [arXiv 2022] **Enhancing Intra-class Information Extraction for Heterophilous Graphs: One Neural Architecture Search Approach** [(Paper)](https://arxiv.org/abs/2211.10990)

### 2021
* [NeurIPS 2021] **Graph Differentiable Architecture Search with Structure Learning** [(Paper)](https://openreview.net/forum?id=kSv_AMdehh3) [(Code)](https://github.com/THUMNLab/AutoGL)
* [NeurIPS 2021] **AutoGEL: An Automated Graph Neural Network with Explicit Link Information** [(Paper)](https://openreview.net/forum?id=PftCCiHVQP) [(Code)](https://github.com/zwangeo/AutoGEL)
* [ICDM 2021] **Heterogeneous Graph Neural Architecture Search** [(Paper)](https://ieeexplore.ieee.org/document/9679011)
* [IJCNN 2021] **Automated Graph Representation Learning for Node Classification** [(Paper)](https://ieeexplore.ieee.org/document/9533811)
* [PRICAI 2021] **ALGNN Auto-Designed Lightweight Graph Neural Network** [(Paper)](https://link.springer.com/chapter/10.1007/978-3-030-89188-6_37) 
* [CIKM 2021] **Pooling Architecture Search for Graph Classification** [(Paper)](https://arxiv.org/pdf/2108.10587.pdf) [(Code)](https://github.com/AutoML-Research/PAS)
* [KDD 2021] **DiffMG Differentiable Meta Graph Search for Heterogeneous Graph Neural Networks** [(Paper)](https://arxiv.org/abs/2010.03250) [(Code)](https://github.com/AutoML-4Paradigm/DiffMG)
* [KDD 2021 DLG Workshop] **Learn Layer-wise Connections in Graph Neural Networks** [(Paper)](https://drive.google.com/file/d/11BxUT80T7FfjbM55YjpX-yvnoxRERCIN/view)
* [ICML 2021] **AutoAttend Automated Attention Representation Search** [(Paper)](http://proceedings.mlr.press/v139/guan21a/guan21a.pdf)
* [SIGIR 2021] **GraphPAS Parallel Architecture Search for Graph Neural Networks** [(Paper)](https://dl.acm.org/doi/abs/10.1145/3404835.3463007)
* [CVPR 2021] **Rethinking Graph Neural Network Search from Message-passing** [(Paper)](https://arxiv.org/abs/2103.14282) [(Code)](https://github.com/phython96/GNAS-MP)
* [GECCO 2021] **Fitness Landscape Analysis of Graph Neural Network Architecture Search Spaces** [(Paper)](https://dl.acm.org/doi/10.1145/3449639.3459318) [(Code)](https://github.com/mhnnunes/fla_nas_gnn)
* [EuroSys 2021 EuroMLSys workshop] **Learned low precision graph neural networks** [(Paper)](https://arxiv.org/abs/2009.09232)
* [WWW 2021]  **Autostg: Neural architecture search for predictions of spatio-temporal graphs** [(Paper)](http://panzheyi.cc/publication/pan2021autostg/paper.pdf) [(Code)](https://github.com/panzheyi/AutoSTG)
* [ICDE 2021] **Search to aggregate neighborhood for graph neural network** [(Paper)](https://arxiv.org/abs/2104.06608) [(Code)](https://github.com/AutoML-4Paradigm/SANE)
* [AAAI 2021] **One-shot graph neural architecture search with dynamic search space** [(Paper)](https://www.aaai.org/AAAI21Papers/AAAI-3441.LiY.pdf)
* [arXiv] **Search For Deep Graph Neural Networks** [(Paper)](https://arxiv.org/pdf/2109.10047.pdf)
* [arXiv] **G-CoS GNN-Accelerator Co-Search Towards Both Better Accuracy and Efficiency** [(Paper)](https://arxiv.org/pdf/2109.08983.pdf)
* [arXiv] **Edge-featured Graph Neural Architecture Search** [(Paper)](https://arxiv.org/pdf/2109.0135.pdf)
* [arXiv] **FL-AGCNS: Federated Learning Framework for Automatic Graph Convolutional Network Search** [(Paper)](https://arxiv.org/abs/2104.04141)

### 2020
* [NeurIPS 2020] **Design space for graph neural networks** [(Paper)](https://arxiv.org/abs/2011.08843) [(Code)](https://github.com/snap-stanford/GraphGym)
* [ICONIP 2020] **Autograph: Automated graph neural network** [(Paper)](https://arxiv.org/abs/2011.11288)
* [BigData 2020] **Graph neural network architecture search for molecular property prediction** [(Paper)](https://arxiv.org/abs/2008.12187) [(Code)](https://github.com/deephyper/nas-gcn)
* [CIKM 2020] **Genetic Meta-Structure Search for Recommendation on Heterogeneous Information Network** [(Paper)](https://arxiv.org/pdf/2102.10550) [(Code)](https://github.com/0oshowero0/GEMS)
* [CIKM 2020 CSSA workshop] **Simplifying architecture search for graph neural network**[(Paper)](https://arxiv.org/abs/2008.11652) [(Code)](https://github.com/AutoML-4Paradigm/SNAG)
* [BRACIS 2020] **Neural architecture search in graph neural networks** [(Paper)](https://arxiv.org/abs/2008.00077) [(Code)](https://github.com/mhnnunes/nas_gnn)
* [IJCAI 2020] **Graph neural architecture search** [(Paper)](https://www.ijcai.org/proceedings/2020/195) [(Code)](https://github.com/GraphNAS/GraphNAS)
* [CVPR 2020] **SGAS: Sequential Greedy Architecture Search** [(Paper)](https://arxiv.org/abs/1912.00195) [(Code)](https://github.com/lightaime/sgas)
* [AAAI 2020] **Learning graph convolutional network for skeleton-based human action recognition by neural searching** [(Paper)](https://arxiv.org/abs/1911.04131) [(Code)](https://github.com/xiaoiker/GCN-NAS)
* [OpenReview 2020] **Efficient graph neural architecture search** [(Paper)](https://openreview.net/forum?id=IjIzIOkK2D6)
* [OpenReview 2020] **FGNAS: FPGA-Aware Graph Neural Architecture Search** [(Paper)](https://openreview.net/forum?id=cq4FHzAz9eA)
* [arXiv 2020] **Evolutionary architecture search for graph neural networks** [(Paper)](https://arxiv.org/abs/2009.10199) [(Code)](https://github.com/IRES-FAU/Evolutionary-Architecture-Search-for-Graph-Neural-Networks) 
* [arXiv 2020] **Probabilistic dual network architecture search on graphs** [(Paper)](https://arxiv.org/abs/2003.09676)

### 2019
* [arXiv 2019] **Auto-gnn: Neural architecture search of graph neural networks** [(Paper)](https://arxiv.org/abs/1909.03184)

## Graph HPO
### 2023
* [TKDE 2023] **Revisiting Embedding Based Graph Analyses Hyperparameters Matter** [(Paper)](https://ieeexplore.ieee.org/abstract/document/9994037)
### 2022
* [CIKM 2022] **Calibrate Automated Graph Neural Network via Hyperparameter Uncertainty** [(Paper)](https://zxj32.github.io/data/CIKM_2022.pdf) 
* [KAIS 2022] **Autonomous graph mining algorithm search with best performance trade-off** [(Paper)](https://link.springer.com/article/10.1007/s10115-022-01683-8)
* [ACL 2022] **KGTuner: Efficient Hyper-parameter Search for Knowledge Graph Learning** [(Paper)](https://arxiv.org/pdf/2205.02460.pdf)
* [arXiv 2022] **Start Small, Think Big On Hyperparameter Optimization for Large-Scale Knowledge Graph Embeddings** [(Paper)](https://arxiv.org/abs/2207.04979)
* [arXiv 2022] **Assessing the Effects of Hyperparameters on Knowledge Graph Embedding Quality** [(Paper)](https://arxiv.org/abs/2207.00473)
### 2021
* [ICML 2021] **Explainable Automated Graph Representation Learning with Hyperparameter Importance** [(Paper)](http://proceedings.mlr.press/v139/wang21f/wang21f.pdf)
* [SIGIR 2021] **Automated Graph Learning via Population Based Self-Tuning GCN** [(Paper)](https://arxiv.org/abs/2107.04713)
* [PRICAI 2021] **Automatic Graph Learning with Evolutionary Algorithms: An Experimental Study** [(Paper)](https://link.springer.com/chapter/10.1007/978-3-030-89188-6_38) 
* [GECCO 2021] **Which Hyperparameters to Optimise? An Investigation of Evolutionary Hyperparameter Optimisation in Graph Neural Network For Molecular Property Prediction** [(Paper)](https://arxiv.org/pdf/2104.06046.pdf)
* [P2PNA 2021] **ASFGNN Automated separated-federated graph neural network** [(Paper)](https://arxiv.org/abs/2011.03248)
* [arXiv 2021] **A novel genetic algorithm with hierarchical evaluation strategy for hyperparameter optimisation of graph neural networks** [(Paper)](https://arxiv.org/abs/2101.09300)
* [arXiv 2021] **Jitune: Just-in-time hyperparameter tuning for network embedding algorithms** [(Paper)](https://arxiv.org/abs/2101.06427)

### 2020
* [ICDM 2020] **Autonomous graph mining algorithm search with best speed/accuracy trade-off** [(Paper)](https://arxiv.org/abs/2011.14925) [(Code)](https://github.com/minjiyoon/ICDM20-AutoGM)

### 2019
* [KDD 2019] **AutoNE: Hyperparameter optimization for massive network embedding** [(Paper)](http://pengcui.thumedialab.com/papers/AutoNE.pdf) [(Code)](https://github.com/tadpole/AutoNE)

## Applications
### Finance
* [CIKM 2022] **Explainable Graph-based Fraud Detection via Neural Meta-graph Search** [(Paper)](https://ponderly.github.io/pub/NGS_CIKM2022.pdf)
### Biology
* [Bioinformatics 2023] **Cancer Drug Response Prediction With Surrogate Modeling-Based Graph Neural Architecture Search** [(Paper)](https://academic.oup.com/bioinformatics/article/39/8/btad478/7239861) [(Code)](https://github.com/BeObm/AutoCDRP)
* [arXiv 2023] **Uncertainty Quantification for Molecular Property Predictions with Graph Neural Architecture Search** [(Paper)](https://arxiv.org/abs/2307.10438) [(Code)](https://github.com/zavalab/ML/tree/master/AUTOGNNUQ)
* [TCBB 2022] **Multi-view Graph Neural Architecture Search for Biomedical Entity and Relation Extraction** [(Paper)](https://ieeexplore.ieee.org/document/9881878)
* [TCBB 2022] **AutoMSR: Auto Molecular Structure Representation Learning for Multi-label Metabolic Pathway Prediction** [(Paper)](https://ieeexplore.ieee.org/document/9864145)
* [AILSCI 2022] **AutoGGN: A gene graph network AutoML tool for multi-omics research** [(Paper)](https://www.biorxiv.org/content/10.1101/2021.04.30.442074v2)
* [BIBM 2021] **Multi-label Metabolic Pathway Prediction with Auto Molecular Structure Representation Learning** [(Paper)](https://ieeexplore.ieee.org/document/9669309)
### Knowledge Graph Embedding
* [arXiv 2021] **AutoSF+: Towards Automatic Scoring Function Design for Knowledge Graph Embedding** [(Paper)](https://arxiv.org/abs/2107.00184)
* [ICDE 2020] **AutoSF: Searching Scoring Functions for Knowledge Graph Embedding** [(Paper)](https://arxiv.org/abs/1904.11682) [(Code)](https://github.com/AutoML-4Paradigm/AutoSF)
### Others
* [CASES 2023] **MaGNAS: A Mapping-Aware Graph Neural Architecture Search Framework for Heterogeneous MPSoC Deployment** [(Paper)](https://arxiv.org/abs/2307.08065) 
* [TIST 2023] **Dual Graph Convolution Architecture Search for Travel Time Estimation** [(Paper)](https://dl.acm.org/doi/10.1145/3591361)
* [WWW 2023] **Automated Self-Supervised Learning for Recommendation** [(Paper)](https://dl.acm.org/doi/10.1145/3543507.3583336)

## Miscellaneous
### Self-supervised Learning
* [LOGS 2022] **AutoGDA Automated Graph Data Augmentation for Node Classification** [(Paper)](https://proceedings.mlr.press/v198/zhao22a.html)
* [ICLR 2022] **Automated Self-Supervised Learning for Graphs** [(Paper)](https://arxiv.org/pdf/2106.05470.pdf) [(Code)](https://github.com/ChandlerBang/AutoSSL)
* [AAAI 2022] **AutoGCL Automated Graph Contrastive Learning via Learnable View Generators** [(Paper)](https://arxiv.org/abs/2109.10259) [(Code)](https://github.com/Somedaywilldo/AutoGCL)
* [ICML 2021] **Graph Contrastive Learning Automated** [(Paper)](https://arxiv.org/abs/2106.07594) [(Code)](https://github.com/Shen-Lab/GraphCL_Automated) 
### Others
* [ICLR 2023] **AutoTransfer AutoML with Knowledge Transfer - An Application to Graph Neural Networks** [(Paper)](https://openreview.net/forum?id=yT8twuGdqCX)
* [arXiv 2022] **AutoGML Fast Automatic Model Selection for Graph Machine Learning** [(Paper)](https://arxiv.org/abs/2206.09280)
* [TKDE 2021] **Automated Unsupervised Graph Representation Learning** [(Paper)](https://ieeexplore.ieee.org/document/9547743/) [(Code)](https://drive.google.com/drive/folders/1F7_LWvEg9Z70OxW2YJmivzg3qJ7tC6mE)
* [arXiv 2022] **Bridging the Gap of AutoGraph between Academia and Industry: Analysing AutoGraph Challenge at KDD Cup 2020** [(Paper)](https://arxiv.org/abs/2204.02625)
 
## Cite

Please consider citing our [survey paper](http://arxiv.org/abs/2103.00742) if you find this repository helpful:
```
@inproceedings{zhang2021automated,
  title={Automated Machine Learning on Graphs: A Survey},
  author={Zhang, Ziwei and Wang, Xin and Zhu, Wenwu},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, {IJCAI-21}},
  year={2021},
  note={Survey track}
}
```
 
 
<!--
hide not very related papers
### Structure Learning
* [ICPR 2020] **AOAM Automatic Optimization of Adjacency Matrix for Graph Convolutional Network** [(Paper)](https://ieeexplore.ieee.org/document/9412046/) [(Code)](https://github.com/xshura/AOAM)
### Explanation
* [ICML 2021 XAI workshop] **Towards Automated Evaluation of Explanations in Graph Neural Networks** [(Paper)](https://arxiv.org/abs/2106.11864) 
-->

