## Papers on Recommendation Systems



### Survey

surveys on recommendation system and computational advertising system

* [Deep Learning based Recommender System A Survey and New Perspectives](https://arxiv.org/abs/1707.07435)



### Embedding

* Sequence Embedding
  * Word2Vec
  * Item2Vec
  * Listing Embedding
* DNN-based Embedding
  * AutoEncoder
  * WDL-ID2Vec
  * NCF-CF2Vec
  * YouTube DNN (Softmax Embedding)
* Graph Embedding
  * Node2Vec
  * LINE
  * DeepWalk
  * EGES

| Model       | Conference | Paper                                                        |
| ----------- | ---------- | ------------------------------------------------------------ |
| Word2Vec | arxiv'13   | [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf); [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf) [**Google**] |
| Item2Vec | arxiv'16   | [Item2Vec: Neural Item Embedding for Collaborative Filtering](https://arxiv.org/ftp/arxiv/papers/1603/1603.04259.pdf) [**Microsoft**] |
| GraphEmb | KDD'14 | [DeepWalk- Online Learning of Social Representations](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf) |
| LINE | WWW'15 | [LINE - Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf) [**Microsoft**] |
| Node2Vec | KDD'16 | [node2vec: Scalable Feature Learning for Networks](https://cs.stanford.edu/people/jure/pubs/node2vec-kdd16.pdf) |
| Youtube DNN | RecSys'16 | [Deep Neural Networks for YouTube Recommendations](https://ai.google/research/pubs/pub45530) [**Google**]|
| NCF | WWW'17   | [Neural Collaborative Filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf) |
| Listing emb | KDD'18 | [Real-time Personalization using Embeddings for Search Ranking at Airbnb](https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb) [**Airbnb**] |
|    | KDD'18   | [Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/pdf/1803.02349.pdf) [**Alibaba**] |
| Product emb | arxiv'19   | [Large-scale Collaborative Filtering with Product Embeddings](https://arxiv.org/abs/1901.04321) [**Amazon**] |
| DeepCF | AAAI'19 | [DeepCF: A Unified Framework of Representation Learning and Matching Function Learning in Recommender System](https://arxiv.org/abs/1901.04704) |

word2vec解释

- [Word2vec Explained Negative-Sampling Word-Embedding Method (2014)](https://arxiv.org/pdf/1402.3722.pdf)
- [Word2vec Parameter Learning Explained (2016)](https://arxiv.org/pdf/1411.2738.pdf)




### CTR Prediction

various CTR prediction models for recommendation systems

* Tree-based series
  * GBDT+LR
* FTRL series
  * FTRL
* FM series
  * FM/FFM
* Deep series

| Model     | Conference | Paper                                                        |
| --------- | ---------- | ------------------------------------------------------------ |
| LR        | WWW'07     | [Predicting Clicks: Estimating the Click-Through Rate for New Ads](https://dl.acm.org/citation.cfm?id=1242643) [**Microsoft**] |
| FM        | ICDM'10    | [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) |
| FTRL      | KDD'13     | [Ad Click Prediction: a View from the Trenches](https://www.researchgate.net/publication/262412214_Ad_click_prediction_a_view_from_the_trenches) [**Google**] |
| GBDT+LR   | ADKDD'14   | [Practical Lessons from Predicting Clicks on Ads at Facebook](https://dl.acm.org/citation.cfm?id=2648589) [**Facebook**] |
| CCPM      | CIKM'15    | [A Convolutional Click Prediction Model](http://www.escience.cn/system/download/73676) |
| FFM       | RecSys'16  | [Field-aware Factorization Machines for CTR Prediction](https://dl.acm.org/citation.cfm?id=2959134) [**Criteo**] |
| WDL | DLRS'16    | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) [**Google**] |
| FNN       | ECIR'16    | [Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/abs/1601.02376) [**RayCloud**] |
| PNN       | ICDM'16    | [Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf) |
| Youtube DNN    | RecSys'16   | [Deep Neural Networks for YouTube Recommendations](https://ai.google/research/pubs/pub45530) [**Google**] |
| DeepFM    | IJCAI'17   | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247), [**Huawei**] |
| NFM       | SIGIR'17   | [Neural Factorization Machines for Sparse Predictive Analytics](https://dl.acm.org/citation.cfm?id=3080777) |
| AFM       | IJCAI'17   | [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/0435.pdf) |
| DCN       | ADKDD'17   | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123) [**Google**] |
| xDeepFM   | KDD'18     | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf) [**Microsoft**] |
| DIN   | KDD'18     | [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf) [**Alibaba**] |
| FwFM      | WWW'18     | [Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising](https://arxiv.org/pdf/1806.03514.pdf) [**LinkedIn, Ablibaba**] |
| FPE   | RecSys'18   | [Field-aware Probabilistic Embedding Neural Network for CTR Prediction](https://dl.acm.org/citation.cfm?id=3240396) |
| AutoInt   | arxiv'18   | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921) |
| SASRec | ICDM'18 | [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) |
| IFM   | AAAI'19   | [Interaction-aware Factorization Machines for Recommender Systems](https://arxiv.org/abs/1902.09757) [**Tencent**]|
| DeepGBM   | KDD'19   | [DeepGBM: A Deep Learning Framework Distilled by GBDT for Online Prediction Tasks](https://www.microsoft.com/en-us/research/publication/deepgbm-a-deep-learning-framework-distilled-by-gbdt-for-online-prediction-tasks/) [**Microsoft**]|
| OENN  | SIGIR'19   | [Order-aware Embedding Neural Network for CTR Prediction](https://dl.acm.org/citation.cfm?id=3331332) [**Huawei**]|
| DIEN  | AAAI'19   | [Deep Interest Evolution Network for Click Through Rate Prediction](https://arxiv.org/abs/1809.03672) [**Alibaba**]|
| DSIN  | arxiv'19   | [Deep Session Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1905.06482) [**Alibaba**]|
| OANN  | arxiv'19  | [Operation-aware Neural Networks for User Response Prediction ](https://arxiv.org/pdf/1904.12579) |
| FGCNN | arxiv'19  | [Feature Generation by Convolutional Neural Network](https://arxiv.org/pdf/1904.04447) [**Huawei**] |
| FiBiNET | RecSys'19  | [FiBiNET: combining feature importance and bilinear feature interaction for click-through rate prediction](https://dl.acm.org/citation.cfm?id=3347043) [**Sina**] |


> CTR预估深度模型演化之路：https://mp.weixin.qq.com/s/jpWS9ec0MCO4ncSZx38r3w https://zhuanlan.zhihu.com/p/86181485
>
> DeepCTR：易用可扩展的深度学习点击率预测算法包： https://zhuanlan.zhihu.com/p/53231955 



### TL related

Transfer Learning related/Multi-Task Learning based Recommendation Systems
* [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/pdf/1706.05098.pdf)
* [A Survey on Transfer Learning](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)


| Model     | Conference | Paper                                                        |
| --------- | ---------- | ------------------------------------------------------------ |
| ESMM      | SIGIR’18   | [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931) [**Alibaba**] |
| MMoE      | KDD’18     | [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/10.1145/3219819.3220007) [**Google**] |
| YouTube-MTL    | RecSys’19    | [Recommending What Video to Watch Next: A Multitask Ranking System](https://daiwk.github.io/assets/youtube-multitask.pdf) [**Google**] |
| DeepMCP        | IJCAI’19     | [Representation Learning-Assisted Click-Through Rate Prediction](https://www.ijcai.org/Proceedings/2019/0634.pdf) [**Alibaba**] |



- Schick, Timo, and Hinrich Schütze. ["It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners."](https://github.com/keyurfaldu/AIgrads/blob/master/summary/PET.md) arXiv preprint [[arXiv:2009.07118](https://arxiv.org/pdf/2009.07118.pdf)] (2020).
- [Zero-Shot Transfer Learning with Synthesized Data for Multi-Domain Dialogue State Tracking](https://github.com/keyurfaldu/AIgrads/blob/master/summary/cai_synthetic_data.md), Giovanni Campagna Agata Foryciarz Mehrad Moradshahi Monica S. Lam, ACL 2020 [[arXiv](https://www.aclweb.org/anthology/2020.acl-main.12.pdf)]
- [DReCa: A General Task Augmentation Strategy for Few-Shot Natural Language Inference](https://github.com/keyurfaldu/AIgrads/blob/master/summary/DReCa.md) Shikhar Murty, Tatsunori B. Hashimoto, Christopher D. Manning, 2020 [[arXiv](https://openreview.net/pdf?id=PqsalKqGudW)]




### FL related

Federated Learning based Recommendation Systems

* [Federated Recommendation Systems](https://ieeexplore.ieee.org/document/9005952)

* [FedFast: Going Beyond Average for Faster Training of Federated Recommender Systems](https://dl.acm.org/doi/abs/10.1145/3394486.3403176)



### DRL related

* Deep Reinforcement Learning based Recommendation Systems


| Conference | Paper                                                        |
| ---------- | ------------------------------------------------------------ |
| WWW'18   | [DRN：A Deep Reinforcement Learning Framework for News Recommendation](https://dl.acm.org/citation.cfm?id=3185994) [**MSRA**]|
| RecSys'18 | [Deep Reinforcement Learning for Page-wise Recommendations](https://dl.acm.org/citation.cfm?id=3240374) [**JD**] |
| Arxiv'17 | [Deep Reinforcement Learning for List-wise Recommendations](https://arxiv.org/abs/1801.00209) [**JD**] |
| KDD'18 | [Stabilizing Reinforcement Learning in Dynamic Environment with Application to Online Recommendation](https://arxiv.org/abs/1801.00209) |
| KDD'18 | [Reinforcement Learning to Rank in E-Commerce Search Engine: Formalization, Analysis, and Application](https://dl.acm.org/citation.cfm?id=3219846) |
|  | https://tech.meituan.com/2018/11/15/reinforcement-learning-in-mt-recommend-system.html [**Meituan**] |



### XAI

Explainable AI and model interpretation methods for ML models

* PDP：[A simple and effective model-based variable importance measure](https://arxiv.org/pdf/1805.04755.pdf). Greenwell, Brandon M., Bradley C. Boehmke, and Andrew J. McCarthy. arXiv preprint arXiv:1805.04755 (2018).
* ICE：[Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation](https://arxiv.org/pdf/1309.6392.pdf). Goldstein, Alex, Adam Kapelner, Justin Bleich, and Emil Pitkin. Journal of Computational and Graphical Statistics 24, no. 1 (2015): 44-65.
* LIME：[LIME: "Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://github.com/keyurfaldu/AIgrads/blob/master/summary/LIME.md) Ribeiro, Sameer Singh, Guestrin, KDD 2016 [[arXiv](https://arxiv.org/pdf/1602.04938.pdf)]
* SHAP：[SHAP: A Unified Approach to Interpreting Model Predictions](https://github.com/keyurfaldu/AIgrads/blob/master/summary/shap.md), Lundberg, Lee, NIPS 2017 [[arXiv](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)]
* DeepLift：[DeepLift: Learning Important Features Through Propagating Activation Differences](https://github.com/keyurfaldu/AIgrads/blob/master/summary/deeplift.md) Avanti Shrikumar et al, ICML 2019 [[arXiv](https://arxiv.org/pdf/1704.02685.pdf)]
* RETAIN
* LRP

Methods

* https://christophm.github.io/interpretable-ml-book/
* [Principles and Practice of Explainable Machine Learning](https://github.com/keyurfaldu/AIgrads/blob/master/summary/explainable_ml.md) Vaishak Belle, Ioannis Papantonis [[arXiv](https://arxiv.org/pdf/2009.11698.pdf)]
* [Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges toward Responsible AI](https://github.com/keyurfaldu/AIgrads/blob/master/summary/xai_concepts.md), Arrieta et al., 2019 [[arXiv](https://arxiv.org/abs/1910.10045)]
* [Definitions, methods, and applications in interpretable machine learning](https://github.com/keyurfaldu/AIgrads/blob/master/summary/interpretable_ml.md) W. James Murdocha, Chandan Singhb, Karl Kumbiera, Reza Abbasi-Asl, and Bin Yua, PNAS 2019 [[PNAS](https://www.pnas.org/content/pnas/116/44/22071.full.pdf)]
* [How Can I Explain This to You? An Empirical Study of Deep Neural Network Explanation Methods](https://github.com/keyurfaldu/AIgrads/blob/master/summary/explainabilty_emperical_study.md) Jeya Vikranth Jeyakumar, Joseph Noor, Yu-Hsi Cheng, Luis Garcia, Mani Srivastava, NIPS 2020 [[arXiv](https://proceedings.neurips.cc//paper/2020/file/2c29d89cc56cdb191c60db2f0bae796b-Paper.pdf)]
* [Towards Transparent and Explainable Attention Models](https://github.com/keyurfaldu/AIgrads/blob/master/summary/diversity_attention.md) Mohankumar, Mitesh Khapra et al. ACL 2020 [[arXiv](https://arxiv.org/abs/2004.14243)]
* [A Framework for Understanding Unintended Consequences of Machine Learning](https://github.com/keyurfaldu/AIgrads/blob/master/summary/formalising_bias.md) Harini Suresh, John V. Guttag, 2020 [[arXiv](https://arxiv.org/pdf/1901.10002.pdf)]

- [Explaining Explanations: Axiomatic Feature Interactions for Deep Networks](https://github.com/keyurfaldu/AIgrads/blob/master/summary/integrated_hessians.md) Janizek, Sturmfels, Lee, 2020 [[arXiv](https://arxiv.org/pdf/2002.04138.pdf)]
- [Explainable AI: A Review of Machine Learning Interpretability Methods](https://www.mdpi.com/1099-4300/23/1/18/pdf)
- [Explaining Explanations: An Overview of Interpretability of Machine Learning](https://arxiv.org/pdf/1806.00069.pdf)
- [How Important Is a Neuron?](https://github.com/keyurfaldu/AIgrads/blob/master/summary/conductance.md), Kedar Dhamdhere, Mukund Sundararajan, Qiqi Yan, Google Research [[arXiv](https://arxiv.org/pdf/1805.12233.pdf)]
- https://cloud.google.com/explainable-ai
- https://github.com/keyurfaldu/AIgrads



### Evaluations

evaluation methods for RS

* Predicting Online Performance of News Recommender Systems Through Richer Evaluation Metrics
* RecSys2018 tutorial



### System

Recommendation systems references, and hashing function for flow allocations

| Company     | Conference | Paper                                                        |
| --------- | ---------- | ------------------------------------------------------------ |
| Tencent      | SIGMOD'15     | [TencentRec: Real-time stream recommendation in practice](https://dl.acm.org/citation.cfm?id=2742785) |
| Uber       | PAPIs'16    | [Scaling Machine Learning as a Service](http://proceedings.mlr.press/v67/li17a.html) |
| Google        | KDD'17     | [TFX: A TensorFlow-Based Production-Scale Machine Learning](https://ai.google/research/pubs/pub46484) |





## Applications

### Feeds RecSys

* News
* Pictures
* Videos (PGC, UGC)
* Musics
* e-commerce
* financial products





### Computational Ad

* Computational advertising systems





### Marketing Growth

* CVR prediction
* LTV prediction





## Reference

* 《计算广告论文、学习资料、业界分享》https://github.com/wzhe06/Ad-papers
* 《Must-read papers on Recommender System》https://github.com/hongleizhang/RSPapers
* 《A review and evaluation of CTR prediction models》https://github.com/anyai/OpenCTR 
* 《Easy-to-use,Modular and Extendible package of deep-learning based CTR models》https://github.com/shenweichen/DeepCTR 
* 《CTR预估深度模型演化之路》https://mp.weixin.qq.com/s/jpWS9ec0MCO4ncSZx38r3w

* Conferences: AAAI、KDD、ICML、WWW、RecSys
