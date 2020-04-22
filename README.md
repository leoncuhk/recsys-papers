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
| Item2Vec | arxiv'16   | [Item2Vec: Neural Item Embedding for Collaborative Filtering](https://arxiv.org/ftp/arxiv/papers/1603/1603.04259.pdf)|
| Node2Vec | KDD'16   | [node2vec: Scalable Feature Learning for Networks](https://cs.stanford.edu/people/jure/pubs/node2vec-kdd16.pdf) |
| Youtube DNN | RecSys'16| [Deep Neural Networks for YouTube Recommendations](https://ai.google/research/pubs/pub45530) [**Google**]|
| NCF | WWW'17   | [Neural Collaborative Filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf) |
| Listing emb | KDD'18 | [Real-time Personalization using Embeddings for Search Ranking at Airbnb](https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb) [**Airbnb**] |
| Product emb | arxiv'19   | [Large-scale Collaborative Filtering with Product Embeddings](https://arxiv.org/abs/1901.04321) [**Amazon**] |
| DeepCF | AAAI'19 | [DeepCF: A Unified Framework of Representation Learning and Matching Function Learning in Recommender System](https://arxiv.org/abs/1901.04704) |




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



### DRL based

* Deep Reinforcement Learning based Recommendation Systems


| Conference | Paper                                                        |
| ---------- | ------------------------------------------------------------ |
| WWW'18   | [DRN：A Deep Reinforcement Learning Framework for News Recommendation](https://dl.acm.org/citation.cfm?id=3185994) [**MSRA**]|
| RecSys'18 | [Deep Reinforcement Learning for Page-wise Recommendations](https://dl.acm.org/citation.cfm?id=3240374) [**JD**] |
| Arxiv'17 | [Deep Reinforcement Learning for List-wise Recommendations](https://arxiv.org/abs/1801.00209) [**JD**] |
| KDD'18 | [Stabilizing Reinforcement Learning in Dynamic Environment with Application to Online Recommendation](https://arxiv.org/abs/1801.00209) |
| KDD'18 | [Reinforcement Learning to Rank in E-Commerce Search Engine: Formalization, Analysis, and Application](https://dl.acm.org/citation.cfm?id=3219846) |
|  | https://tech.meituan.com/2018/11/15/reinforcement-learning-in-mt-recommend-system.html [**Meituan**] |



### Evaluations

evaluation methods for RS

* Predicting Online Performance of News Recommender Systems Through Richer Evaluation Metrics
* RecSys2018 tutorial



### XAI

Explainable AI and model interpretation methods for ML models

* PDP
* ICE
* SHAP
* LIME
* RETAIN
* LRP



### System

Recommendation systems references, and hashing function for flow allocations

| Company     | Conference | Paper                                                        |
| --------- | ---------- | ------------------------------------------------------------ |
| Tencent      | SIGMOD'15     | [TencentRec: Real-time stream recommendation in practice](https://dl.acm.org/citation.cfm?id=2742785) |
| Uber       | PAPIs'16    | [Scaling Machine Learning as a Service](http://proceedings.mlr.press/v67/li17a.html) |
| Google        | KDD'17     | [TFX: A TensorFlow-Based Production-Scale Machine Learning](https://ai.google/research/pubs/pub46484) |



### Computational Ad

Computational advertising systems

* 



## Reference

* 《计算广告论文、学习资料、业界分享》https://github.com/wzhe06/Ad-papers
* 《Must-read papers on Recommender System》https://github.com/hongleizhang/RSPapers
* 《A review and evaluation of CTR prediction models》https://github.com/anyai/OpenCTR 
* 《Easy-to-use,Modular and Extendible package of deep-learning based CTR models》https://github.com/shenweichen/DeepCTR 
* 《CTR预估深度模型演化之路》https://mp.weixin.qq.com/s/jpWS9ec0MCO4ncSZx38r3w
