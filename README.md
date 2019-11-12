# NLP-papers
some NLP papers to share

不固定分享一些不错的NLP方向论文

摘要
学习一个领域一般成系列地去看相关论文比较好。在没有什么头绪的时候，一般优秀的综述文章是一个很好的切入口。
这种文章一般会按时序或枝干体系来进行介绍，读者能从里面感受到领域的发展变化过程，也能直接从其参考文献中找到所需的论文。
本文记录一些NLP领域比较好的综述文章供读者学习研究（比较成熟的 子领域就不说了，这里主要介绍NLP领域内几个尚需继续更好地解决的子领域和一些较新较好的综述文）

1.零样本学习【这个其实不管是CV还是NLP领域其实都在研究，也都待进一步解决】
2019年来自新加坡南洋理工大学的综述长文【这篇文章强烈推荐】
Wei Wang, Vincent W. Zheng, Han Yu, and Chunyan Miao.(2019). A Survey of Zero-Shot Learning: Settings, Methods, and Applications. ACM Trans. Intell. Syst. Technol.10, 2, Article 13 (January 2019), 37 pages.

本文自己也写过一篇零样本的总数文章，可以参考下。（一种解决范式）
https://zhuanlan.zhihu.com/p/82397000

2 小样本学习【这个其实不管是CV还是NLP领域其实都在研究，也都待进一步解决】
来自港科大和第四范式的Few-shot learning综述长文：Generalizing from a Few Examples: A Survey on Few-Shot Learning
https://arxiv.org/abs/1904.05046

3.迁移学习
参考 最佳目录呀(总体记录网址)：
http://transferlearning.xyz/
http://transferlearning.xyz/#3theory-and-survey-%E7%90%86%E8%AE%BA%E4%B8%8E%E7%BB%BC%E8%BF%B0

（1）迁移学习领域最具代表性的综述是A survey on transfer learning，杨强老师署名的论文，虽然比较早，发表于2009-2010年，对迁移学习进行了比较权威的定义。 – The most influential survey on transfer learning.
Pan, S. J., & Yang, Q. (2009). A survey on transfer learning. IEEE Transactions on knowledge and data engineering, 22(10), 1345-1359.

另外还有一些比较新的综述Latest survey：

用transfer learning进行sentiment classification的综述：A Survey of Sentiment Analysis Based on Transfer Learning
Liu, R., Shi, Y., Ji, C., & Jia, M. (2019). A Survey of Sentiment Analysis Based on Transfer Learning. IEEE Access, 7, 85401-85412.

2019 一篇新survey：Transfer Adaptation Learning: A Decade Survey
Zhang, L. (2019). Transfer Adaptation Learning: A Decade Survey. arXiv preprint arXiv:1903.04687.

2019 一篇新survey：A Comprehensive Survey on Transfer Learning：https://arxiv.org/pdf/1911.02685.pdf

2018 一篇迁移度量学习的综述: Transfer Metric Learning: Algorithms, Applications and Outlooks
Luo, Y., Wen, Y., Duan, L., & Tao, D. (2018). Transfer metric learning: Algorithms, applications and outlooks. arXiv preprint arXiv:1810.03944.

2017-2018 一篇最近的非对称情况下的异构迁移学习综述：Asymmetric Heterogeneous Transfer Learning: A Survey
Friedjungová, M., & Jirina, M. (2017). Asymmetric Heterogeneous Transfer Learning: A Survey. In DATA (pp. 17-27).

2018 Neural style transfer的一个survey：Neural Style Transfer: A Review
Jing, Y., Yang, Y., Feng, Z., Ye, J., Yu, Y., & Song, M. (2019). Neural style transfer: A review. IEEE transactions on visualization and computer graphics.

2018 深度domain adaptation的一个综述：Deep Visual Domain Adaptation: A Survey
Wang, M., & Deng, W. (2018). Deep visual domain adaptation: A survey. Neurocomputing, 312, 135-153.

2017 多任务学习的综述，来自香港科技大学杨强团队：A survey on multi-task learning
Zhang, Y., & Yang, Q. (2017). A survey on multi-task learning. arXiv preprint arXiv:1707.08114.

2017 异构迁移学习的综述：A survey on heterogeneous transfer learning
Day, O., & Khoshgoftaar, T. M. (2017). A survey on heterogeneous transfer learning. Journal of Big Data, 4(1), 29.

2017 跨领域数据识别的综述：Transfer learning for cross-dataset recognition: a survey
Zhang, J., Li, W., & Ogunbona, P. (2017). Transfer learning for cross-dataset recognition: a survey. arXiv preprint arXiv:1705.04396.

2016 A survey of transfer learning。其中交代了一些比较经典的如同构、异构等学习方法代表性文章。
Weiss, K., Khoshgoftaar, T. M., & Wang, D. (2016). A survey of transfer learning. Journal of Big data, 3(1), 9.

另外这个领域 戴老板的论文也是非常有必要读的（非综述，个人强推）
戴文渊. (2009). 基于实例和特征的迁移学习算法研究 (Doctoral dissertation, 上海: 上海交通大学).

4.弱监督学习
这个比较推荐南京大学周志华老师的综述论文
Zhou, Z. H. (2017). A brief introduction to weakly supervised learning. National Science Review, 5(1), 44-53.
中文介绍网址：https://www.jiqizhixin.com/articles/2018-03-05


5.预训练模型
2019 google的T5模型论文，把它当成综述来看就介绍的挺好：
Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2019). Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683.

2018 elmo：ELMO是“Embedding from Language Models”的简称，其论文题目：“Deep contextualized word representation”。这篇论文来自华盛顿大学的工作，最后是发表在今年的 NAACL 会议上，并获得了最佳论文。
Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep contextualized word representations. arXiv preprint arXiv:1802.05365.

2018 OpenAI GPT&GPT-2：Improving Language Understanding by Generative Pre-Training
Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training. URL https://s3-us-west-2. amazonaws. com/openai-assets/researchcovers/languageunsupervised/language understanding paper. pdf.

2018 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding。
Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
这篇论文把预训练语言表示方法分为了基于特征的方法（代表 ELMo）和基于微调的方法（代表 OpenAI GPT）。

bert后还有一些改进模型比如华为刘群/百度的ERNIE,XLNet等相关文章


6 其他方向，还有一些比较新的不同方向的综述文：
集中记录网址：参考：https://www.zhuanzhi.ai/vip/df6ba5fa1b7975e17bae0504a3ddbade

注意力机制：Hu, D. (2019, September). An introductory survey on attention mechanisms in nlp problems. In Proceedings of SAI Intelligent Systems Conference (pp. 432-448). Springer, Cham.

注意力机制：Chaudhari, S., Polatkan, G., Ramanath, R., & Mithal, V. (2019). An attentive survey of attention models. arXiv preprint arXiv:1904.02874.

Elvis Saravia and Soujanya：PoriaElvis Saravia and Soujanya Poria：NLP方方面面都有涉及，颇有一些横贯全局的意思。
网址：https://nlpoverview.com/index.html

情感分析领域：Zhang, L., Wang, S., & Liu, B. (2018). Deep learning for sentiment analysis: A survey. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 8(4), e1253.

