# Cyberbullying Detection System

### Introduction
Cyberbullying is a rapidly growing issue fueled by the expansion of social networks. This project introduces a detection system based on tweets, combining a **two-stage classification pipeline** with an **explainability module** to ensure trust and transparency in the model’s decisions.  

### Methodology
1. **Data Preprocessing**  
   Text cleaning (lowercasing, removal of URLs/mentions/punctuation, stopwords, stemming) and binary label creation for safe vs abusive content.  

2. **Dataset Rebalancing**  
   Distribution analysis of classes and adoption of text augmentation (synonym replacement) to handle imbalance without generating unnatural text.  

3. **Feature Extraction**  
   - **Bag of Words** for frequency-based representation.  
   - **TF-IDF** to emphasize discriminative tokens.  
   - **Custom-trained Word2Vec** on domain-related corpora (≈120k preprocessed sentences on cyberbullying and offensive content), enhancing the semantic and syntactic understanding of abusive language.  

4. **Two-Stage Classification**  
   - **Stage 1**: Binary detection (cyberbullying / non-cyberbullying).  
   - **Stage 2**: Fine-grained classification (age, gender, religion, etc.).  
   Nested cross-validation was employed for robust hyperparameter optimization and unbiased evaluation.  

5. **Explainability Module (main focus)**  
   - **Global**:  
     - Feature importance analysis from Random Forest.  
     - Pattern mining (closed and maximal itemsets) to uncover recurring linguistic structures across classes.  
   - **Local**:  
     - TreeInterpreter for word-level contribution analysis, highlighting both **present and absent features** that shaped the prediction.  
   This combination ensures both a global perspective on model behavior and fine-grained explanations at the tweet level.  

6. **Graphical User Interface (GUI)**  
   Built with Tkinter, featuring:  
   - Classification view (binary + multiclass outputs).  
   - Explanation view (feature contributions, word distributions, representative itemsets).  

### Strengths
- **Domain-specific Word2Vec embeddings** for richer representations.  
- Modular and scalable pipeline.  
- **Two-stage classification** for improved realism and interpretability.  
- **Strong explainability integration**, combining global insights and local instance-level transparency.  
- User-friendly GUI accessible to non-technical users.  

### Disclaimer
This project deals with texts containing offensive or discriminatory language.  
During preprocessing and documentation, most offensive terms were **obfuscated or masked**, but some may still appear in clear text.  
If you are sensitive to such content, please be aware that it may occasionally emerge in the dataset, visualizations, or explanations.  
The inclusion of these terms is solely for research and educational purposes, and does not reflect the views or intentions of the author. 

## Third-Party Datasets

This project uses third-party datasets for research and educational purposes. These datasets are not authored by the repository owner and are not covered by the MIT License that applies to the source code of this repository.

### Cyberbullying Classification

- **Dataset:** Cyberbullying Classification
- **Source:** Kaggle — Larxel (Andrew MVD)
- **File:** `cyberbullying_tweets.csv`
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Original research reference:** J. Wang, K. Fu, C. T. Lu, *SOSNet: A Graph Convolutional Network Approach to Fine-Grained Cyberbullying Detection*, IEEE BigData 2020.

### Cyberbully Detection Dataset

- **Dataset:** Cyberbully Detection Dataset
- **Source:** Kaggle
- **File:** `cb_multi_labeled_balanced.csv`
- **Authors:** Mohamad Ahmadinejad, Nashid Shahriar, Lisa Fan
- **License:** The Kaggle dataset page currently does not specify a license.

This dataset remains subject to the terms and rights of its original authors and provider and is not covered by this repository's MIT License.

### HateXplain-derived Dataset

- **Dataset:** CyberBullying Detection Dataset / HateXplain-derived data
- **Source:** Kaggle
- **File:** `final_hateXplain.csv`
- **Original dataset:** HateXplain

Original reference:

Binny Mathew, Punyajoy Saha, Seid Muhie Yimam, Chris Biemann, Pawan Goyal, and Animesh Mukherjee, *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*, Proceedings of the AAAI Conference on Artificial Intelligence, 2021.

The dataset and any derived data remain subject to the terms and attribution requirements of their respective original sources.

## Academic Disclaimer

This project was developed as part of university coursework for academic, research, and educational purposes.

The repository explores machine-learning techniques for cyberbullying detection and may contain datasets or examples including offensive, abusive, or otherwise sensitive language.

The models, classifications, and results produced by this project are experimental and should not be considered suitable for automated moderation, profiling, or other production use without further review and validation.

The software is provided "as is", without warranty of any kind. See the MIT License for details.

---
