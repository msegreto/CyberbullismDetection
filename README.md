# Cyberbullying Detection System

## English

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

Academic Disclaimer

This project was developed as part of university coursework for academic, research, and educational purposes.

The repository explores machine-learning techniques for cyberbullying detection and may contain datasets or examples including offensive, abusive, or otherwise sensitive language.

The models, classifications, and results produced by this project are experimental and should not be considered suitable for automated moderation, profiling, or other production use without further review and validation.

The software is provided "as is", without warranty of any kind. See the MIT License for details.

---
