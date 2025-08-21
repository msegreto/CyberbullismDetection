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

---

## Italiano

### Introduzione
Il fenomeno del cyberbullismo è in costante crescita a causa della diffusione globale dei social network. Questo progetto propone un sistema di rilevamento del cyberbullismo basato sull’analisi di tweet, con una pipeline che integra **classificazione in due stadi** e un **modulo di spiegabilità** pensato per aumentare fiducia e trasparenza nelle decisioni del modello.  

### Metodologia
1. **Preprocessing dei dati**  
   Pulizia del testo (minuscolizzazione, rimozione di URL/menzioni/punteggiatura, stopword, stemming) e creazione di etichette binarie per distinguere contenuti sicuri da quelli offensivi.  

2. **Bilanciamento del dataset**  
   Analisi della distribuzione delle classi e utilizzo di tecniche di data augmentation (sostituzione con sinonimi) per mantenere la naturalezza linguistica senza introdurre rumore artificiale.  

3. **Estrazione delle feature**  
   - **Bag of Words** per una rappresentazione semplice e frequenziale.  
   - **TF-IDF** per valorizzare i termini più discriminativi.  
   - **Word2Vec addestrato ad hoc** su dataset affini a cyberbullismo e contenuti offensivi (≈120k frasi pre-processate), così da catturare meglio relazioni semantiche e sintattiche del dominio.  

4. **Classificazione in due stadi**  
   - **Stadio 1**: rilevamento cyberbullismo vs non-cyberbullismo.  
   - **Stadio 2**: classificazione fine-grained per identificare il tipo specifico (età, genere, religione, ecc.).  
   L’uso della validazione annidata (nested cross-validation) assicura iperparametri ottimali e valutazioni non distorte.  

5. **Modulo di Explainability (focus principale)**  
   - **Globale**:  
     - Analisi delle feature importance del Random Forest.  
     - Pattern mining con itemset chiusi e massimali per scoprire strutture linguistiche ricorrenti nelle classi di cyberbullismo.  
   - **Locale**:  
     - Uso di TreeInterpreter per scomporre le decisioni a livello di singolo tweet, evidenziando parole presenti e **assenze significative** che influenzano la classificazione.  
   Questo approccio bilancia comprensione ad alto livello (pattern e feature globali) con interpretabilità puntuale (analisi di predizioni specifiche).  

6. **Interfaccia Grafica (GUI)**  
   Applicazione Tkinter con due viste:  
   - Classificazione (output binario/multiclasse).  
   - Spiegazione (feature contribuenti, distribuzione parole, itemset rappresentativi).  

### Punti di Forza
- Embedding **Word2Vec custom** mirato al dominio.  
- Pipeline modulare e scalabile.  
- **Approccio a due stadi**, realistico e interpretabile.  
- **Explainability integrata**, con metodi globali e locali che garantiscono trasparenza.  
- GUI intuitiva e accessibile.  

### Disclaimer
Il progetto utilizza testi contenenti termini offensivi o discriminatori.  
Durante il preprocessing e la documentazione, la maggior parte delle parole offensive è stata **oscurata o mascherata**, ma qualcuna potrebbe essere rimasta in chiaro.  
Chi fosse particolarmente sensibile a questo tipo di contenuti è pregato di tenere presente che essi potrebbero comunque comparire nei dataset, nelle visualizzazioni o nelle spiegazioni.  
L’inclusione di tali termini ha esclusivamente finalità di ricerca e studio e **non riflette in alcun modo le opinioni dell’autore**.  
