
![updt wllpth](https://github.com/user-attachments/assets/033addb6-e04c-40ff-9a6e-d81b19cc88bf)


**Project Summary**

This project simulates a real-world health-tech deployment scenario where insufficient medical data required the creation of a structured synthetic dataset to enable machine learning–based disease classification.

I worked within the Data Science team supporting the development of an AI-assisted symptom triage system for WellaPath, a digital health platform serving underserved populations in Nigeria.

The objective was to classify patient symptom profiles into 7 common diseases using supervised ML models and determine the most reliable algorithm for downstream deployment.

**Link**: https://wellapath.org/?fbclid=PAdGRleANObFBleHRuA2FlbQIxMQABp-m21CedaMOhiGMr-ESSFtHAaar9upTO22khbRkPQ-9FLpqtVM8rIlpKpMju_aem_EVfWsgLsmwh_mtZ83BRCng


**Problem Context**

Healthcare access in low-resource environments suffers from:

1. Lack of structured patient records

2. Delayed diagnosis

3. Symptom overlap across diseases

4. Data scarcity for ML training

Because collected data was insufficient, I designed and used a synthetic data generation pipeline reflecting real Nigerian disease prevalence and symptom distributions to train robust models.

Dataset used: Synthetic data generator design (attached)

**Dataset Description**

1. 7 diseases: Malaria, Pneumonia, Typhoid Fever, Measles, Lassa Fever, Influenza, Diarrheal Disease

2. Binary symptom encoding (fever, cough, rash, vomiting, diarrhea, etc.)

3. Probabilistic symptom assignment per disease

4. Noise injection for realism (2%)

5. Balanced to reflect realistic but imperfect health data
   
Each row represents one simulated patient.

**Modeling Approach**

Six classification models were trained and compared:
| Model               | Accuracy | Why It Matters                                    |
| ------------------- | -------- | ------------------------------------------------- |
| Logistic Regression | **0.85** | Strong baseline, interpretable for medical logic  |
| SVM                 | 0.84     | Handles complex boundaries well                   |
| Random Forest       | 0.82     | Robust to noise & non-linear symptom interactions |
| KNN                 | 0.81     | Sensitive to overlapping symptom clusters         |
| Decision Tree       | 0.80     | Overfit-prone but interpretable                   |
| Naive Bayes         | 0.80     | Limited by independence assumptions               |


  (For Full model evaluation and confusion matrix interpretation see Model Performance Report)
  

Confusion matrix visualization used in the analysis:

![Matrix](https://github.com/user-attachments/assets/d539e17d-110f-4de3-854f-111bc7f74806)

**Importance of Model Accuracies**

Accuracy was critical because:

1. Misclassification in healthcare = wrong triage recommendations
2. High accuracy ensures reliable early guidance before referral
3. Demonstrates model robustness even on synthetic but noisy data
4. Helps select models safe enough for real-world digital health support

Why Logistic Regression and Random Forest stood out:

1. Logistic Regression → transparent decision logic (important for health trust)
2. Random Forest → reduced noise impact and better generalization

Rare disease detection challenge:
Models performed best on common diseases (Malaria/Pneumonia) but struggled more on low-prevalence classes like Measles and Lassa Fever — highlighting real-world class imbalance issues in healthcare AI.

**Summary**

This project represents a real-world health AI deployment scenario where limited clinical data required the design of a domain-informed synthetic dataset to enable machine learning–based disease classification for underserved Nigerian communities. I engineered symptom-driven data, trained and benchmarked six classifiers, and evaluated them using confusion matrices to identify models suitable for reliable triage support. Logistic Regression (85%) and SVM (84%) delivered the highest accuracy, while Random Forest (82%) was selected as the most robust for deployment due to its stability under noisy, non-organic data.

⚠ **Disclaimer**

Synthetic data included.
Models built for educational and research demonstration. Not medical advice.






