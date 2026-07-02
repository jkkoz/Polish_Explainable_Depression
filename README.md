# Can Machines Explain Depression? Explainable Machine Learning for Depression Detection in Polish Texts

## Abstract
The diagnosis of depression in Poland can be seen as a problematic process. Since it has been shown that the language of individuals changes based on their mental health, machine learning models can be utilized to differentiate between depressed and non-depressed people. Despite major works in this area, there is a lack of research on the Polish language. This study addressed this by collecting and labeling Polish textual data from social media, creating a gold-standard dataset validated by Poles with a background in psychology. Various machine learning methods were compared to confirm that the same performance as that of their English-based counterparts can be achieved. Ensemble methods performed best, achieving an accuracy of 72\% and ROC-AUC of 74\%. It was confirmed that self-focus language is the biggest discriminator between the two classes, also among Polish users. 

Subjective evaluation with 11 valid participants was performed. Psychology experts and non-experts assessed the Understandability, Trust, Decision-making, and Actionability of three different explanation methods of the classifier's predictions. The results suggested that textual explanations are the clearest, both to an average person and to a psychology specialist. Actionability showed that professionals are willing to change their minds about a diagnosis upon seeing the explanations, however, they need further information about how these explanations can be integrated into the diagnostic process to trust the machines. 

## Repository Description

This repository contains all the working files for this research project.

**Data Collection** provides code for how data was scraped and cleaned along with the full dataset.     
**Sentence Transformers** shows how data was preprocessed and used for training with Sentence Transformenrs.     
**Feature Vector Construction** highlights how diacritis were restored, linguistical features collected, and how sentiment, emoji and aboslutist words were extracted as well.     
**Models with Feature Vector** utilize that feature vector to compare Linear SVM, Logistic Regression, RBF SVM and AutoGluon's ensemble models when trained on this feature vector (or a subset of features). Inspection of feature importance is also present in the code.     
**Explanations** provide two different ways of generating explanations: SHAP for the ensemble models and Counterfactuals for Logistic Regression.      

#### This project was created as a Computer Science Bachelor Thesis at Vrije Universiteit Amsterdam (2026).
