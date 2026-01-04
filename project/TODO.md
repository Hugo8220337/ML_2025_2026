# 16/01/2026 - 18:00

* [x] Fully test EMS
* [x] Implement Naive Bayes
* [x] Implement DBSCAN and GMM
* [x] Implement Dimensionality reduction
* [ ] Study use case for isolation forest
* [ ] Study use case for elastic net instead of linear regression
* [ ] Study use case for CNN, LSTM and Bi-LSTM
* [ ] Streamlit front end [Implementation Plan](Dashboard.md)
* [ ] Deploy
* [ ] Write documentation <!-- ahahahahahhahahaha  -->


# IMPORTANT

* [x] Fix preprocessing leakage
* [x] Let ems manage preprocessing
* [x] Try to improve NLP
* [x] Remember to remove hardcoded variable from debugs
* [x] Improve caching
* [x] Support unsupervised algorithms in EMS
* [x] Support dimensionality reduction in EMS
* [ ] Option to skip full data training



🔴 Broken Inference Pipeline (Deployment Blocker) The ems function returns the best classifier model, but it discards the fitted vectorizer and dimensionality_reducer.

* Why this matters: When you eventually deploy this (as mentioned in your TODO.md), you will have a trained classifier (e.g., SVM) that expects input vectors of size 50. However, you won't have the specific TF-IDF vectorizer or PCA model used to create those inputs. You cannot process new user input without them.

* Recommendation: Your ems return object needs to include the entire pipeline (Preprocessor -> Vectorizer -> Reducer -> Classifier), or you need to use sklearn.pipeline.Pipeline objects to bundle them together.