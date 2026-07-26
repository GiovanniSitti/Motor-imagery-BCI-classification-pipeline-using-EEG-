# Motor-imagery-BCI-classification-pipeline-using-EEG-
This project focuses on the classification of motor imagery EEG signals using an ML based pipeline.                                      
The code processes EEG data from a BCI dataset and applies band-pass filtering in the sensorimotor rhythm frequency range.
Training epochs are extracted around motor imagery events and then used to train a Common Spatial Pattern model.
CSP is used to identify spatial filters that maximize the variance difference between the two motor imagery classes.
The extracted spatial features are classified using Linear Discriminant Analysis with automatic shrinkage regularization.
The pipeline also includes a pseudo-online testing procedure, where predictions are generated over sliding windows of EEG data.
Final performance metrics such as accuracy, sensitivity, specificity, balanced accuracy, and misclassification rate are saved for evaluation.
