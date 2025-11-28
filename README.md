# Machine Learning Repository

A comprehensive collection of machine learning tutorials, exercises, and implementations covering fundamental to advanced topics in Python.

## Overview

This repository contains hands-on tutorials and exercises for various machine learning algorithms and techniques. Each topic includes detailed explanations, code examples, and practice exercises to help you understand and implement machine learning concepts.

## Topics Covered

### Core Machine Learning Algorithms

1. **Linear Regression**
   - Simple linear regression with one variable
   - Multivariate linear regression
   - Implementation examples with real-world datasets

2. **Gradient Descent**
   - Understanding optimization algorithms
   - Implementation from scratch
   - Practical exercises

3. **Model Persistence**
   - Saving and loading models using Pickle
   - Using Joblib for model serialization
   - Best practices for model deployment

4. **Data Preprocessing**
   - One-hot encoding for categorical variables
   - Train-test split techniques
   - Feature engineering fundamentals

### Classification Algorithms

5. **Logistic Regression**
   - Binary classification problems
   - Multiclass classification
   - Real-world applications (insurance, HR data)

6. **Decision Trees**
   - Tree-based classification
   - Understanding splits and nodes
   - Exercises with Titanic dataset

7. **Support Vector Machines (SVM)**
   - Classification with SVM
   - Kernel functions
   - Digit recognition exercises

8. **Random Forest**
   - Ensemble learning basics
   - Random forest implementation
   - Feature importance analysis

9. **K-Nearest Neighbors (KNN)**
   - Instance-based learning
   - Distance metrics
   - Classification examples

10. **Naive Bayes**
    - Probabilistic classification
    - Spam email filtering
    - Titanic survival prediction

### Advanced Techniques

11. **K-Fold Cross Validation**
    - Model evaluation techniques
    - Preventing overfitting
    - Performance metrics

12. **K-Means Clustering**
    - Unsupervised learning
    - Clustering algorithms
    - Customer segmentation

13. **Grid Search**
    - Hyperparameter tuning
    - Cross-validation with grid search
    - Model optimization

14. **Regularization**
    - L1 and L2 regularization
    - Ridge and Lasso regression
    - Preventing overfitting

15. **Principal Component Analysis (PCA)**
    - Dimensionality reduction
    - Feature extraction
    - Visualization techniques

16. **Bagging**
    - Bootstrap aggregating
    - Ensemble methods
    - Diabetes and heart disease prediction

### Feature Engineering

17. **Outlier Detection**
    - Percentile-based methods
    - Z-score method
    - Interquartile Range (IQR) method
    - Handling outliers in real datasets

## Repository Structure

```
Machine-Learning/
├── 1_linear_reg/                    # Linear regression basics
├── 2_linear_reg_multivariate/      # Multiple features regression
├── 3_gradient_descent/              # Optimization algorithms
├── 4_save_model/                    # Model persistence
├── 5_one_hot_encoding/              # Categorical encoding
├── 6_train_test_split/              # Data splitting
├── 7_logistic_reg/                  # Binary classification
├── 8_logistic_reg_multiclass/       # Multiclass classification
├── 9_decision_tree/                 # Decision tree algorithms
├── 10_svm/                          # Support Vector Machines
├── 11_random_forest/                # Random Forest ensemble
├── 12_KFold_Cross_Validation/       # Cross-validation techniques
├── 13_kmeans/                       # K-means clustering
├── 14_naive_bayes/                  # Naive Bayes classifier
├── 15_gridsearch/                   # Hyperparameter tuning
├── 16_regularization/               # Regularization techniques
├── 17_knn_classification/           # K-Nearest Neighbors
├── 18_PCA/                          # Principal Component Analysis
├── 19_Bagging/                      # Bagging ensemble method
└── FeatureEngineering/              # Feature engineering techniques
    ├── 1_outliers/                  # Percentile-based outliers
    ├── 2_outliers_z_score/          # Z-score method
    └── 3_outlier_IQR/               # IQR method
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook
- Required Python packages (see requirements.txt)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/balu548411/Machine-Learning.git
cd Machine-Learning
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Navigate to any topic folder and open the tutorial notebooks to get started.

## Requirements

All required packages are listed in `requirements.txt`. Key dependencies include:

- NumPy: Numerical computing
- Pandas: Data manipulation and analysis
- Matplotlib: Data visualization
- Scikit-learn: Machine learning algorithms and utilities
- Jupyter: Interactive notebook environment

## Usage

Each topic folder contains:
- Tutorial notebooks with detailed explanations
- Exercise notebooks for hands-on practice
- Sample datasets for experimentation
- Solution files where applicable

### Recommended Learning Path

1. Start with **Linear Regression** to understand the basics
2. Learn **Gradient Descent** for optimization
3. Master **Data Preprocessing** techniques
4. Explore **Classification Algorithms** (Logistic Regression, Decision Trees, etc.)
5. Dive into **Advanced Techniques** (Cross-validation, Grid Search, Regularization)
6. Practice **Feature Engineering** for real-world applications

## Datasets

This repository includes various datasets for practice:
- Home prices data
- Iris dataset
- Titanic dataset
- Insurance data
- HR data
- Heart disease data
- Diabetes data
- And more...

## Exercises

Most topics include exercise notebooks to reinforce learning:
- Practice problems with real datasets
- Step-by-step solutions
- Additional challenges to test your understanding

## Contributing

Contributions are welcome! If you'd like to improve this repository:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This repository is open source and available for educational purposes.

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)

## Author

Created as a comprehensive learning resource for machine learning enthusiasts.

---

Happy Learning!
