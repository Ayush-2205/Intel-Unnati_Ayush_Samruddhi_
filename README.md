# Intel-Unnati_Ayush_Samruddhi_
# Integrating Explainable AI Techniques for Anomaly Detection in Encrypted Traffic

##  Course Project  
This project is developed by Ayush Mohanty and Samruddhi S Shetty as part of the **Intel Unnati- AI Program** at **Marri Laxman Reddy Institute of Technology and Management** guided by [Mr.M.VishweswarReddy]. The aim was to integrate **Machine Learning** and **Explainable AI (SHAP)** techniques to detect anomalies in encrypted network traffic. 

## Introduction

In our increasingly connected digital landscape, safeguarding networks has emerged as a top priority. The surge in encrypted network traffic poses new challenges, as conventional monitoring and detection methods often struggle to spot hidden threats. Since encryption masks the actual content of data packets, detecting malicious activity becomes more complex. This project aims to address this issue by utilizing Explainable Artificial Intelligence (XAI) for identifying anomalies in encrypted network traffic. By applying advanced machine learning algorithms and leveraging SHAP (SHapley Additive exPlanations) for model interpretation, this approach uncovers patterns that improve the accuracy of anomaly detection. The ultimate objective is to strengthen cybersecurity while offering clear, interpretable insights into the reasoning behind predictions.

Explainable AI plays a crucial role in network security by shedding light on the decision-making process of models. This transparency builds trust, allowing security teams to validate predictions and respond effectively to threats. The project employs three powerful machine learning algorithms—XGBoost, Random Forest, and Gradient Boosting—while using SHAP to explain their outputs, helping security professionals understand which factors drive anomaly detection for faster, more informed decision-making.

---

## Machine Learning Algorithms Used

1. Support Vector Machine (SVM)

SVM is a classification algorithm that identifies an optimal hyperplane to separate classes. It excels in high-dimensional spaces and handles both linear and non-linear classification tasks effectively. In this context, SVM is used to distinguish normal traffic from suspicious activity by maximizing the boundary between classes.

Accuracy: 99.4%

2. Logistic Regression

Logistic Regression is a fundamental statistical technique for binary classification. It models the probability of an outcome using the logistic function. While simpler than ensemble methods, Logistic Regression provides a strong baseline for anomaly detection in network environments.

Accuracy: 98.4%

3. Perceptron

The Perceptron is a basic neural network model that establishes a linear decision boundary by adjusting weights based on misclassifications. Though limited in its ability to manage complex patterns, it offers foundational insights into anomaly detection in network traffic.

Accuracy: 92%

4. Random Forest

Random Forest is an ensemble learning algorithm that constructs multiple decision trees and merges their outputs for better accuracy. Each tree is built on random subsets of data, which reduces overfitting and variance. Its robustness makes it highly effective in detecting network anomalies, especially in noisy or complex datasets.

Accuracy: 91.5%

5. Gradient Boosting

Gradient Boosting develops models in sequence, where each successive model aims to correct the mistakes of its predecessor. Unlike Random Forest, which builds trees independently, Gradient Boosting iteratively refines predictions, making it suitable for detecting subtle irregularities in encrypted network traffic. However, this method is more computationally intensive.

Accuracy: 90.5%

6. XGBoost (Extreme Gradient Boosting)

XGBoost is a highly efficient and accurate gradient boosting technique. It incorporates parallel processing, pruning, and regularization to prevent overfitting. Its scalability makes it ideal for handling extensive datasets, which is why it is frequently used in competitive machine learning settings. In this project, XGBoost is applied to detect anomalies within encrypted traffic, delivering dependable results.

Accuracy: 90%

---

## Explainable AI and SHAP

### Explainable AI (XAI)
Explainable AI refers to techniques that make machine learning models transparent and understandable. In cybersecurity, XAI is essential for ensuring that predictions can be trusted and verified. It empowers analysts to see the rationale behind model decisions, identify biases, and make improvements where necessary.

Using XAI in encrypted traffic anomaly detection has several advantages:

• Debugging and Improvement: Helps identify weaknesses, biases, and areas for model enhancement.
• Operational Efficiency: Assists security teams in prioritizing alerts and taking timely actions.
• Transparency: Enhances trust by providing a clear understanding of model decisions.
• Accountability: Supports regulatory compliance and facilitates auditing processes.

### SHAP (SHapley Additive exPlanations)
SHAP is an XAI method grounded in cooperative game theory that assigns an importance value to each feature, explaining its contribution to a specific prediction. SHAP enhances both global and individual prediction interpretation.

#### **How SHAP Works:**
• Visualization: Provides intuitive graphs such as summary plots, force plots, and dependence plots.
• Feature Importance: Shows how each feature influences the model's outcome.
• Global and Local Interpretability: Offers a broad overview of model behavior and detailed explanations for individual cases.

#### **Applying SHAP in Network Anomaly Detection:**
- Explains the reasoning behind classifying packets as anomalous.

- Highlights important features to detect suspicious patterns.

- Demystifies complex models like XGBoost, Random Forest, and Gradient Boosting.

- Helps security teams respond faster by pinpointing the root cause of anomalies.

In this project, SHAP is integrated to explain and visualize the predictions from the models, enabling comprehensive understanding of the factors leading to abnormal network behavior.

---

## Installation

To set up the project locally, follow these steps:

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows use 'env\Scripts\activate'

# Install dependencies
pip install -r requirements.txt
```

---

## Dependencies

Ensure you have the following libraries installed:
- Python 3.8+
- pandas
- numpy
- xgboost
- scikit-learn
- shap
- matplotlib
- seaborn

Install everything using:

pip install -r requirements.txt

---

## How to Use the Project

1. Model Training: Run the corresponding script to train each machine learning model on your dataset.

2.  Model Evaluation: Assess each model’s performance using accuracy, precision, recall, and F1-score.

3.  Generate Explanations: Use SHAP to visualize and interpret model predictions.

Example commands:

python train_xgboost.py
python train_random_forest.py
python train_gradient_boosting.py

---

## Conclusion

By combining Explainable AI with anomaly detection for encrypted network traffic, this project enhances both the effectiveness and transparency of cybersecurity measures. Through SHAP, security professionals gain actionable insights into model behavior, enabling faster detection, better threat response, and stronger defense mechanisms against evolving cyber threats.
