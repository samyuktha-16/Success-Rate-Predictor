# Success Rate Predictor

## Overview
The Success Rate Predictor project uses machine learning techniques to evaluate and predict the success of startup ideas. It integrates TF-IDF for text feature extraction, Latent Dirichlet Allocation (LDA) for topic modeling, and Gradient Boosting for classification to analyze textual data and provide actionable insights.

## Features
- **Data Collection & Preprocessing**: Includes data cleaning, normalization, and preparation.
- **Model Development**: Utilizes TF-IDF, LDA, and Gradient Boosting for accurate prediction.
- **Feature Engineering**: Extracts key features and insights from textual data.
- **User Interaction**: Interface for inputting startup ideas and receiving success predictions.

## Code Explanation

### Main Components

1. **Data Collection & Preprocessing**
   - The data is loaded from a CSV file using `pandas`.
   - Text data is cleaned and preprocessed to remove stopwords.
   - Features are extracted using TF-IDF and LDA.

2. **Model Development**
   - **`LdaTransformer` Class**: Custom transformer for applying LDA topic modeling.
   - **`predict_success_rate` Function**: Function to preprocess startup ideas and predict their success using the trained model.

3. **Pipeline Setup**
   - A pipeline is created that combines TF-IDF and LDA for feature extraction.
   - SMOTE is used to handle class imbalance.
   - Gradient Boosting is applied for classification.

4. **Script Execution**
   - **Training**: The model is trained using `pipeline.fit()`.
   - **Prediction**: Predictions are made using `pipeline.predict()` and probabilities are calculated using `pipeline.predict_proba()`.

### Key Functions

- **`preprocess_text(text)`**: Cleans and preprocesses input text by removing stopwords.
- **`LdaTransformer` Class**: Implements LDA topic modeling, with methods for fitting and transforming data.
- **`predict_success_rate(ideas, model, threshold)`**: Predicts the success rate of startup ideas based on a threshold probability.

### Running the Code

To run the code, execute the `predict_success.py` script. This script trains the model and provides a command-line interface for entering startup ideas and receiving predictions.

```bash
python predict_success.py
```
Follow the on-screen instructions to input startup ideas and view predictions.

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/yourusername/Success-Rate-Predictor.git
cd Success-Rate-Predictor
pip install -r requirements.txt
```

## Files

-predict_success.py: Main script for training and running the prediction model.

-requirements.txt: Lists the required Python packages.

-`startup_ideas.csv`: Contains startup ideas and their outcomes.

-notebooks/: Includes Jupyter notebooks for exploration and analysis.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Thanks to all contributors and resources that supported the development of this project.
