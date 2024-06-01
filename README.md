
# Deep Learning Challenge: Predicting Successful Funding Applicants for Alphabet Soup

## Background

Alphabet Soup, a nonprofit foundation, aims to identify applicants with the best chance of success for funding their ventures. Using historical data of over 34,000 organizations, we developed a binary classifier to predict the success of these applicants. The dataset includes metadata such as application type, affiliated sector, use case for funding, and more.

## Repository

The repository for this project is named [deep-learning-challenge](https://github.com/yourusername/deep-learning-challenge).

## Project Structure

- **Credit_Risk** (directory)
  - `credit_risk_classification.ipynb`
  - `lending_data.csv`
- `README.md`

## Skills Used

- pandas, scikit-learn, StandardScaler, pd.get_dummies, train_test_split, TensorFlow, Keras, neural networks, binary classification, data preprocessing, feature engineering, model training, model evaluation, model optimization, callbacks, HDF5 file handling, Google Colab

## Instructions

### Step 1: Preprocess the Data

1. **Read the Data**: Load `charity_data.csv` into a Pandas DataFrame.
2. **Identify Target and Features**: 
   - Target: `IS_SUCCESSFUL`
   - Features: All other columns excluding `EIN` and `NAME`.
3. **Drop Unnecessary Columns**: Remove `EIN` and `NAME`.
4. **Analyze Unique Values**: Determine the number of unique values for each column.
5. **Combine Rare Categories**: For columns with more than 10 unique values, combine rare categories.
6. **Encode Categorical Variables**: Use `pd.get_dummies()` to one-hot encode categorical variables.
7. **Split the Data**: Split into features array `X` and target array `y`, then into training and testing datasets using `train_test_split`.
8. **Scale the Data**: Use `StandardScaler` to scale the training and testing feature datasets.

### Step 2: Compile, Train, and Evaluate the Model

1. **Design the Neural Network**:
   - Determine the number of input features and nodes.
   - Create the first hidden layer with an appropriate activation function.
   - Add a second hidden layer if necessary.
   - Create an output layer with an appropriate activation function.
2. **Compile and Train**:
   - Compile the model.
   - Create a callback to save the model's weights every five epochs.
   - Train the model and evaluate its performance.
3. **Save the Model**: Export the results to an HDF5 file named `AlphabetSoupCharity.h5`.

### Step 3: Optimize the Model

1. **Optimize for Accuracy**: Aim for a target predictive accuracy higher than 75% using the following techniques:
   - Adjust input data (drop columns, create bins for rare occurrences, adjust bin values).
   - Modify the model (add neurons, add hidden layers, change activation functions, adjust epochs).
2. **Repeat Preprocessing**: Adjust preprocessing steps based on modifications.
3. **Redesign the Neural Network**: Modify the network structure for better performance.
4. **Save Optimized Model**: Export results to `AlphabetSoupCharity_Optimization.h5`.

### Step 4: Write a Report

1. **Overview**: Explain the purpose of the analysis.
2. **Data Preprocessing**:
   - Identify targets and features.
   - Remove non-relevant variables.
3. **Model Details**:
   - Specify the number of neurons, layers, and activation functions.
   - Discuss model performance and optimization attempts.
4. **Summary**: Summarize results and recommend alternative models.

### Step 5: Submit Your Work

1. **Download Colab Notebooks**: Save notebooks to your computer.
2. **Move Files to Repository**: Place them in the `Deep Learning Challenge` directory.
3. **Push to GitHub**: Commit and push your changes to GitHub.

## Resources

- Documentation: TensorFlow, Keras, pandas, scikit-learn
- Community Help: StackOverflow, tutoring
- Previous Assignments: Data preprocessing, machine learning basics, neural networks

---
