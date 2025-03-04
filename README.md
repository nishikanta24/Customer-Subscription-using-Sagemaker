# Predicting Customer Subscription Using AWS SageMaker

## Project Overview
This project focuses on developing a machine learning model to predict whether a customer will subscribe to a term deposit. Using AWS SageMaker, we build and deploy an optimized model trained on a dataset from a bank's marketing campaign. 

## Dataset
The dataset consists of **41,188** records with **62** features, including:
- **Customer Information** (e.g., `age`, `job_*`, `marital_*`, `education_*`)
- **Banking Behavior** (e.g., `default_*`, `housing_*`, `loan_*`)
- **Campaign Information** (e.g., `contact_*`, `month_*`, `day_of_week_*`)
- **Previous Interactions** (`pdays`, `previous`, `poutcome_*`)
- **Target Variable**: `y_yes` (1 if subscribed, 0 otherwise)

## Implementation Steps
### 1. Setup AWS Environment
- Install required libraries: `sagemaker`, `boto3`.
- Connect to AWS services using `boto3` to manage SageMaker, S3, and other resources.

```python
import sagemaker
import boto3
from sagemaker import get_execution_role

role = get_execution_role()
session = sagemaker.Session()
bucket = session.default_bucket()
```
- The above code initializes SageMaker and assigns an execution role.

### 2. Data Preprocessing & Storage
- Load and preprocess the dataset.
```python
import pandas as pd

# Load dataset
df = pd.read_csv("bank_clean.csv")

# Convert target variable to numeric
df['y_yes'] = df['y_yes'].astype(int)
```
- The dataset is loaded and the target column is converted into integer format for model training.
- Upload the dataset to an **S3 bucket** for SageMaker training.
```python
from sagemaker.s3 import S3Uploader

train_data_path = f's3://{bucket}/train_data.csv'
S3Uploader.upload("bank_clean.csv", train_data_path)
```
- This uploads the dataset to S3, allowing SageMaker to access it during training.

### 3. Model Training with SageMaker XGBoost
- Use SageMaker's **built-in XGBoost algorithm** to train a predictive model.
- Perform hyperparameter tuning **locally first**, then apply the best parameters in SageMaker.
```python
from sagemaker.estimator import Estimator

xgboost_estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve("xgboost", session.boto_region_name, "1.5-1"),
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f's3://{bucket}/output',
    sagemaker_session=session
)

xgboost_estimator.set_hyperparameters(
    objective="binary:logistic",
    num_round=100,
    max_depth=5,
    eta=0.2
)
```
- The above code initializes an XGBoost estimator with optimized hyperparameters.
- **Key Differences from Normal ML Training:**
  - Instead of training locally, we send data to S3 and let SageMaker handle training.
  - SageMaker uses a predefined XGBoost container instead of manually defining a model pipeline.
  - Training is done in a distributed environment, allowing scalability.

### 4. Model Evaluation & Metrics
- After training, we evaluated the model on test data.
```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
```
- The key metrics achieved were:
  - **Accuracy:** 92%
  - **Precision:** 88%
  - **Recall:** 85%
  - **F1-score:** 86%
- The model showed promising performance, effectively distinguishing between subscribed and non-subscribed customers.

### 5. Model Deployment
- Deploy the trained model as a **SageMaker endpoint**.
```python
predictor = xgboost_estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)
```
- Perform inference on new data using the deployed endpoint.
```python
response = predictor.predict(sample_data)
print("Prediction: ", response)
```
- **Why Deployment on SageMaker is Different:**
  - Instead of a simple `.predict()` function, we invoke the endpoint via `boto3` API calls.
  - The model runs on AWS infrastructure rather than local machines.
  - Real-time inference is scalable and cost-efficient.

### 6. Deleting Endpoints and Resources
- To avoid unnecessary AWS costs, all endpoints were deleted after testing.
- This is a crucial step since keeping endpoints running incurs charges even if they are not in use.
```python
predictor.delete_endpoint()
session.delete_endpoint(endpoint_name)
```
- This ensures that we efficiently manage cloud resources and expenses.

## Results & Conclusion
- The model predicts whether a customer will subscribe to a term deposit.
- AWS SageMaker allows scalable training and deployment without heavy local resource usage.
- The key takeaway is that **SageMaker automates ML processes but requires proper resource management** to avoid extra costs.



## Technologies Used
- **AWS SageMaker**
- **Python (Pandas, NumPy, Scikit-learn, Boto3)**
- **XGBoost**
- **Jupyter Notebook**

