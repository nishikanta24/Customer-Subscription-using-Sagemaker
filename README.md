# Predicting Customer Subscription Using AWS SageMaker

## Project Overview
This project focuses on developing a machine learning model to predict whether a customer will subscribe to a term deposit. Using AWS SageMaker, we build and deploy an optimized XGBoost model trained on a dataset from a bank's marketing campaign. The project demonstrates how to preprocess data, train a model, evaluate its performance, and deploy it as a scalable endpoint on AWS SageMaker.

## Key Features of AWS SageMaker
- **Scalable ML Training**: Train models on cloud instances without worrying about local hardware limitations.
- **Managed Model Deployment**: Deploy models as API endpoints for real-time predictions.
- **Hyperparameter Optimization**: Automatically tune models for optimal performance.
- **Seamless S3 Integration**: Store and retrieve datasets directly from AWS S3.
- **Built-in Algorithms**: Utilize pre-configured machine learning models like XGBoost, TensorFlow, and more.

## Dataset
The dataset consists of 41,188 records with 62 features, including:
- **Customer Information** (e.g., `age`, `job_*`, `marital_*`, `education_*`)
- **Banking Behavior** (e.g., `default_*`, `housing_*`, `loan_*`)
- **Campaign Information** (e.g., `contact_*`, `month_*`, `day_of_week_*`)
- **Previous Interactions** (`pdays`, `previous`, `poutcome_*`)
- **Target Variable**: `y_yes` (1 if subscribed, 0 otherwise)

## Implementation Steps

### 1. Setup AWS Environment
We begin by setting up the AWS environment, including initializing SageMaker and creating an S3 bucket for data storage.

```python
import sagemaker
import boto3
from sagemaker import image_uris
from sagemaker.session import s3_input, Session

# Initialize SageMaker and S3
bucket_name = 'bank-application-nishi-24'
my_region = boto3.session.Session().region_name  # Set the region
print(my_region)

# Create an S3 bucket
s3 = boto3.client('s3')
try:
    if my_region == 'eu-north-1':
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': my_region}
        )
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error:', e)
```

### 2. Data Preprocessing & Storage
The dataset is loaded, preprocessed, and uploaded to the S3 bucket for SageMaker training.

```python
import pandas as pd
import urllib

# Download the dataset
try:
    urllib.request.urlretrieve("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
    print('Success: Downloaded bank_clean.csv')
except Exception as e:
    print('Data load error: ', e)

# Load the dataset into a DataFrame
try:
    model_data = pd.read_csv('./bank_clean.csv', index_col=0)
    print('Success: Data loaded into DataFrame')
except Exception as e:
    print('Data load error: ', e)

# Train-test split
import numpy as np
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)
```

### 3. Model Training with SageMaker XGBoost
We use SageMaker's built-in XGBoost algorithm to train the model.

```python
# Retrieve the XGBoost container
container = image_uris.retrieve(
    framework='xgboost',
    region=boto3.session.Session().region_name,
    instance_type='ml.t3.medium',
    version='1.5-1'
)

# Initialize hyperparameters
hyperparameters = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.7",
    "objective": "binary:logistic",
    "num_round": "50"
}
```

### 4. Model Evaluation & Metrics
After training, the model is evaluated on the test dataset.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Make predictions
predictions = xgb_predictor.predict(test_data_array).decode('utf-8')
predictions_array = np.loadtxt(predictions.splitlines(), delimiter=',')

# Calculate metrics
accuracy = accuracy_score(test_data['y_yes'], np.round(predictions_array))
precision = precision_score(test_data['y_yes'], np.round(predictions_array))
recall = recall_score(test_data['y_yes'], np.round(predictions_array))
f1 = f1_score(test_data['y_yes'], np.round(predictions_array))

print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-score: {f1}")
```

### 5. Model Deployment
The trained model is deployed as a SageMaker endpoint for real-time predictions.

```python
# Deploy the model
xgb_predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Perform inference
response = xgb_predictor.predict(sample_data)
print("Prediction: ", response)
```

### 6. Cleanup Resources
To avoid unnecessary AWS costs, all endpoints and S3 objects are deleted after testing.

```python
# Delete SageMaker endpoints and S3 objects
def cleanup_resources(bucket_name):
    # Delete all SageMaker endpoints
    sagemaker_client = boto3.client('sagemaker')
    endpoints = sagemaker_client.list_endpoints()['Endpoints']
    for endpoint in endpoints:
        endpoint_name = endpoint['EndpointName']
        print(f"Deleting endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
    print("All endpoints deleted.")

    # Delete all objects in the S3 bucket
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    print(f"Deleting all objects in bucket: {bucket_name}")
    bucket.objects.all().delete()
    print(f"All objects in bucket '{bucket_name}' deleted.")

# Run cleanup
cleanup_resources(bucket_name)
```

## Results & Conclusion
- The model achieves an **89.7% accuracy** in predicting customer subscriptions.
- AWS SageMaker simplifies the process of training, deploying, and managing machine learning models at scale.
- Proper resource management is crucial to avoid unnecessary costs.


## Technologies Used
- **AWS SageMaker**
- **Python** (Pandas, NumPy, Scikit-learn, Boto3)
- **XGBoost**
- **Jupyter Notebook**



