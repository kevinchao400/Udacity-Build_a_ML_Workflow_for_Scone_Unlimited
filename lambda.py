# SeralizeImage Function

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the S3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, '/tmp/image.png')

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

# ImageClassification Function

import json
import sagemaker
import base64
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2024-09-25-06-16-53-094'

def lambda_handler(event, context):
    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])

    # Instantiate a Predictor
    predictor = Predictor(ENDPOINT)

    # Set the serializer to image/png
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction
    inferences = predictor.predict(image)

    # We return the data back to the Step Function    
    event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }

# FilterInference Function

import json

THRESHOLD = 0.80

def lambda_handler(event, context):

    # Grab the inferences from the event
    body = json.loads(event['body'])
    inferences = json.loads(body['inferences'])

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = max(inferences) > THRESHOLD

    # If our threshold is met, pass our data back out of the Step Function
    if meets_threshold:
        pass
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
