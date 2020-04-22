
#serverless/sls deploy


import json


def hello(event, context):
    #rank = event.queryStringParameters.rank
    
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event 
       
    }
    
    #print(event)
    response = {
        "statusCode": 200,
        "headers": { "Access-Control-Allow_Origin" : "*"} , 
        "body": json.dumps(body)
    }

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """
