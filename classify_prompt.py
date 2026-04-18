import boto3
import json

def call_bedrock_model(message: str, model_id: str = 'deepseek.v3.2'):
  session = boto3.Session(profile_name="GSB570-BedrockOnly-490332585640")
  client = session.client("bedrock-runtime", region_name="us-west-2")

  payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 500,
    "messages": [{"role": "user", "content": message}],
  }

  response = client.invoke_model(
    modelId=model_id,
    body=json.dumps(payload)
  )

  response_body = json.loads(response["body"].read())
  #print("Call Model RAW response",response_body)
  return response_body["choices"][0]["message"]["content"]

def classify_prompt(prompt: str, system_prompt_path: str = "classifier_instructions.txt") -> str:
  with open(system_prompt_path) as f:
    template = f.read()
    prompt = template.replace("{{USER_INPUT}}", prompt)
  classification_response = call_bedrock_model(prompt, model_id="deepseek.v3.2")
  return classification_response.strip().lower()