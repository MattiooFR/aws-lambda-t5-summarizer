import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration



def encode(tokenizer, text):
    """encodes the question and context with a given tokenizer"""
    device = torch.device("cpu")
    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_Text = "summarize: " + preprocess_text
    return tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


def decode(tokenizer, summary_ids):
    """decodes the tokens to the answer with a given tokenizer"""
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def serverless_pipeline(model_path="./t5"):
    """Initializes the model and tokenzier and returns a predict function that ca be used as pipeline"""
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    def predict(text):
        """predicts the answer on an given question and context. Uses encode and decode method from above"""
        tokenized_text = encode(tokenizer, text)
        summary_ids = model.generate(
            tokenized_text,
            num_beams=4,
            min_length=30,
            max_length=200,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return output

    return predict


# initializes the pipeline
summarizer_pipeline = serverless_pipeline()


def handler(event, context):
    try:
        # loads the incoming event into a dictonary
        body = json.loads(event["body"])
        # uses the pipeline to predict the summary
        summary = summarizer_pipeline(text=body["text"])
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True,
            },
            "body": json.dumps({"summary": summary}),
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True,
            },
            "body": json.dumps({"error": repr(e)}),
        }
