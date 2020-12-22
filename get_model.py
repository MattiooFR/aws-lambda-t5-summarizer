from transformers import T5Tokenizer, T5ForConditionalGeneration


def get_model(model):
    """Loads model from Hugginface model hub"""
    try:
        model = T5ForConditionalGeneration.from_pretrained(model)
        model.save_pretrained("./t5")
    except Exception as e:
        raise (e)


def get_tokenizer(tokenizer):
    """Loads tokenizer from Hugginface model hub"""
    try:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer)
        tokenizer.save_pretrained("./t5")
    except Exception as e:
        raise (e)


get_model("t5-small")
get_tokenizer("t5-small")
