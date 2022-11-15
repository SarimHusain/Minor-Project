from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers import pipeline

from config import defaults


def test():
    model = AutoModelForQuestionAnswering.from_pretrained(defaults["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(defaults["model_path"])

    model.to(defaults["device"])

    qa_model = pipeline("question-answering", 
        model=model, 
        tokenizer=tokenizer, 
        device=model.device)

    context = input('Context: ')

    while True:
        print()
        question = input('Question: ')
        answer = qa_model(question = question, context = context)["answer"]
        print(f"Answer: {answer}")
        # for k, v in qa_model(question = question, context = context).items():
        #     print(f"{str(k).upper()}: {v}")

