from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("vitus9988/klue-roberta-small-ner-identified")
model = AutoModelForTokenClassification.from_pretrained("vitus9988/klue-roberta-small-ner-identified")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
example = """
저는 서울특별시 강남대로 56길 100호에 삽니다. 전화번호는 010-1234-5678이고 주민등록번호는 123456-1234567입니다. 메일주소는 hugging@face.com입니다.
"""

ner_results = nlp(example)
for i in ner_results:
    print(i)