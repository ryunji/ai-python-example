# STEP 1. import modules
from transformers import pipeline
#from transformers import AutoTokenizer, AutoModelForSequenceClassification

# STEP 2. create inference instance
#tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
#model     = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")

# STEP 2. create inference instance
#classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

# STEP 3. prepare input data
text = "삼성전자 주가가 하락했다."

# STEP 4. inference
#inputs = tokenizer(text, return_tensors="pt")
#with torch.no_grad():
#    logits = model(**inputs).logits

# STEP 4. inference
result = classifier(text)

# 4-1. preprocessing(data -> tensor(blob)) : 사람이 읽을 수 있는 데이터가 tensor가 읽을 수 있는 데이터가 됨.
# 4-2. inference(tensor(blog) -> logit)
# 4-3. postprocessing(logit -> data)

# STEP 5. visualize
print(result)