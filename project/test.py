import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from frontend.modules.news import get_article
from inference import predict

n = get_article('https://www.aljazeera.com/news/2026/1/15/ice-officer-shoots-venezuelan-immigrant-in-minneapolis-what-we-know')

result = predict(n['title'], n['text'])
print


