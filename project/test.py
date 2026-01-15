import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from frontend.modules.news import get_article
from frontend.modules.inference import predict

new_list = [
    'https://theonion.com/powerful-bidet-blasts-hole-clean-through-man/',
    'https://theonion.com/conservatives-say-renee-good-was-brainwashed-by-bible-into-loving-thy-neighbor/',
    'https://theonion.com/vivid-sex-dream-about-steely-mcbeam-again/',
    'https://theonion.com/melania-trump-casts-longtime-aide-into-well-of-gloom/',
    'https://theonion.com/rfk-jr-coughs-up-pair-of-jeans/'

]

new_list2 = [
    'https://www.bbc.com/news/articles/cd0ydjvxpejo',
    'https://www.theguardian.com/us-news/2026/jan/13/upenn-trump-jews-list',
    'https://edition.cnn.com/2026/01/15/politics/trump-health-care-plan',
    'https://www.nbcnews.com/business/consumer/saks-bankruptcy-luxury-retail-rcna254193',

    

]

for new in new_list2:
    n = get_article(new)

    result = predict(n['title'], n['text'])
    with open('output.txt', 'a+') as f:
        print(result['final_prediction'].iloc[0], file=f)
        print(result['confidence'].iloc[0], file=f)


