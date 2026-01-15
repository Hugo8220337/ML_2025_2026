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

for n in new_list:
    n = get_article(n)

    result = predict(n['title'], n['text'])
    with open('output.txt', 'w') as f:
        print(result['final_prediction'].iloc[0], file=f)
        print(result['confidence'].iloc[0], file=f)


