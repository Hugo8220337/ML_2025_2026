import contextlib
import os
import sys
from frontend.modules.news import get_article
from frontend.modules.inference import predict



@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = fnull
            sys.stderr = fnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

real = [
    'https://www.bbc.com/news/articles/cd0ydjvxpejo',
    'https://www.theguardian.com/us-news/2026/jan/13/upenn-trump-jews-list',
    'https://edition.cnn.com/2026/01/15/politics/trump-health-care-plan',
    'https://www.nbcnews.com/business/consumer/saks-bankruptcy-luxury-retail-rcna254193',
    'https://edition.cnn.com/politics/live-news/trump-venezuela-machado-01-15-26'
]

fake = [
    'https://beforeitsnews.com/opinion-conservative/2026/01/iran-enters-its-final-hour-despite-reports-of-10000-dead-iran-still-rises-streets-erupt-as-u-s-and-israel-prepare-to-act-3734111.html',
    'https://www.thegatewaypundit.com/2026/01/watch-youre-left-wing-hack-karoline-leavitt-unloads/',
    'https://www.naturalnews.com/2026-01-15-russia-prepares-nuclear-response-america-faces-defeat.html',
    'https://www.zerohedge.com/geopolitical/ron-paul-making-imperialism-great-again',
    'https://theonion.com/elon-musk-files-for-full-custody-of-all-u-s-children/'
]

for new in fake:
    n = get_article(new)

    with suppress_output():
        result = predict(n['title'], n['text'])

            
    print("title: "+result['title'].iloc[0])
    print("text: "+result['text'].iloc[0][:50])
    print("topic: "+ str(result['topic'].iloc[0]))
    print("anomaly: " + str(result['anomaly'].iloc[0]))
    print("stance: " + result['stance'].iloc[0])
    print("clickbait: " + str(result['clickbait'].iloc[0]))
    print("prediction: "+ result['final_prediction'].iloc[0])
    print("confidence: "+ str(result['confidence'].iloc[0]))



