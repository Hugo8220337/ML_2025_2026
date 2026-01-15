from newspaper import Article


def get_article(url):
    article = Article(url)
    article.download()
    article.parse()
    
    return {
        'title': article.title,
        'text': article.text,
    }

