from newspaper import Article, Config

def get_article(url):
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    config.request_timeout = 10

    try:
        article = Article(url, config=config)
        article.download()
        article.parse()
        
        if len(article.text) < 50:
            print(f"Warning: {url} returned empty text. Likely blocked.")
            return None

        return {
            'title': article.title,
            'text': article.text,
            'url': url
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None