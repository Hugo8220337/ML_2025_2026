from newspaper import Article

url = 'https://www.bbc.com/news/articles/cjw1nxe5pvlo'
article = Article(url)

article.download()
article.parse()

print(article.title)
print(article.text)