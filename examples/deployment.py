from omesa.containers import Pipeline

pl = Pipeline(name='20_news', store='db')
pl.load()

print(pl.classify(["some raw text"]))
