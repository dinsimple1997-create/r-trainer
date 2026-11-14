import json
T = open('template.html', encoding='utf-8').read()
G = open('guide.html', encoding='utf-8').read()
D = open('examples.json', encoding='utf-8').read().replace('</', '<\\/')
html = T.replace('__GUIDE__', G).replace('__DATA__', D)
open('r-trainer.html', 'w', encoding='utf-8').write(html)
print(len(html.encode('utf-8')) / 1e6, 'MB')
