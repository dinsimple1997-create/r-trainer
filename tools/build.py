import json
T = open('template.html').read()
G = open('guide.html').read()
M = open('methods.html').read()
S = open('syntax_ref.html').read()
D = open('examples.json').read().replace('</', '<\\/')
SD = open('syntax.json').read().replace('</', '<\\/')
n_syn = len(json.loads(open('syntax.json').read())['ex'])
out = (T.replace('__GUIDE__', G).replace('__METHODS__', M).replace('__SYNREF__', S)
        .replace('__SYNCOUNT__', str(n_syn)).replace('__DATA__', D).replace('__SYNDATA__', SD))
assert '__' not in out.split('<script type="application/json"')[0].replace('__proto__', ''), 'placeholder left'
open('r-trainer.html', 'w').write(out)
print(len(out) // 1024, 'KB')
