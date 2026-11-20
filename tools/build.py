import json
T = open('template.html').read()
G = open('guide.html').read()
M = open('methods.html').read()
S = open('syntax_ref.html').read()
XR = open('exp_ref.html').read()
XD = open('expdata.json').read().replace('</', '<\\/')
D = open('examples.json').read().replace('</', '<\\/')
SD = open('syntax.json').read().replace('</', '<\\/')
n_syn = len(json.loads(open('syntax.json').read())['ex'])
out = (T.replace('__GUIDE__', G).replace('__METHODS__', M).replace('__SYNREF__', S).replace('__EXPREF__', XR)
        .replace('__SYNCOUNT__', str(n_syn)).replace('__DATA__', D).replace('__SYNDATA__', SD).replace('__EXPDATA__', XD))
import os, re as _re
E3 = open('exam3/exam3.html', encoding='utf-8').read()
out = (out.replace('__EX3BOXES__', str(E3.count('type="checkbox"')))
          .replace('__EX3AUTO__', str(len(_re.findall(r'class="tq" id="[^"]+" data-kind="(?:num|mc)"', E3)))))
assert '__' not in out.split('<script type="application/json"')[0].replace('__proto__', ''), 'placeholder left'
open('r-trainer.html', 'w').write(out)
print(len(out) // 1024, 'KB')
