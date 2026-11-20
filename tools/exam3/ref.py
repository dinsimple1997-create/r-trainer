from lib import *


def atlas():
    return f'''
<section id="atlas">
  <span class="daytag">Справочник</span>
  <h2>Атлас выводов R: что печатает каждая команда и куда смотреть</h2>
  {table(['Команда', 'Что печатает', 'Куда смотреть первым делом', 'Типичный вопрос'], [
      ['<code>summary(lm(y ~ x))</code>', 'Call, Residuals, Coefficients, s, R², F', 'Call (есть ли log), строка X: Estimate, SE, t, p', 'интерпретировать наклон, H₀, ДИ, s, R²'],
      ['<code>confint(fit)</code>', '2,5 % и 97,5 % для каждого коэффициента', 'строка X', '«interpret the 95% CI»; у glm — профильный ДИ, экспоненцировать'],
      ['<code>anova(fit)</code>', 'Df, Sum Sq, Mean Sq, F по строкам', '<b>Type I</b>: каждая строка после предыдущих', 'SSR, SSE, MSE; F = t² в SLR'],
      ['<code>anova(fit1, fit2)</code>', 'Res.Df, RSS двух моделей, Df, Sum of Sq, F, p', 'Sum of Sq = RSS₁ − RSS₂; F = (это / Df) / (RSS₂ / Res.Df₂)', 'частичный F; lack of fit; взаимодействие'],
      ['<code>car::Anova(fit, type = 3)</code>', 'Sum Sq, Df, F по строкам', 'каждая строка после всех остальных', '«adds anything after the others?»'],
      ['<code>predict(fit, new, interval = "confidence")</code>', 'fit, lwr, upr', 'для <b>среднего</b>', 'mean response'],
      ['<code>predict(fit, new, interval = "prediction")</code>', 'fit, lwr, upr', 'для <b>одного</b> объекта, всегда шире', 'individual'],
      ['<code>summary(glm(…, binomial))</code>', 'Coefficients с z, Null/Residual deviance, AIC', 'Estimate X → e<sup>b</sup> = OR', 'OR, вероятность, Вальд'],
      ['<code>anova(fit0, fit, test = "Chisq")</code>', 'Resid. Df, Resid. Dev, Deviance, Pr(&gt;Chi)', 'Deviance = падение девианса', 'LRT, drop-in-deviance'],
      ['<code>predict(glm, type = "response")</code>', 'вероятность (binomial) или среднее (poisson)', '—', 'прогноз на исходной шкале'],
      ['<code>predict(glm, se.fit = TRUE)</code>', 'логит (или log μ) и его SE', 'ДИ на логите, потом перевести концы', 'CI for probability'],
      ['<code>summary(glm(…, poisson))</code>', 'как binomial', 'e<sup>b</sup> = отношение средних; Residual deviance / df', 'передисперсия'],
      ['<code>glm(…, quasipoisson)</code>', 't вместо z, Dispersion parameter', 'SE × √φ, оценки те же', 'что изменилось'],
      ['<code>rstandard / hatvalues / cooks.distance</code>', 'числа на каждое наблюдение', '|r*| &gt; 3; hat большой; D &gt; 1', 'выбросы, влияние'],
      ['<code>car::vif(fit)</code>', 'по числу на X', '&gt; 10', 'мультиколлинеарность'],
      ['<code>MuMIn::dredge(fit)</code>', 'таблица моделей: коэффициенты, df, logLik, AICc, delta, weight', 'строка 1, delta ≤ 2', 'лучшая модель, альтернативы']])}
  <h3>Как читать маленькие p</h3>
  <p><code>1.06e-07</code> = 1,06 × 10⁻⁷ = 0,000000106. <code>&lt;2e-16</code> — меньше, чем R умеет показать. В ответе пиши «p &lt; 0.001». Звёздочки: *** &lt; 0,001; ** &lt; 0,01; * &lt; 0,05; . &lt; 0,1.</p>
</section>
'''


def formulas():
    return f'''
<section id="cheat">
  <span class="daytag">Справочник</span>
  <h2>Все формулы словами</h2>
  <h3>Простая регрессия</h3>
  {table(['Что', 'Формула', 'Словами'], [
      ['Прямая', 'Ŷ = b₀ + b₁X', 'свободный член + наклон × X'],
      ['Остаток', 'e = Y − Ŷ', 'было минус прогноз; + → модель недооценила'],
      ['s', '√(SSE / (n − 2))', 'типичный промах прямой, в единицах Y'],
      ['SE(b₁)', 's / √[(n − 1)s<sub>X</sub>²]', 'меньше при большом n и широком X'],
      ['t', 'b / SE', 'на сколько SE оценка от нуля'],
      ['ДИ', 'b ± t*·SE', 't* для df остатков'],
      ['R²', 'SSR / SST = 1 − SSE/SST', 'доля объяснённого'],
      ['F (SLR)', 'MSR / MSE = t²', 'тот же тест, что t'],
      ['SE среднего при X₀', 's·√[1/n + (X₀ − X̄)²/((n − 1)s<sub>X</sub>²)]', 'минимум при X̄'],
      ['SE одного при X₀', '√(SE среднего² + s²)', 'всегда больше'],
      ['Калибровка', 'X̂ = (Y₀ − b₀) / b₁', 'обратный ход по прямой'],
      ['r', '±√R²; b₁ = r·s<sub>Y</sub>/s<sub>X</sub>', 'знак как у наклона']])}
  <h3>Логарифмы</h3>
  {table(['Модель', 'Наклон →', 'Про что'], [
      ['log Y ~ X', 'e<sup>b</sup> на +1 X; (e<sup>b</sup> − 1)·100 %', 'медиана Y'],
      ['Y ~ log X', 'b·ln 2 при удвоении X', 'среднее Y'],
      ['log Y ~ log X', '2<sup>b</sup> при удвоении X; +1 % X → ≈ +b % Y', 'медиана Y'],
      ['ДИ', 'преобразовать концы (e или 2<sup>…</sup>)', '—']])}
  <h3>Логистическая и пуассоновская</h3>
  {table(['Что', 'Формула'], [
      ['Шансы', 'p / (1 − p)'],
      ['Логит', 'ln[p / (1 − p)] = b₀ + b₁X'],
      ['Вероятность из логита', '1 / (1 + e<sup>−l</sup>)'],
      ['Отношение шансов', 'e<sup>b₁</sup> на +1 X; e<sup>c·b₁</sup> на +c'],
      ['Вальд', 'z = b / SE; ДИ b ± 1,96·SE'],
      ['LRT', 'Deviance<sub>null</sub> − Deviance<sub>full</sub> ~ χ² с df = разница параметров'],
      ['Пуассон', 'log μ = b₀ + b₁X; μ = e<sup>l</sup>; e<sup>b₁</sup> = отношение средних'],
      ['Передисперсия', 'Residual deviance / df; &gt; 2 — проблема; quasi: SE × √φ']])}
  <h3>Множественная регрессия</h3>
  {table(['Что', 'Формула'], [
      ['df остатков', 'n − p (p — все коэффициенты с b₀)'],
      ['Частичный F', '[(SSE<sub>red</sub> − SSE<sub>full</sub>) / (df<sub>red</sub> − df<sub>full</sub>)] / (SSE<sub>full</sub> / df<sub>full</sub>)'],
      ['Общий F', 'H₀: все наклоны = 0; df = (p − 1, n − p)'],
      ['Скорректированное среднее (ANCOVA)', 'Ȳ<sub>i</sub> − b(X̄<sub>i</sub> − X̄)'],
      ['Наклон группы при взаимодействии', 'b<sub>X</sub> + b<sub>X:группа</sub>'],
      ['Разница групп при X₀', 'b<sub>группа</sub> + b<sub>X:группа</sub>·X₀'],
      ['Оптимум параболы', 'X = −b₁ / (2b₂); b₂ &lt; 0 — максимум'],
      ['VIF', '1 / (1 − R<sub>j</sub>²); &gt; 10 — проблема'],
      ['Стандартизованный остаток', 'e / [s√(1 − h)]; ±2 у 95 %'],
      ['Cook’s D', 'большой при большом h и большом остатке; &gt; 1 — проверить'],
      ['AIC', '−2 lnL + 2k (k включает σ)'],
      ['AICc', 'AIC + 2k(k + 1)/(n − k − 1)'],
      ['BIC', '−2 lnL + k·ln n'],
      ['PRESS', 'Σ(Y<sub>i</sub> − Ŷ<sub>(−i)</sub>)²']])}
</section>
'''


def traps():
    items = [
        'Причинность в наблюдении. Только «is associated with». «Causes» — когда X назначали случайно.',
        'Нет единиц и контекста. «X increases Y by b» без кг, г/сут и без поросят — минус балл.',
        'Забыли «on average» в интерпретации наклона.',
        'Экспоненцировали при log только у X. При Y ~ log X ответ — b × 0,693 в единицах Y.',
        'Сказали «среднее» вместо «медиана» при log(Y) — или «медиана» вместо «среднее» в Пуассоне.',
        'Сложили проценты: 10 недель по −14 % — это не −140 %, а e<sup>10b</sup>.',
        'Минус в экспоненте: e<sup>−0,0184</sup> = 0,982, не 1,018. Клавиша (−).',
        'Перепутали CI и PI. Один объект — prediction.',
        'Самая узкая полоса — при среднем log X, если X логарифмирован.',
        'Толковали свободный член, когда X = 0 вне данных.',
        '«The line fits» вместо «no evidence of lack of fit».',
        'anova(fit) — это Type I. Не делать по нему вывод «X не значим при остальных».',
        'OR — про шансы, не про вероятность.',
        'Ждали, что LRT = z². В glm они разные.',
        'Не проверили передисперсию у счётов (Residual deviance / df).',
        'В модели с взаимодействием коэффициент группы — разница при X = 0.',
        'В квадратичной модели толковали b₁ отдельно.',
        'Сравнивали AIC моделей с разным Y (log Y и Y) или разным числом наблюдений.',
        'Из отбора модели вывели «не вошедший X не влияет».',
        'Независимость проверяли по графику, а не по дизайну (кто единица эксперимента).']
    return f'''
<section id="traps">
  <span class="daytag">Справочник</span>
  <h2>Где теряют баллы: 20 ловушек</h2>
  <ol>{''.join(f'<li>{x}</li>' for x in items)}</ol>
</section>
'''


def gloss():
    pairs = [('наклон', 'slope'), ('свободный член', 'intercept'), ('остаток', 'residual'), ('прогноз', 'fitted / predicted value'), ('стандартная ошибка', 'standard error (SE)'),
             ('остаточное стандартное отклонение', 'residual standard error (s, RMSE)'), ('доверительный интервал', 'confidence interval'), ('прогнозный интервал', 'prediction interval'),
             ('степени свободы', 'degrees of freedom'), ('сумма квадратов', 'sum of squares (SSE, SSR, SST)'), ('отсутствие соответствия', 'lack of fit'), ('чистая ошибка', 'pure error'),
             ('калибровка', 'calibration (inverse prediction)'), ('экстраполяция', 'extrapolation'), ('медиана', 'median'), ('удвоение', 'doubling'), ('множитель', 'multiplicative change, fold change'),
             ('шансы', 'odds'), ('логит', 'log odds, logit'), ('отношение шансов', 'odds ratio'), ('отношение средних (Пуассон)', 'rate ratio, multiplicative change in the mean'),
             ('девианс', 'deviance'), ('тест отношения правдоподобия', 'likelihood ratio test, drop-in-deviance test'), ('профильный ДИ', 'profile likelihood interval'),
             ('передисперсия', 'overdispersion'), ('при прочих равных', 'holding the other variables constant / comparing … with the same …'), ('конфаундер', 'confounding variable'),
             ('частичный F-тест', 'extra-sum-of-squares F test, partial F test'), ('последовательные / частичные суммы', 'sequential (Type I) / partial (Type III) sums of squares'),
             ('индикатор', 'indicator (dummy) variable'), ('опорный уровень', 'reference level'), ('скорректированное среднее', 'adjusted mean'), ('параллельные прямые', 'parallel lines (ANCOVA)'),
             ('взаимодействие', 'interaction'), ('квадратичный член', 'quadratic term'), ('центрирование', 'centering'), ('мультиколлинеарность', 'multicollinearity'),
             ('рычаг', 'leverage (hat value)'), ('влиятельное наблюдение', 'influential observation'), ('выбор модели', 'model selection'), ('переобучение', 'overfitting'),
             ('кросс-валидация', 'cross-validation'), ('единица эксперимента', 'experimental unit')]
    return f'''
<section id="gloss">
  <span class="daytag">Справочник</span>
  <h2>Русский ↔ English</h2>
  {table(['По-русски', 'In English'], pairs)}
</section>
'''


def sources():
    return '''
<section id="src">
  <span class="daytag">Справочник</span>
  <h2>На чём основано</h2>
  <ul class="src">
    <li>Canvas, STAT 5101 (Fall 2026): лекции m5-s1 … m5-s23 и m6-s1 … m6-s28, лабораторные Lab08–Lab12. Все файлы лежат в <code>Загрузки\\STAT5101_Exam3\\Module5</code> и <code>\\Module6</code>.</li>
    <li>Practice M5 Part 1 и Part 2, HW05 Part I (Practice Module 6) с решениями преподавателя — <code>\\Practice</code>.</li>
    <li>Формат экзамена: страницы Exam 1 Overview и Exam 2 Overview, Course Schedule, STAT 5101 Exam #2 Practice.</li>
    <li>Учебные выводы R со свинофермы построены на данных, которые я сгенерировал, и проверены пересчётом: t = Estimate / SE, F = t², ДИ, прогнозы, девиансы, AICc сходятся.</li>
    <li>Стиль и подход — твой план подготовки к Exam 2 (проект Exam 2 STAT5101).</li>
  </ul>
</section>
'''
