import math
from lib import *

A1, A2, A3 = EV['A1'], EV['A2'], EV['A3']
B1, B2, B3, B4 = EV['B1'], EV['B2'], EV['B3'], EV['B4']
e = math.exp


def eq(n, q, ans, note=''):
    nb = f'<p class="small">{note}</p>' if note else ''
    return f'<details class="q eq"><summary><span class="qnum">Q{n}</span>{q}</summary><div class="body"><div class="en">{ans}</div>{nb}</div></details>'


def timer(tid, minutes=120):
    return f'<div class="timer" data-min="{minutes}" id="{tid}"><span class="clock">{minutes:02d}:00</span><button class="btn tstart" type="button">Старт</button><button class="btn tpause" type="button">Пауза</button><button class="btn treset" type="button">Сброс</button><span class="small">минут: </span><input class="tmin" type="number" min="1" value="{minutes}"></div>'


def stim(title, text):
    return f'<div class="exam"><span class="lbl">{title}</span><div class="en">{text}</div></div>'


def mockA():
    q = []
    q.append(eq(1, 'Is this a randomized experiment or an observational study? What does that allow you to conclude?',
                '<p>A randomized experiment: each piglet was randomly assigned an iron dose. Differences in haemoglobin can therefore be attributed to the dose (causal conclusions), for piglets like those in the study.</p>'))
    q.append(eq(2, 'Explain why the investigators used log(dose) rather than dose as the explanatory variable.',
                '<p>The doses double (50, 100, 200, 400 mg), so they are equally spaced on the log scale, and the response is expected to increase by a constant amount for each doubling, with a diminishing gain per mg at higher doses. With log(dose) the relationship is linear.</p>'))
    q.append(eq(3, 'Interpret the estimated slope in the context of the study, on the dose scale.',
                f'<p>Doubling the iron dose increases mean haemoglobin by {en(A1["b1"],3)} × ln 2 = {en(A1["b1"]*math.log(2),2)} g/dL.</p>', 'Логарифм только у X → не экспоненцируем, умножаем на 0,693.'))
    lo, hi = A1['b1'] - A1['tcrit'] * A1['se1'], A1['b1'] + A1['tcrit'] * A1['se1']
    q.append(eq(4, f'Compute a 95% confidence interval for the slope (t* = {en(A1["tcrit"],3)} on 30 df) and express it per doubling of dose.',
                f'<p>{en(A1["b1"],4)} ± {en(A1["tcrit"],3)} × {en(A1["se1"],4)} = ({en(lo,3)}, {en(hi,3)}). Per doubling: × 0.693 → ({en(lo*math.log(2),2)}, {en(hi*math.log(2),2)}) g/dL.</p>'))
    q.append(eq(5, 'State H₀ and Hₐ for the slope, give the test statistic and conclusion.',
                f'<p>H₀: β₁ = 0; Hₐ: β₁ ≠ 0. t = {en(A1["b1"],4)} / {en(A1["se1"],4)} = {en(A1["b1"]/A1["se1"],2)} on 30 df, p = 5.2 × 10⁻⁷. Strong evidence that mean haemoglobin increases with log(dose).</p>'))
    q.append(eq(6, 'Using the coefficients, compute the predicted mean haemoglobin for a dose of 200 mg.',
                f'<p>{en(A1["b0"],4)} + {en(A1["b1"],4)} × ln 200 = {en(A1["b0"],4)} + {en(A1["b1"],4)} × 5.298 = {en(A1["fit"],2)} g/dL (matches predict()).</p>'))
    q.append(eq(7, 'A farmer asks what haemoglobin to expect in one particular piglet given 200 mg. Which interval from the output do you report, and what is it?',
                f'<p>The prediction interval, because it is for one individual: ({en(A1["plo"],2)}, {en(A1["phi"],2)}) g/dL. The confidence interval ({en(A1["clo"],2)}, {en(A1["chi"],2)}) is for the mean of all piglets given 200 mg.</p>'))
    q.append(eq(8, 'Show how the width of the prediction interval follows from the standard error of the mean prediction (0.1123) and the residual standard error.',
                f'<p>SE(pred) = √(0.1123² + {en(A1["s"],4)}²) = {en(math.sqrt(A1["se_fit"]**2+A1["s"]**2),4)}. Half-width = {en(A1["tcrit"],3)} × {en(math.sqrt(A1["se_fit"]**2+A1["s"]**2),4)} = {en(A1["tcrit"]*math.sqrt(A1["se_fit"]**2+A1["s"]**2),3)}, so {en(A1["fit"],3)} ± {en(A1["tcrit"]*math.sqrt(A1["se_fit"]**2+A1["s"]**2),2)}.</p>'))
    q.append(eq(9, 'At what dose are the confidence bands narrowest? Explain.',
                f'<p>At the mean of log(dose), {en(A1["meanlog"],3)}, i.e. dose = e^{en(A1["meanlog"],3)} = {en(e(A1["meanlog"]),0)} mg. The SE of the predicted mean contains (X₀ − X̄)², which is zero when X₀ equals the mean of the explanatory variable, here log(dose).</p>', 'Не 187,5 мг (среднее доз), а геометрическое среднее 141 мг.'))
    cl = (11 - A1['b0']) / A1['b1']
    q.append(eq(10, 'Anaemia is defined as haemoglobin below 11 g/dL. Estimate the dose at which predicted mean haemoglobin equals 11 g/dL.',
                f'<p>log(dose) = (11 − {en(A1["b0"],4)}) / {en(A1["b1"],4)} = {en(cl,3)}; dose = e^{en(cl,3)} = {en(e(cl),0)} mg.</p>', 'Калибровка и потом e, потому что X = log(dose).'))
    q.append(eq(11, 'Use the lack-of-fit output to decide whether the straight line in log(dose) is adequate.',
                f'<p>F = {en(A1["lof"]["F"],3)} on 2 and 28 df, p = {en(A1["lof"]["p"],2)}. No evidence of lack of fit; the four dose means are consistent with a straight line in log(dose).</p>'))
    q.append(eq(12, 'Interpret the residual standard error and R².',
                f'<p>Haemoglobin of individual piglets typically differs from the fitted line by about {en(A1["s"],2)} g/dL. About {en(100*A1["r2"],0)}% of the variation in haemoglobin is explained by its linear relationship with log(dose).</p>'))
    # A2
    q.append(eq(13, 'Why is logistic regression appropriate here instead of simple linear regression?',
                '<p>The response is binary (seropositive or not). A straight line for the probability could give values outside 0–1 and the variance of a 0/1 response depends on p. Logistic regression models the log odds as a linear function of age, so fitted probabilities stay between 0 and 1.</p>'))
    q.append(eq(14, 'Interpret the slope for age as an odds ratio for 1 week and for 4 weeks.',
                f'<p>e^{en(A2["b1"],4)} = {en(e(A2["b1"]),3)}: each additional week of age multiplies the odds of seropositivity by 1.32 (32% higher odds). Over 4 weeks: e^(4 × {en(A2["b1"],4)}) = {en(e(4*A2["b1"]),2)}, about three times the odds.</p>'))
    q.append(eq(15, 'Give a 95% confidence interval for the weekly odds ratio. Which interval in the output did you use and why?',
                f'<p>Profile likelihood CI from confint(): e^{en(A2["ci"][1],4)} = {en(e(A2["ci"][1]),3)} to e^{en(A2["ci"][3],4)} = {en(e(A2["ci"][3]),3)}. The profile interval is preferred over the Wald interval because it relies less on the normal approximation.</p>'))
    q.append(eq(16, 'Compute the Wald 95% CI for the slope and compare it with the profile interval.',
                f'<p>{en(A2["b1"],4)} ± 1.96 × {en(A2["se1"],4)} = ({en(A2["wald"][0],4)}, {en(A2["wald"][1],4)}); profile ({en(A2["ci"][1],4)}, {en(A2["ci"][3],4)}). They are very similar here; the profile interval is not symmetric around the estimate.</p>'))
    q.append(eq(17, 'Estimate the probability that a 16-week-old pig is seropositive. Show the steps.',
                f'<p>Logit = {en(A2["b0"],4)} + {en(A2["b1"],4)} × 16 = {en(A2["b0"]+16*A2["b1"],3)}; odds = e^{en(A2["b0"]+16*A2["b1"],3)} = {en(e(A2["b0"]+16*A2["b1"]),3)}; p = odds / (1 + odds) = {en(A2["p16"],3)}.</p>'))
    q.append(eq(18, 'At what age is the estimated probability of seropositivity 50%?',
                f'<p>p = 0.5 when the logit is 0: age = {en(-A2["b0"],4)} / {en(A2["b1"],4)} = {en(-A2["b0"]/A2["b1"],1)} weeks.</p>'))
    q.append(eq(19, 'Report the likelihood ratio (drop-in-deviance) test for age. Why is it not equal to z²?',
                f'<p>323.05 − 226.86 = 96.18 on 1 df, p &lt; 2 × 10⁻¹⁶. z² = 7.86² = {en(A2["z"]**2,1)}. In generalized linear models the Wald test and the likelihood ratio test rely on different approximations, so they need not agree; the LRT is preferred.</p>'))
    q.append(eq(20, 'Can you conclude that ageing causes seroconversion? Comment on the design.',
                '<p>No. This is an observational, cross-sectional study: pigs of different ages were sampled at one time and differ in more than age (batch, exposure history, vaccination timing). The model describes an association. Sampling one pig per pen supports independence.</p>'))
    # A3
    q.append(eq(21, 'Interpret the slope for ammonia as a rate ratio per 1 ppm and per 10 ppm.',
                f'<p>e^{en(A3["b1"],4)} = {en(e(A3["b1"]),3)}: each additional ppm of ammonia is associated with a 6.9% higher mean number of coughs. Per 10 ppm: e^{en(10*A3["b1"],3)} = {en(e(10*A3["b1"]),2)}, nearly double.</p>'))
    q.append(eq(22, 'Estimate the mean number of coughs per 15 minutes at 10 ppm and at 25 ppm.',
                f'<p>e^({en(A3["b0"],4)} + 10 × {en(A3["b1"],4)}) = {en(A3["mu10"],2)}; e^({en(A3["b0"],4)} + 25 × {en(A3["b1"],4)}) = {en(A3["mu25"],2)} coughs.</p>', 'Пуассон: e от прогноза — среднее, не медиана.'))
    q.append(eq(23, 'Is there evidence of overdispersion? What are the consequences if it is ignored?',
                f'<p>Residual deviance / df = 111.04 / 46 = {en(A3["rd"]/float(A3["rdf"]),2)} &gt; 2, so there is overdispersion. If ignored, the standard errors are too small, confidence intervals too narrow and p-values too small.</p>'))
    q.append(eq(24, 'Compare the Poisson and quasi-Poisson fits. Show how the quasi-Poisson SE is obtained.',
                f'<p>The estimates are identical. The quasi-Poisson SE = {en(A3["se1"],5)} × √{en(A3["phi"],3)} = {en(A3["se1"]*math.sqrt(A3["phi"]),4)}. The statistic becomes t = {en(A3["b1"]/A3["se1q"],2)} (instead of z = 7.27), p = 3.2 × 10⁻⁵. Ammonia is still clearly associated with coughing, but with less certainty.</p>'))
    q.append(eq(25, 'In this model, does e^(b₀ + b₁X) estimate the mean or the median number of coughs? Explain.',
                '<p>The mean. With a log link the logarithm is applied to the mean μ, so exponentiating the linear predictor gives the mean. (With log(Y) in an ordinary regression, exponentiating gives the median.)</p>'))
    q.append(eq(26, 'Give one reason why the association between ammonia and coughing may not be causal.',
                '<p>The barns were observed, not assigned ammonia levels. Barns with high ammonia may also have poorer ventilation, higher dust, more respiratory pathogens or higher stocking density, any of which could cause coughing.</p>'))
    return f'''
<section id="d19">
  <span class="daytag">День 19 · 2,5 ч · Пробный экзамен A</span>
  <h2>Пробный экзамен A (модуль 5): 26 вопросов, 2 часа</h2>
  <p>Как на настоящем: только калькулятор и твои записи, без R и без этого файла. Отвечай письменно по-английски. Потом сверяй — ответы под стрелками — и выписывай ошибки в журнал (день 20).</p>
  {timer('timerA')}
  {stim('Iron supplementation and haemoglobin (Q1–Q12)', '<p>Thirty-two piglets from different litters were randomly assigned at day 3 to one of four doses of injectable iron: 50, 100, 200 or 400 mg (8 piglets per dose). Haemoglobin (g/dL) was measured at weaning. A simple linear regression of haemoglobin on log(dose) (natural log) was fitted.</p>')}
  {pre('A1')}
  {pre('A1_pred')}
  {pre('A1_lof')}
  {''.join(q[:12])}
  {stim('Mycoplasma hyopneumoniae seroprevalence (Q13–Q20)', f'<p>In a cross-sectional survey, one pig was sampled from each of 240 pens across several farms. Age (weeks, 4–24) was recorded and each pig was classified as seropositive (pos = 1) or seronegative (pos = 0); 96 pigs were positive.</p>')}
  {pre('A2')}
  {pre('A2_lrt')}
  {pre('A2_ci')}
  {''.join(q[12:20])}
  {stim('Ammonia and coughing (Q21–Q26)', '<p>In 48 finishing barns, the number of coughs during a 15-minute observation was counted and the ammonia concentration (ppm, 5–30) was measured. Poisson and quasi-Poisson regressions were fitted.</p>')}
  <div class="grid2"><div>{pre('A3')}</div><div>{pre('A3_q')}</div></div>
  {''.join(q[20:])}
</section>
'''


def mockB():
    q = []
    q.append(eq(1, 'Interpret the coefficient for backfat in context.',
                f'<p>Comparing carcasses with the same loin depth and of the same sex, each additional mm of backfat is associated with a {en(-B1["b_backfat"],2)} percentage-point lower mean lean meat percentage.</p>'))
    q.append(eq(2, 'Interpret the coefficient for gilt. What is the reference group?',
                f'<p>Barrows are the reference (gilt = 0). Gilts have a mean lean percentage {en(B1["b_gilt"],2)} points higher than barrows with the same backfat and loin depth (p = 0.042).</p>'))
    lo, hi = B1['b_loin'] - B1['tcrit'] * B1['se_loin'], B1['b_loin'] + B1['tcrit'] * B1['se_loin']
    q.append(eq(3, f'Compute a 95% CI for the loin depth coefficient (t* = 2.000 on 60 df) and interpret it.',
                f'<p>{en(B1["b_loin"],4)} ± 2.000 × {en(B1["se_loin"],4)} = ({en(lo,3)}, {en(hi,3)}). For carcasses with the same backfat and sex, each extra mm of loin depth is associated with 0.02 to 0.14 points higher mean lean percentage.</p>'))
    q.append(eq(4, 'In the simple regression of lean on loin, the slope is −0.018 (p = 0.76). In the multiple regression it is +0.082 (p = 0.008). Explain why.',
                '<p>The two slopes answer different questions. Loin depth is positively correlated with backfat (heavier, fatter pigs also have deeper loins), and backfat strongly lowers lean percentage. In the simple regression, carcasses with deeper loins also tend to have more backfat, which cancels the positive association. The multiple regression compares carcasses with the same backfat (and sex), revealing it.</p>', 'Тот же конфаундинг, что с размером гнезда в дне 11.'))
    q.append(eq(5, 'Predict the mean lean percentage for a gilt with 14 mm backfat and 60 mm loin depth.',
                f'<p>{en(B1["b_Intercept"],3)} − {en(-B1["b_backfat"],4)} × 14 + {en(B1["b_loin"],4)} × 60 + {en(B1["b_gilt"],4)} = {en(B1["b_Intercept"]+B1["b_backfat"]*14+B1["b_loin"]*60+B1["b_gilt"],2)}%.</p>'))
    q.append(eq(6, 'Using the two error sums of squares, carry out the test that compares lean ~ backfat with the full model. State H₀.',
                f'<p>H₀: β_loin = β_gilt = 0. F = [(85.396 − 68.769)/2] / (68.769/60) = 8.314 / 1.146 = {en(B1["cmp"]["F"],2)} on 2 and 60 df, p = {en(B1["cmp"]["p"],4)}. Adding loin depth and sex significantly improves the model with backfat.</p>'))
    q.append(eq(7, 'What null hypothesis does the F-statistic at the bottom of the summary test?',
                '<p>H₀: β_backfat = β_loin = β_gilt = 0, i.e. none of the three explanatory variables is linearly associated with mean lean percentage. F = 64.73 on 3 and 60 df rejects it, but it does not say which variables matter.</p>'))
    q.append(eq(8, 'Give the dimensions of the X matrix and the error degrees of freedom.',
                '<p>64 rows (carcasses) and 4 columns (a column of ones, backfat, loin, gilt). Error df = n − p = 64 − 4 = 60.</p>'))
    # B2
    q.append(eq(9, 'What does the interaction term bw:feederwetdry represent, and what does the test show?',
                f'<p>It allows the slope of feed intake on body weight to differ between feeder types. Its estimate {en(B2["b_int"],5)} (p = 0.00066; the model comparison gives F = {en(B2["cmp"]["F"],2)} = 3.729², same p) is strong evidence that the effect of body weight on intake depends on the feeder type: the lines are not parallel.</p>'))
    q.append(eq(10, 'Give the estimated slope of feed intake on body weight for each feeder type.',
                f'<p>Dry (reference): {en(B2["b_bw"],4)} kg/day per kg. Wet-dry: {en(B2["b_bw"],4)} + {en(B2["b_int"],4)} = {en(B2["b_bw"]+B2["b_int"],4)} kg/day per kg.</p>'))
    q.append(eq(11, 'Estimate the difference in mean intake (wet-dry − dry) at 40 kg and at 90 kg.',
                f'<p>40 kg: {en(B2["b_w"],4)} + {en(B2["b_int"],5)} × 40 = {en(B2["b_w"]+40*B2["b_int"],3)} kg/day. 90 kg: {en(B2["b_w"],4)} + {en(B2["b_int"],5)} × 90 = {en(B2["b_w"]+90*B2["b_int"],3)} kg/day. The advantage of wet-dry feeders grows with body weight.</p>'))
    q.append(eq(12, 'A colleague says: "feederwetdry has p = 0.31, so feeder type does not matter." Do you agree?',
                '<p>No. In a model with an interaction, the feederwetdry coefficient is the difference between feeders at body weight 0 kg, which is outside the data (30–100 kg). It says nothing about the difference at realistic weights, where it is 0.15–0.50 kg/day.</p>'))
    q.append(eq(13, 'Pens were randomly assigned to feeder type. What can you conclude about causality?',
                '<p>Because feeder type was randomized, the difference in intake between feeders (at a given body weight) can be interpreted as caused by the feeder. Body weight itself was not assigned, so its slope is an association.</p>'))
    # B3
    q.append(eq(14, 'Is this still a linear regression model? Explain.',
                '<p>Yes. The model is linear in the parameters: adg = β₀ + β₁·ne + β₂·ne² + ε; ne² is just another explanatory variable.</p>'))
    q.append(eq(15, 'What does the sign of the quadratic coefficient tell you?',
                f'<p>It is negative ({en(B3["b2"],1)}), so the curve is hump-shaped and has a maximum. The quadratic term is significant (p = 0.00011), so a straight line would not be adequate.</p>'))
    q.append(eq(16, 'Estimate the net energy that maximizes ADG and the ADG at that level.',
                f'<p>NE = −b₁/(2b₂) = −{en(B3["b1"],2)} / (2 × {en(B3["b2"],2)}) = {en(B3["xopt"],2)} Mcal/kg; predicted ADG = {en(B3["yopt"],0)} g/day.</p>'))
    q.append(eq(17, 'Why can the coefficient for ne (2092.8) not be interpreted as the change in ADG for a one-unit increase in ne?',
                '<p>Because ne cannot change while ne² is held constant; the effect of ne depends on its current value. The linear coefficient only helps locate the maximum.</p>'))
    q.append(eq(18, 'The VIFs are 1126. Is this a problem here, and how could it be reduced?',
                '<p>ne and ne² are almost perfectly correlated over 2.2–2.7, which inflates the standard errors of β₁ and β₂ and makes them hard to interpret separately. It does not affect the fitted curve, the optimum or predictions. Centering ne (e.g. ne − 2.45) before squaring removes most of the correlation.</p>'))
    # B4
    rows = B4['rows']
    q.append(eq(19, 'Which model is best by AICc? Which models are reasonable alternatives?',
                f'<p>Best: weanwt + creep + temp + stocking (AICc = {en(rows[0]["aicc"],1)}, weight 0.65). Model 2, which also includes weaning age, has ΔAICc = {en(rows[1]["d"],2)} &lt; 2 and is a reasonable alternative. Model 3 (Δ = {en(rows[2]["d"],2)}) is much less supported.</p>'))
    q.append(eq(20, 'How many parameters (k) are counted for the best model, and why?',
                '<p>k = 6: intercept, four slopes (weanwt, creep, temp, stocking) and the residual variance σ².</p>'))
    q.append(eq(21, 'Interpret the VIFs for weanwt (12.5) and weanage (12.1).',
                '<p>Both exceed 10: weaning weight and weaning age are strongly correlated, so each can be predicted well from the other (R² ≈ 0.92). Their standard errors are inflated, which is why weanage is not significant in the full model (p = 0.31) even though age is related to growth.</p>'))
    q.append(eq(22, 'Model selection did not choose weanage. Can you conclude that weaning age has no effect on ADG?',
                '<p>No. Weaning age is highly correlated with weaning weight, so once weight is in the model, age adds little. Model selection chooses between correlated variables somewhat arbitrarily; omitted variables are not shown to be unrelated to the response.</p>'))
    q.append(eq(23, 'Are there influential observations or outliers?',
                f'<p>The largest Cook’s distance is {en(B4["cook_max"],2)}, far below 1, so no observation is unduly influential. Standardized residuals range from {en(B4["rs"][0],2)} to {en(B4["rs"][1],2)}; with 50 pens a few values just beyond ±2 are expected, and none exceed ±3.</p>'))
    q.append(eq(24, 'If BIC had chosen a smaller model than AICc, which would you prefer for prediction and why?',
                '<p>AICc. BIC penalizes each parameter more (ln 50 = 3.9 vs 2), so it selects simpler models. For prediction a slightly too complex model usually loses little, while a too simple model gives biased predictions.</p>'))
    q.append(eq(25, 'The farm wants to estimate the effect of creep feeding. Describe how to choose the covariates.',
                '<p>Run model selection on the potential confounders only (weaning weight, age, temperature, stocking), without creep. Then add creep to the selected model(s) and estimate its coefficient. Check that the estimate is similar across the good models (within ΔAICc ≤ 2) to show the conclusion is robust.</p>'))
    q.append(eq(26, 'Why is the prediction error estimated from the same data used for model selection too optimistic, and what would be better?',
                '<p>The data were used twice, to choose variables and to estimate their coefficients, so the usual standard errors ignore the selection step and in-sample error understates real prediction error. Use a test set or cross-validation to estimate out-of-sample error.</p>'))
    return f'''
<section id="d20">
  <span class="daytag">День 20 · 2,5 ч · Пробный экзамен B</span>
  <h2>Пробный экзамен B (модуль 6): 26 вопросов, 2 часа</h2>
  {timer('timerB')}
  {stim('Carcass lean meat percentage (Q1–Q8)', '<p>At a slaughter plant, 64 carcasses were measured: lean meat percentage (lean), backfat thickness (mm), loin muscle depth (mm) and sex (gilt = 1 for gilts, 0 for barrows). This is observational data.</p>')}
  {pre('B1')}
  {pre('B1_slr')}
  {pre('B1_cmp')}
  {''.join(q[:8])}
  {stim('Feeder type and feed intake (Q9–Q13)', '<p>Forty finishing pens were randomly assigned to dry or wet-dry feeders (20 each). Average daily feed intake (kg/day) and average body weight (kg, 30–100) were recorded for each pen.</p>')}
  {pre('B2')}
  {pre('B2_cmp')}
  {''.join(q[8:13])}
  {stim('Dietary energy and growth (Q14–Q18)', '<p>Thirty-six nursery pens were randomly assigned to one of six diets with net energy 2.20 to 2.70 Mcal/kg (6 pens each). ADG (g/day) was recorded.</p>')}
  {pre('B3')}
  {pre('B3_vif')}
  {''.join(q[13:18])}
  {stim('Nursery growth: diagnostics and model selection (Q19–Q26)', '<p>Records from 50 nursery batches: ADG, weaning weight (kg), weaning age (days), creep feeding (1/0), room temperature (°C) and space allowance (m²/pig). All subsets of the five variables were compared by AICc.</p>')}
  {pre('B4_dredge')}
  {pre('B4_full')}
  {pre('B4_vif')}
  {pre('B4_cook')}
  {''.join(q[18:])}

  <h3>Журнал ошибок</h3>
  <p>Каждую ошибку из пробников и тренажёров — сюда. Сохраняется в браузере.</p>
  <textarea id="errlog" placeholder="Что спросили / что я ответил / какое правило забыл"></textarea>
</section>
'''
