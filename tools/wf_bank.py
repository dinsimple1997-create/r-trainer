"""End-to-end analyses ('Полный анализ') -> wfdata.json.

Each workflow: a question, a ready data set and 6-11 steps from read.csv to the final check. A step may ask
to pick the right code first; then the console (code + R output, computed here like R) and plots appear,
followed by questions on the output. Variants of the same workflow use different data, so conclusions differ
(effect / no effect, clean data / a planted problem). Run from tools/: python3 wf_bank.py
"""
import json
import numpy as np
import wf_a, wf_b, wf_c

GROUPS = [["means", "Сравнение средних"], ["tables", "Таблицы"], ["regr", "Регрессия"], ["glm", "Логистическая и Пуассон"],
          ["mixed", "Смешанные модели"], ["plan", "Планирование"]]


def build():
    rng = np.random.default_rng(20261121)
    W = wf_a.build_a(rng) + wf_b.build_b(rng) + wf_c.build_c(rng)
    order = {g: i for i, (g, _) in enumerate(GROUPS)}
    W.sort(key=lambda w: order[w["g"]])  # stable: keeps the order inside a group
    seen = {}
    for i, w in enumerate(W):
        w["id"] = i + 1
        seen[w["key"]] = seen.get(w["key"], 0) + 1
        w["var"] = seen[w["key"]]
    for w in W:
        w["nvar"] = seen[w["key"]]
    return W


if __name__ == "__main__":
    W = build()
    json.dump(dict(groups=GROUPS, W=W), open("wfdata.json", "w"), ensure_ascii=False, separators=(",", ":"))
    nq = sum(len(s["qs"]) + (1 if "pick" in s else 0) for w in W for s in w["steps"])
    print(len(W), "workflows,", sum(len(w["steps"]) for w in W), "steps,", nq, "questions")
