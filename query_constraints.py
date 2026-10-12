"""Interpret and enforce explicit percentage conditions without model guesses."""
import re

OPERATORS = {"이상": "ge", "이하": "le", "초과": "gt", "미만": "lt", "정확히": "eq", "동일": "eq"}
PERCENT = re.compile(
    r"(실기|수능)(?:전형|반영비율|반영|비중|비율)?(?:이|가|는|은)?"
    r"(\d+(?:\.\d+)?)(?:%|퍼센트|프로)(이상|이하|초과|미만|정확히|동일)?"
)


def parse_percentages(text):
    compact = re.sub(r"\s+", "", text)
    return [{"component": m[1], "operator": OPERATORS.get(m[3], "eq"), "value": float(m[2])}
            for m in PERCENT.finditer(compact)]


def conditions(constraints):
    if constraints.get("percentages"):
        return constraints["percentages"]
    if constraints.get("minimum_percent") is not None:
        return [{"component": constraints["component"], "operator": "ge", "value": constraints["minimum_percent"]}]
    return []


def satisfies(value, condition):
    target = condition["value"]
    return {"ge": value >= target, "le": value <= target, "gt": value > target,
            "lt": value < target, "eq": value == target}[condition["operator"]]


def invalid_conditions(items):
    for component in {c["component"] for c in items}:
        selected = [c for c in items if c["component"] == component]
        if any(not 0 <= c["value"] <= 100 for c in selected):
            return True
        edges = sorted({0.0, 100.0, *(c["value"] for c in selected)})
        candidates = edges + [(a + b) / 2 for a, b in zip(edges, edges[1:])]
        if not any(all(satisfies(v, c) for c in selected) for v in candidates):
            return True
    lower_bounds = []
    for component in {c["component"] for c in items}:
        selected = [c for c in items if c["component"] == component and c["operator"] in ("ge", "gt", "eq")]
        lower = max([0.0] + [c["value"] for c in selected])
        strict = any(c["operator"] == "gt" and c["value"] == lower for c in selected)
        lower_bounds.append((lower, strict))
    total = sum(v for v, _ in lower_bounds)
    if total > 100 or (total == 100 and any(strict for _, strict in lower_bounds)):
        return True
    return False


def text_satisfies_percentages(text, constraints):
    for c in conditions(constraints):
        values = re.findall(re.escape(c["component"]) + r"[^\d\n%]{0,20}(\d+(?:\.\d+)?)\s*%", text)
        if not values or not all(satisfies(float(v), c) for v in values):
            return False
    return True
