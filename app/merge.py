import re
from .models import RuleRecord


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def validate_sentence(sentence: str) -> str:
    sentence = sentence.strip().strip('"`')
    if not sentence or "{" in sentence or "}" in sentence or "```" in sentence:
        raise ValueError("model output is not a rule sentence")
    if len(re.findall(r"[.!?](?:\s|$)", sentence)) > 1:
        raise ValueError("model output contains more than one sentence")
    return sentence


def append_rule(rule: RuleRecord, sentence: str, program: dict | None = None) -> bool:
    sentence = validate_sentence(sentence)
    existing = {normalize(x) for x in rule.DETAILED_RULE}
    if normalize(sentence) in existing:
        return False
    rule.DETAILED_RULE.append(sentence)
    if program is not None:
        rule.PROGRAM = program
    return True
