from __future__ import annotations

import re
from typing import Any


# ``when`` is valid rule language (for example, "extract the value when it is
# adjacent to its label").  Only reject explicit alternate-source/fallback
# instructions here.
_HOPS = re.compile(r"\b(otherwise|fallback|then use|instead use|if not found|regex)\b", re.I)


def validate_sentence(sentence: Any, payload: dict[str, Any]) -> str:
    if not isinstance(sentence, str):
        raise ValueError("sentence must be text")
    value = " ".join(sentence.strip().split()).strip('`"')
    if not value or len(value.split()) > 40 or any(token in value for token in ("{", "}", "```")):
        raise ValueError("sentence failed length or format gate")
    if len(re.findall(r"[.!?](?:\s|$)", value)) > 1 or _HOPS.search(value):
        raise ValueError("sentence contains multiple sentences or fallback hops")
    for key in (payload.get("old_value"), payload.get("new_value")):
        if key is not None and len(str(key)) >= 3 and str(key).casefold() in value.casefold():
            raise ValueError("sentence hard-codes the correction value")
    return value
