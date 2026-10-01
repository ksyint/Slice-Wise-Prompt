"""Region instructions and explicit anatomy vocabulary substitutions."""

import json
import re
from pathlib import Path


WH_WORDS = frozenset(("what", "which", "where", "when", "why", "who", "how"))


class AnatomyLexicon:
    def __init__(self, replacements=None):
        self.replacements = dict(replacements or {})
        for source, target in self.replacements.items():
            if not isinstance(source, str) or not source.strip():
                raise ValueError("Anatomy source terms must be nonempty strings")
            if not isinstance(target, str) or not target.strip():
                raise ValueError("Anatomy target terms must be nonempty strings")
        normalized = [term.casefold() for term in self.replacements]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Anatomy mappings duplicate a case-insensitive source term")
        self.lookup = {term.casefold(): value for term, value in self.replacements.items()}
        alternatives = sorted(self.replacements, key=lambda value: (-len(value), value))
        self.pattern = (
            re.compile(
                r"(?<!\w)(" + "|".join(re.escape(term) for term in alternatives) + r")(?!\w)", re.I
            )
            if alternatives
            else None
        )

    @classmethod
    def read(cls, path):
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError("Anatomy vocabulary must be a JSON object")
        return cls(value)

    def rewrite(self, text):
        substitutions = []
        if self.pattern is None:
            return text, substitutions

        def replace(match):
            source = match.group()
            target = self.lookup[source.casefold()]
            substitutions.append({"source": source, "target": target, "offset": match.start()})
            return target

        return self.pattern.sub(replace, text), substitutions


def question_prefix(question, color="red"):
    if color not in ("blue", "red"):
        raise ValueError("Prompt boundaries must be blue or red")
    text = question.strip()
    if not text:
        raise ValueError("Cannot rewrite an empty question")
    tokens = re.findall(r"\w+", text.casefold())
    prefix = (
        f"Within the {color}-outlined area of the CT volume,"
        if tokens and tokens[0] in WH_WORDS
        else f"Inside the {color} boundary,"
    )
    existing = re.compile(
        r"^(?:Within the (?:red|blue)-outlined area of the CT volume,|Inside the (?:red|blue) boundary,)\s*",
        re.I,
    )
    text = existing.sub("", text)
    return f"{prefix} {text}"


def region_instruction(color="red", thickness=1):
    if color not in ("blue", "red") or thickness < 1 or thickness % 2 == 0:
        raise ValueError("Region instructions need an odd positive contour thickness")
    return (
        f"The CT volume contains a {color} outline with a thickness of {thickness} pixel(s). "
        "Use the outlined region to answer the question. "
        "When several structures occur inside the outline, describe only the requested structure. "
        "For a closed question with no relevant finding, answer No finding. "
        "For an open question with no relevant abnormality, answer No, the requested abnormality is absent. "
        "Keep the answer concise and use the question terminology."
    )


def rewrite_record(record, color="red", thickness=1, lexicon=None):
    lexicon = lexicon or AnatomyLexicon()
    question, replacements = lexicon.rewrite(record["question"])
    question = question_prefix(question, color)
    answer = record["answer"]
    if not record["closed"]:
        answer = re.sub(r"^Within the (?:red|blue) boundary,\s*", "", answer, flags=re.I)
        answer = f"Within the {color} boundary, {answer.strip()}"
    output = dict(record)
    output.update(question=question, answer=answer)
    output["rewrite"] = {
        "source_question": record.get("rewrite", {}).get("source_question", record["question"]),
        "source_answer": record.get("rewrite", {}).get("source_answer", record["answer"]),
        "color": color,
        "thickness": thickness,
        "substitutions": replacements,
    }
    vision = {key: output[key] for key in ("volume", "prompt") if key in output}
    output["conversations"] = [
        {"role": "system", "content": region_instruction(color, thickness)},
        {"role": "user", "content": question},
        {"role": "vision", "content": vision},
        {"role": "assistant", "content": answer},
    ]
    return output


def rewriting_report(before, after):
    if len(before) != len(after):
        raise ValueError("Rewriting report requires aligned record lists")
    return {
        "records": len(before),
        "closed_answers_preserved": all(
            left["answer"] == right["answer"]
            for left, right in zip(before, after)
            if left["closed"]
        ),
        "questions_changed": sum(
            left["question"] != right["question"] for left, right in zip(before, after)
        ),
        "open_answers_changed": sum(
            left["answer"] != right["answer"]
            for left, right in zip(before, after)
            if not left["closed"]
        ),
        "vocabulary_substitutions": sum(len(row["rewrite"]["substitutions"]) for row in after),
    }
