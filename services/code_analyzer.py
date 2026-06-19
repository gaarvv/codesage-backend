import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.1-8b-instant"


# ── JSON extraction (handles markdown fences + raw JSON) ─────────────────────

def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    # 1. Direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass
    # 2. Strip markdown fences
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 3. Find first {...} block
    m2 = re.search(r"\{[\s\S]*\}", raw)
    if m2:
        try:
            return json.loads(m2.group())
        except Exception:
            pass
    return {"error": "Could not parse response", "raw": raw[:400]}


# ── Prompts ──────────────────────────────────────────────────────────────────

_ANALYZE_SYS = """You are CodeSage, an expert code reviewer.
Analyze the {language} code and respond with ONLY a valid JSON object — no markdown, no backticks, no commentary.

Required JSON structure:
{{
  "summary": "One sentence describing what the code does",
  "quality_score": 7,
  "bugs": [
    {{"line": 3, "severity": "high|medium|low", "issue": "description", "fix": "suggested fix"}}
  ],
  "suggestions": [
    {{"type": "performance|readability|security|best_practice", "description": "improvement"}}
  ],
  "complexity": {{
    "time": "O(n)",
    "space": "O(1)",
    "explanation": "brief reason"
  }},
  "positives": ["one thing the code does well"]
}}"""

_EXPLAIN_SYS = """You are CodeSage. Explain the {language} code clearly for a learner.
Respond with ONLY a valid JSON object — no markdown, no backticks.

Required JSON structure:
{{
  "overview": "What this code does in 1–2 sentences",
  "breakdown": [
    {{"lines": "1–5", "explanation": "what these lines do"}}
  ],
  "key_concepts": ["concept1", "concept2"],
  "example_use": "How you would call or use this code"
}}"""

_CONVERT_SYS = """You are CodeSage. Convert the code from {from_lang} to {to_lang}.
Respond with ONLY a valid JSON object — no markdown, no backticks.

Required JSON structure:
{{
  "converted_code": "the full converted code as a string",
  "notes": "key differences or gotchas between the two languages"
}}"""


# ── Public API ────────────────────────────────────────────────────────────────

def analyze(code: str, language: str) -> dict:
    try:
        res = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _ANALYZE_SYS.format(language=language)},
                {"role": "user", "content": f"Analyze this {language} code:\n\n{code}"},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        return _extract_json(res.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}


def explain(code: str, language: str) -> dict:
    try:
        res = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _EXPLAIN_SYS.format(language=language)},
                {"role": "user", "content": f"Explain this {language} code:\n\n{code}"},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        return _extract_json(res.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}


def convert(code: str, from_language: str, to_language: str) -> dict:
    try:
        res = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": _CONVERT_SYS.format(
                        from_lang=from_language, to_lang=to_language
                    ),
                },
                {
                    "role": "user",
                    "content": f"Convert from {from_language} to {to_language}:\n\n{code}",
                },
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        return _extract_json(res.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}
