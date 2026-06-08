#!/usr/bin/env python3
import json
import os
import re
import urllib.error
import urllib.request


COLOR_KEYWORDS = {
    "red": ["red", "빨간", "빨강", "붉은"],
    "blue": ["blue", "파란", "파랑", "푸른"],
    "green": ["green", "초록", "녹색"],
    "yellow": ["yellow", "노란", "노랑"],
    "white": ["white", "하얀", "흰색"],
    "black": ["black", "검은", "검정"],
    "gray": ["gray", "grey", "회색"],
}

OBJECT_KEYWORDS = {
    "chair": ["chair", "의자"],
    "bed": ["bed", "침대"],
    "table": ["table", "책상", "테이블"],
    "sofa": ["sofa", "소파", "쇼파"],
    "gate": ["gate", "문", "게이트"],
    "refrigerator": ["refrigerator", "fridge", "냉장고"],
    "cup": ["cup", "컵"],
    "bag": ["bag", "가방"],
    "laptop": ["laptop", "노트북"],
    "target": ["target", "목표"],
}

ACTION_KEYWORDS = {
    "log": ["log", "record", "기록", "로그", "남겨"],
    "patrol": ["patrol", "순찰"],
    "explore": ["explore", "탐색"],
    "stop": ["stop", "정지", "멈춰"],
}


def _first_match(text, table, default):
    lower = (text or "").lower()
    compact = lower.replace(" ", "")
    for key, words in table.items():
        if any(word in lower or word in compact for word in words):
            return key
    return default


def symbolic_grounding(command):
    target_color = _first_match(command, COLOR_KEYWORDS, "any")
    target_object = _first_match(command, OBJECT_KEYWORDS, "chair")
    action = _first_match(command, ACTION_KEYWORDS, "log")
    return {
        "raw_command": command,
        "grounding_type": "offline_symbolic_fallback",
        "llm_used": False,
        "target_color": target_color,
        "target_object": target_object,
        "target_name": f"{target_color}_{target_object}",
        "mission": "exploration_first_target_discovery",
        "mission_mode": "coverage_patrol",
        "on_detection_action": action,
        "confidence": 0.62,
        "reason": "Keyword fallback used because a local LLM response was unavailable.",
    }


def _json_object_from_text(text):
    text = (text or "").strip()
    if not text:
        raise ValueError("empty LLM response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(match.group(0))


def _normalize_grounding(data, command, grounding_type):
    target_color = normalize_token(data.get("target_color") or data.get("color") or "any", default="any")
    target_object = normalize_token(data.get("target_object") or data.get("object") or "target", default="target")
    action = str(data.get("on_detection_action") or data.get("action") or "log").lower()
    mission = str(data.get("mission") or "exploration_first_target_discovery")
    confidence = float(data.get("confidence", 0.8))
    return {
        "raw_command": command,
        "grounding_type": grounding_type,
        "llm_used": True,
        "target_color": target_color,
        "target_object": target_object,
        "target_name": f"{target_color}_{target_object}",
        "mission": mission,
        "mission_mode": str(data.get("mission_mode") or "coverage_patrol"),
        "on_detection_action": action,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(data.get("reason") or "Local LLM grounded the natural-language command."),
    }


def normalize_token(value, default):
    text = str(value or default).strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def ground_with_ollama(command, model=None, endpoint=None, timeout=8.0):
    model = model or os.environ.get("OLLAMA_MODEL", "llama3.1")
    endpoint = endpoint or os.environ.get("OLLAMA_ENDPOINT", "http://127.0.0.1:11434/api/generate")
    prompt = f"""
You are a compact VLN command grounding module for an indoor exploration robot.
Convert the command into a JSON object only. No markdown.

target_object can be any short English snake_case indoor object noun, such as chair, bed, table, sofa, refrigerator, cup, bag, laptop.
Allowed target_color values include red, blue, green, yellow, white, black, gray, brown, purple, pink, orange, any.
Allowed on_detection_action values: log, patrol, explore, stop.
mission must be exploration_first_target_discovery.
mission_mode must be coverage_patrol.

Command:
{command}

JSON schema:
{{
  "target_object": "chair",
  "target_color": "red",
  "on_detection_action": "log",
  "mission": "exploration_first_target_discovery",
  "mission_mode": "coverage_patrol",
  "confidence": 0.0,
  "reason": "short reason"
}}
""".strip()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    data = _json_object_from_text(body.get("response", ""))
    return _normalize_grounding(data, command, f"local_ollama_llm:{model}")


def ground_with_hf_local(command, endpoint=None, timeout=30.0):
    endpoint = endpoint or os.environ.get("S_NAV_LOCAL_LLM_ENDPOINT", "http://127.0.0.1:8790/api/ground")
    payload = {"command": command}
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not body.get("ok"):
        raise RuntimeError(body.get("message", "local HuggingFace LLM returned an error"))
    grounding = body.get("grounding")
    if not isinstance(grounding, dict):
        raise RuntimeError("local HuggingFace LLM did not return grounding JSON")
    grounding["raw_command"] = command
    grounding["llm_used"] = True
    return grounding


def ground_command(command, provider="auto", model=None, endpoint=None, require_llm=False):
    provider = (provider or "auto").lower()
    if provider in ("hf_local", "local_hf", "transformers"):
        return ground_with_hf_local(command, endpoint=endpoint)
    if provider in ("ollama", "auto"):
        try:
            return ground_with_ollama(command, model=model, endpoint=endpoint)
        except (OSError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            if provider == "ollama" or require_llm:
                raise RuntimeError(f"local LLM grounding failed: {exc}") from exc
    if provider not in ("auto", "symbolic", "rules", "rule"):
        raise ValueError(f"unsupported LLM provider: {provider}")
    return symbolic_grounding(command)
