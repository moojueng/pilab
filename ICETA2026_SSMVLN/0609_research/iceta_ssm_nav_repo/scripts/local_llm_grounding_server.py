#!/usr/bin/env python3
import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ALLOWED_COLORS = {
    "red", "blue", "green", "yellow", "white", "black", "gray",
    "brown", "purple", "pink", "orange", "any",
}
ALLOWED_ACTIONS = {"log", "patrol", "explore", "stop"}
COLOR_ALIASES = {
    "빨간": "red",
    "빨강": "red",
    "붉은": "red",
    "파란": "blue",
    "파랑": "blue",
    "푸른": "blue",
    "초록": "green",
    "녹색": "green",
    "노란": "yellow",
    "노랑": "yellow",
    "하얀": "white",
    "흰색": "white",
    "검은": "black",
    "검정": "black",
    "회색": "gray",
}
OBJECT_ALIASES = {
    "의자": "chair",
    "침대": "bed",
    "책상": "table",
    "테이블": "table",
    "소파": "sofa",
    "쇼파": "sofa",
    "냉장고": "refrigerator",
    "문": "door",
    "컵": "cup",
    "가방": "bag",
    "노트북": "laptop",
    "티비": "tv",
    "텔레비전": "tv",
    "화분": "plant",
    "쓰레기통": "trash_bin",
}

MODEL = None
TOKENIZER = None
DEVICE = None
CONFIG = None
LOCK = threading.Lock()


def extract_json(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in LLM response: {text[:300]}")
    return json.loads(match.group(0))


def validate_grounding(data, command):
    target_color = normalize_color(data.get("target_color", ""), command)
    target_object = normalize_object(data.get("target_object", "")) or infer_object(command)
    action = normalize_action(data.get("on_detection_action", ""), command)
    if target_color not in ALLOWED_COLORS:
        raise ValueError(f"invalid target_color from LLM: {target_color}")
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,48}", target_object):
        raise ValueError(f"invalid target_object from LLM: {target_object}")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"invalid on_detection_action from LLM: {action}")
    confidence = float(data.get("confidence", 0.75))
    return {
        "raw_command": command,
        "grounding_type": f"local_hf_llm:{CONFIG.model}",
        "llm_used": True,
        "target_color": target_color,
        "target_object": target_object,
        "target_name": f"{target_color}_{target_object}",
        "mission": "exploration_first_target_discovery",
        "mission_mode": "coverage_patrol",
        "on_detection_action": action,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(data.get("reason") or "Local HuggingFace LLM grounded the command."),
    }


def normalize_object(value):
    raw = str(value or "").strip().lower()
    if raw in OBJECT_ALIASES:
        return OBJECT_ALIASES[raw]
    raw = raw.replace("-", "_").replace(" ", "_")
    raw = re.sub(r"[^a-z0-9_]+", "", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw


def normalize_color(value, command):
    raw = str(value or "").strip().lower()
    if raw in ALLOWED_COLORS:
        return raw
    compact = str(command or "").lower().replace(" ", "")
    for word, color in COLOR_ALIASES.items():
        if word in compact:
            return color
    for color in ALLOWED_COLORS:
        if color != "any" and color in compact:
            return color
    return "any"


def infer_object(command):
    compact = str(command or "").lower().replace(" ", "")
    for word, obj in OBJECT_ALIASES.items():
        if word in compact:
            return obj
    for obj in ["chair", "bed", "table", "sofa", "gate", "refrigerator", "cup", "bag", "laptop", "tv", "plant"]:
        if obj in compact:
            return obj
    return "target"


def normalize_action(value, command):
    raw = str(value or "").strip().lower()
    if raw in ALLOWED_ACTIONS:
        return raw
    compact = str(command or "").lower().replace(" ", "")
    if any(word in compact for word in ["정지", "멈춰", "stop"]):
        return "stop"
    if any(word in compact for word in ["순찰", "patrol"]):
        return "patrol"
    if any(word in compact for word in ["탐색", "explore"]):
        return "explore"
    return "log"


def build_messages(command):
    system = (
        "You extract a robot target from Korean or English commands. "
        "Answer with exactly one JSON object and nothing else. "
        "Every field value must be lowercase English. "
        "Never output arrays or candidate lists."
    )
    user = f"""
Allowed mappings:
- 빨간, 빨강, red -> red
- 파란, 파랑, blue -> blue
- 초록, 녹색, green -> green
- 노란, 노랑, yellow -> yellow
- 하얀, white -> white
- 검은, black -> black
- 회색, gray -> gray
- 의자, chair -> chair
- 침대, bed -> bed
- 책상, 테이블, table -> table
- 소파, 쇼파, sofa -> sofa
- 냉장고, refrigerator -> refrigerator
- 컵, cup -> cup
- 가방, bag -> bag
- 노트북, laptop -> laptop
- 로그, 기록, record, log, 남겨 -> log
- 순찰, patrol -> patrol
- 탐색, explore -> explore
- 정지, stop -> stop

If the object is not in the examples, translate it to a short English snake_case noun.
If no color is specified, use any.

Examples:
전체 집을 순찰하면서 빨간 의자를 발견하면 로그 남겨
{{"target_color":"red","target_object":"chair","on_detection_action":"log","confidence":0.95,"reason":"빨간 의자와 로그 명령"}}

파란 침대를 찾으면 기록해
{{"target_color":"blue","target_object":"bed","on_detection_action":"log","confidence":0.95,"reason":"파란 침대와 기록 명령"}}

노란 소파를 발견하면 로그 남겨
{{"target_color":"yellow","target_object":"sofa","on_detection_action":"log","confidence":0.9,"reason":"노란 소파와 로그 명령"}}

Now convert this command:
{command}

Return only the filled JSON object.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_grounding(command):
    messages = build_messages(command)
    if hasattr(TOKENIZER, "apply_chat_template") and TOKENIZER.chat_template:
        prompt = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"SYSTEM: {messages[0]['content']}\nUSER: {messages[1]['content']}\nASSISTANT:"
    inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
    with LOCK:
        with torch.no_grad():
            output = MODEL.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=TOKENIZER.eos_token_id,
            )
    generated = output[0][inputs["input_ids"].shape[-1]:]
    text = TOKENIZER.decode(generated, skip_special_tokens=True).strip()
    return validate_grounding(extract_json(text), command)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            self.send_json({
                "ok": True,
                "model": CONFIG.model,
                "device": str(DEVICE),
                "llm": "huggingface_transformers",
            })
            return
        self.send_json({"ok": False, "message": "not found"}, status=404)

    def do_POST(self):
        if self.path == "/api/ground":
            try:
                payload = self.read_json()
                command = str(payload.get("command", "")).strip()
                if not command:
                    self.send_json({"ok": False, "message": "empty command"}, status=400)
                    return
                self.send_json({"ok": True, "grounding": generate_grounding(command)})
            except Exception as exc:
                self.send_json({"ok": False, "message": str(exc)}, status=500)
            return
        self.send_json({"ok": False, "message": "not found"}, status=404)


def load_model(args):
    global MODEL, TOKENIZER, DEVICE, CONFIG
    CONFIG = args
    DEVICE = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    TOKENIZER = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    MODEL = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    MODEL.to(DEVICE)
    MODEL.eval()
    print(f"Loaded local LLM: {args.model} on {DEVICE}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Local HuggingFace LLM server for VLN/object-goal command grounding.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    load_model(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Local LLM grounding server: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
