#!/usr/bin/env python3
"""
local_inference_server.py
Lightweight OpenAI-compatible inference server using HuggingFace transformers.
Avoids vLLM multiprocessing issues.

Usage:
  CUDA_VISIBLE_DEVICES=2 python baselines/local_inference_server.py \
      --model /path/to/model --port 8010
"""
from __future__ import annotations
import argparse, json, time, threading, uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = None
TOKENIZER = None
LOCK = threading.Lock()


def load_model(model_path: str, dtype: str = "bfloat16"):
    global MODEL, TOKENIZER
    print(f"[server] Loading tokenizer from {model_path}...")
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"[server] Loading model...")
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    MODEL.eval()
    print(f"[server] Model loaded. Device map: {MODEL.hf_device_map if hasattr(MODEL, 'hf_device_map') else 'auto'}")


def generate(messages: list[dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
    text = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = TOKENIZER(text, return_tensors="pt").to(MODEL.device)
    with torch.no_grad():
        out = MODEL.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 1e-6),
            do_sample=temperature > 0.01,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return TOKENIZER.decode(new_tokens, skip_special_tokens=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default access log

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": ARGS.served_name, "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.0)
        t0 = time.time()
        with LOCK:
            content = generate(messages, max_tokens=max_tokens, temperature=temperature)
        elapsed = time.time() - t0
        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(t0),
            "model": ARGS.served_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)
        print(f"[server] {elapsed:.2f}s | {content[:80]!r}")


ARGS = None

def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-name", default=None)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--dtype", default="bfloat16")
    ARGS = parser.parse_args()
    if ARGS.served_name is None:
        ARGS.served_name = Path(ARGS.model).name
    load_model(ARGS.model, ARGS.dtype)
    server = HTTPServer(("0.0.0.0", ARGS.port), Handler)
    print(f"[server] Listening on port {ARGS.port}  model={ARGS.served_name}")
    server.serve_forever()


if __name__ == "__main__":
    main()
