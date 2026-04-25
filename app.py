"""
AI 图像生成 Web 应用
启动: python app.py
访问: http://localhost:5000
"""

import base64
import os
import re
import time
from pathlib import Path

import requests
from flask import Flask, render_template, request, jsonify, send_from_directory

# ============ 配置 ============
API_KEY = os.environ.get("API_KEY", "")
BASE_URL = os.environ.get("BASE_URL", "https://aicanapi.com/v1")
DEFAULT_MODEL = "grok-4-image"
OUTPUT_DIR = "output"

IMAGE_API_MODELS = {
    "dall-e-3", "gpt-image-1", "gpt-image-1-all", "gpt-image-1.5-all",
    "gpt-image-2", "flux-kontext-pro", "flux.1-dev",
}

ALL_MODELS = [
    "grok-4-image", "grok-3-image", "dall-e-3",
    "gpt-image-1", "gpt-image-1-all", "gpt-image-1.5-all",
    "gpt-image-2", "flux-kontext-pro", "flux.1-dev",
]

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

app = Flask(__name__)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def encode_image_bytes(file_bytes: bytes, filename: str) -> tuple[str, str]:
    img_b64 = base64.b64encode(file_bytes).decode()
    ext = Path(filename).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/png")
    return img_b64, mime


def save_from_url(url: str, filename: str):
    resp = requests.get(url, timeout=60)
    with open(filename, "wb") as f:
        f.write(resp.content)


def generate_via_image_api(model, prompt, images_data, size, quality, n):
    body = {"model": model, "prompt": prompt, "size": size, "n": n}

    if images_data:
        image_inputs = []
        for img_b64, mime in images_data:
            image_inputs.append({
                "type": "input_image",
                "image_url": f"data:{mime};base64,{img_b64}",
            })
        body["image"] = image_inputs

    resp = requests.post(f"{BASE_URL}/images/generations", headers=HEADERS, json=body, timeout=300)
    if resp.status_code != 200:
        err = resp.json().get("error", {}).get("message", resp.text)
        return None, f"API 错误 ({resp.status_code}): {err}"

    data = resp.json()
    timestamp = int(time.time())
    saved = []
    for i, img in enumerate(data.get("data", [])):
        filename = f"image_{timestamp}_{i}.png"
        filepath = f"{OUTPUT_DIR}/{filename}"
        if img.get("b64_json"):
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(img["b64_json"]))
            saved.append(filename)
        elif img.get("url"):
            save_from_url(img["url"], filepath)
            saved.append(filename)
    return saved, None


def generate_via_chat_api(model, prompt, images_data):
    content = []
    if images_data:
        for img_b64, mime in images_data:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img_b64}"},
            })
    content.append({"type": "text", "text": prompt})

    body = {"model": model, "messages": [{"role": "user", "content": content}]}
    resp = requests.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=body, timeout=300)

    if resp.status_code != 200:
        err = resp.json().get("error", {}).get("message", resp.text)
        return None, f"API 错误 ({resp.status_code}): {err}"

    data = resp.json()
    timestamp = int(time.time())
    saved = []
    text_reply = ""

    msg = data["choices"][0]["message"]
    msg_content = msg.get("content", "")

    if isinstance(msg_content, list):
        for i, block in enumerate(msg_content):
            if block.get("type") == "image_url":
                url = block["image_url"]["url"]
                filename = f"image_{timestamp}_{i}.png"
                filepath = f"{OUTPUT_DIR}/{filename}"
                if url.startswith("data:"):
                    b64_data = url.split(",", 1)[1]
                    with open(filepath, "wb") as f:
                        f.write(base64.b64decode(b64_data))
                else:
                    save_from_url(url, filepath)
                saved.append(filename)
            elif block.get("type") == "text" and block.get("text", "").strip():
                text_reply += block["text"].strip() + "\n"
    elif isinstance(msg_content, str) and msg_content.strip():
        urls = re.findall(r'!\[.*?\]\((https?://[^\s)]+)\)', msg_content)
        urls += re.findall(r'(https?://[^\s"\'<>]+\.(?:png|jpg|jpeg|webp|gif))', msg_content)
        urls = list(dict.fromkeys(urls))
        for i, url in enumerate(urls):
            filename = f"image_{timestamp}_{i}.png"
            filepath = f"{OUTPUT_DIR}/{filename}"
            save_from_url(url, filepath)
            saved.append(filename)
        if not urls:
            text_reply = msg_content

    return saved, text_reply if not saved and text_reply else None


@app.route("/")
def index():
    return render_template("index.html", models=ALL_MODELS, default_model=DEFAULT_MODEL)


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


def build_role_prompt(prompt, has_a, has_b, count_a, count_b):
    """根据上传的图片角色，构造带角色说明的 prompt"""
    if has_a and has_b:
        return (
            f"[第1组图片是「模板/风格参考图」(共{count_a}张)，"
            f"第2组图片是「需要被修改的目标图」(共{count_b}张)。"
            f"请参照模板图的风格/构图来调整目标图] {prompt}"
        )
    elif has_a:
        return f"[以下是「模板/风格参考图」(共{count_a}张)，请参照此风格生成] {prompt}"
    elif has_b:
        return f"[以下是「需要被修改的目标图」(共{count_b}张)] {prompt}"
    return prompt


@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.form.get("prompt", "").strip()
    model = request.form.get("model", DEFAULT_MODEL)
    size = request.form.get("size", "1024x1024")
    quality = request.form.get("quality", "auto")
    n = int(request.form.get("n", 1))

    if not prompt:
        return jsonify({"error": "请输入提示词"}), 400

    # 处理 A（模板图）和 B（目标图）
    images_a = []
    for f in request.files.getlist("images_a"):
        if f and f.filename:
            img_b64, mime = encode_image_bytes(f.read(), f.filename)
            images_a.append((img_b64, mime))

    images_b = []
    for f in request.files.getlist("images_b"):
        if f and f.filename:
            img_b64, mime = encode_image_bytes(f.read(), f.filename)
            images_b.append((img_b64, mime))

    # 合并：A 图在前，B 图在后（顺序对应 prompt 中的角色说明）
    images_data = images_a + images_b
    enhanced_prompt = build_role_prompt(prompt, bool(images_a), bool(images_b), len(images_a), len(images_b))

    try:
        if model in IMAGE_API_MODELS:
            saved, err = generate_via_image_api(model, enhanced_prompt, images_data, size, quality, n)
        else:
            saved, err = generate_via_chat_api(model, enhanced_prompt, images_data)

        if err:
            return jsonify({"error": err}), 500
        if not saved:
            return jsonify({"error": "未能从回复中提取到图片"}), 500

        return jsonify({
            "images": [f"/output/{name}" for name in saved],
            "prompt": prompt,
            "model": model,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
