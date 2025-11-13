import os
import re
from pathlib import Path
from typing import Optional

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)


def load_env_file():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


load_env_file()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Please create a .env file with GEMINI_API_KEY=<your key>.")
GEMINI_MODEL = "models/gemini-2.5-flash-lite"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:generateContent"

ALLOWED_ORIGIN = "http://localhost:5002"
EMOTION_OPTIONS = [
    "Disgust",
    "Fear",
    "Anger",
    "Ecstasy",
    "Amusement",
    "Amazement",
    "Pride",
    "Adoration",
    "Desire",
    "Interest",
    "Neutral",
    "Serenity",
    "Contentment",
    "Relief",
    "Cuteness",
    "Realization",
    "Confusion",
    "Embarassment",
    "Distress",
    "Disappointment",
    "Sadness",
    "Pain",
    "Guilt",
]
EMOTION_LOOKUP = {name.lower(): name for name in EMOTION_OPTIONS}
LANGUAGE_OPTIONS = [
    "English (US)",
    "Korean",
    "Japanese",
    "Bulgarian",
    "Czech",
    "Danish",
    "Greek",
    "Spanish",
    "Estonian",
    "Finnish",
    "Hungarian",
    "Italian",
    "Dutch",
    "Polish",
    "Portuguese",
    "Romanian",
    "Arabic",
    "German",
    "French",
    "Hindi",
    "Indonesian",
    "Russian",
    "Vietnamese",
]
LANGUAGE_LOOKUP = {name.lower(): name for name in LANGUAGE_OPTIONS}
LENGTH_TOLERANCE_LOWER = 2
LENGTH_TOLERANCE_UPPER = 4


def build_prompt(
    base_text: str,
    target_length: int,
    previous_length: Optional[int] = None,
    target_emotion: Optional[str] = None,
    target_language: Optional[str] = None,
) -> str:
    target_length = max(1, target_length)
    feedback = ""
    if previous_length is not None and (
        previous_length < target_length - LENGTH_TOLERANCE_LOWER
        or previous_length > target_length + LENGTH_TOLERANCE_UPPER
    ):
        feedback = (
            f"\nThe prior attempt produced {previous_length} characters. "
            f"Adjust to stay within {target_length - LENGTH_TOLERANCE_LOWER} and "
            f"{target_length + LENGTH_TOLERANCE_UPPER} characters, preferring a longer result over a shorter one."
        )

    instructions = [
        "You are an expert Korean copywriter who rewrites text while keeping the original intent and tone.",
        "Follow these rules strictly:",
        f"- Produce about {target_length} characters (count every visible character and space), staying between {target_length - LENGTH_TOLERANCE_LOWER} and {target_length + LENGTH_TOLERANCE_UPPER} characters.",
        "- If you need additional length, extend with coherent, contextually relevant detail; avoid repeating the exact same phrases.",
        "- Maintain fluency in Korean, preserve key information, and keep the original speaker’s voice and intensity.",
        "- Preserve the original register and sentiment: honorifics stay honorifics, casual speech stays casual, and profanity or strong language must remain explicit (you may amplify it if needed to reach length).",
        "- Avoid trailing padding characters (like ▒) and avoid leading/trailing whitespace.",
        "- Return only the rewritten sentence with no explanations or numbering.",
        "- 이모지(emoji)는 사용하지 마세요.",
    ]

    if target_emotion:
        instructions.append(
            f"- Rewrite the text so it powerfully conveys the emotion '{target_emotion}'. Keep the original context while matching the requested emotional intensity."
        )
        instructions.append(
            "- 해당 감정이 적극적으로 드러나도록 문장을 적극 개선해줘. 필요하다면 문맥만 유지한 채 문장 구조를 바꿔도 좋아."
        )

    if target_language:
        instructions.append(
            f"- Produce the final text entirely in {target_language}. If the source is already in that language, refine it while keeping the meaning."
        )

    prompt = "\n".join(instructions)
    prompt += f"{feedback}\n\nOriginal text:\n{base_text}\n"
    return prompt


def build_emotion_prompt(text: str) -> str:
    options = ", ".join(EMOTION_OPTIONS)
    return (
        "Analyze the following text and select exactly one emotion name from the list provided."
        "\nRespond with only the chosen emotion (case-insensitive match to the list)."
        f"\nAvailable emotions: {options}."
        "\n\nText:\n"
        f"{text}\n"
    )


def build_language_prompt(text: str) -> str:
    options = ", ".join(LANGUAGE_OPTIONS)
    return (
        "Determine which language from the given list best matches the text snippet."
        "\nRespond with only the selected language name exactly as shown in the list."
        f"\nAvailable languages: {options}."
        "\n\nText:\n"
        f"{text}\n"
    )


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


def call_gemini(prompt: str) -> str:
    response = requests.post(
        GEMINI_ENDPOINT,
        params={"key": GEMINI_API_KEY},
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                    ],
                }
            ]
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return (
        payload.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )


def analyze_emotion(text: str) -> str:
    prompt = build_emotion_prompt(text)
    try:
        result = call_gemini(prompt).strip()
        if not result:
            return "Neutral"
        lower_result = result.lower()
        if lower_result in EMOTION_LOOKUP:
            return EMOTION_LOOKUP[lower_result]
        tokens = re.findall(r"[a-z]+", lower_result)
        for token in tokens:
            if token in EMOTION_LOOKUP:
                return EMOTION_LOOKUP[token]
    except requests.RequestException:
        pass
    return "Neutral"


def analyze_language(text: str) -> str:
    prompt = build_language_prompt(text)
    try:
        result = call_gemini(prompt).strip()
        if not result:
            return "Unknown"
        lower_result = result.lower()
        if lower_result in LANGUAGE_LOOKUP:
            return LANGUAGE_LOOKUP[lower_result]
        tokens = re.findall(r"[a-z()]+", lower_result)
        for token in tokens:
            if token in LANGUAGE_LOOKUP:
                return LANGUAGE_LOOKUP[token]
    except requests.RequestException:
        pass
    return "Unknown"


@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    base_text = (data.get("baseText") or "").strip()
    target_length = int(data.get("targetLength") or 0)
    target_emotion = data.get("targetEmotion") or None
    target_language = data.get("targetLanguage") or None
    analysis_only = bool(data.get("analysisOnly"))

    if not base_text:
        return jsonify({"text": "", "error": "missing_base_text"}), 400
    if target_length <= 0:
        target_length = len(base_text)

    if analysis_only:
        emotion = analyze_emotion(base_text)
        language = analyze_language(base_text)
        return jsonify({"text": base_text, "emotion": emotion, "language": language}), 200

    try:
        text = ""
        previous_len: Optional[int] = None

        for _ in range(4):
            prompt = build_prompt(
                base_text,
                target_length,
                previous_len,
                target_emotion=target_emotion,
                target_language=target_language,
            )
            text = call_gemini(prompt).strip()
            current_len = len(text)
            if (
                target_length - LENGTH_TOLERANCE_LOWER
                <= current_len
                <= target_length + LENGTH_TOLERANCE_UPPER
            ):
                break
            previous_len = current_len

        emotion = analyze_emotion(text)
        language = analyze_language(text)

        return jsonify({"text": text, "emotion": emotion, "language": language}), 200
    except requests.RequestException as exc:
        status = (
            exc.response.status_code
            if getattr(exc, "response", None) is not None
            else 502
        )
        detail = (
            exc.response.text
            if getattr(exc, "response", None) is not None
            else str(exc)
        )
        return jsonify({"text": base_text, "error": detail}), status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

