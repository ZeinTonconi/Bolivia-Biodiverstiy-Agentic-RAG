import os

from dotenv import load_dotenv
load_dotenv()
# test_models_requests.py
import os, requests

k = os.getenv("OPENAI_API_KEY")
print("Key present:", bool(k))
if not k:
    print("Set OPENAI_API_KEY in environment first.")
    raise SystemExit(1)

url = "https://api.openai.com/v1/models"
resp = requests.get(url, headers={"Authorization": f"Bearer {k}"}, timeout=10)
print("Status:", resp.status_code)
print("Headers:")
for h in ("x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"):
    print(" ", h, "=", resp.headers.get(h))
print("Body (first 500 chars):")
print(resp.text[:500])

# test_minimal_completion.py
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
print("Key present:", bool(openai.api_key))

try:
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",    # or "gpt-4o" / "gpt-3.5-turbo" if available in your account
        messages=[{"role":"user","content":"Hello"}],
        max_tokens=1,
        temperature=0
    )
    print("Status: success")
    print("Choices:", resp.get("choices"))
    print("Usage:", resp.get("usage"))
except Exception as e:
    # Print full exception repr to capture underlying API message
    import traceback
    print("Exception:", repr(e))
    traceback.print_exc()
