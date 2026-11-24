
"""Find which Gemini model works for ClaimsPilot."""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from claimspilot.agent import TOOLS
from claimspilot.evaluation import compare, load_messy, messy_truth
from claimspilot.extract import llm_extract
from claimspilot.llm import GeminiClient

DEFAULT_MODELS = [
    "gemini-3.5-flash-lite",
]

models = sys.argv[1:] or DEFAULT_MODELS

KEY = os.environ.get("GEMINI_API_KEY")

if not KEY:
    print("ERROR: GEMINI_API_KEY not found.")
    print("Check that your .env file exists and contains your key.")
    sys.exit(1)

note = load_messy()[7]
truth = messy_truth(note)


def short(error: Exception) -> str:
    return f"{type(error).__name__}: {str(error)[:160]}".replace("\n", " ")


print(f"{'MODEL':<28} {'TEXT':<7} {'TOOL':<7} {'EXTRACT':<10} {'SECS':<7} NOTES")
print("-" * 90)

for model in models:
    start = time.time()
    text_ok = False
    tool_ok = False
    fields = "-"
    notes = ""

    try:
        # Create the client using the existing GeminiClient interface.
        client = GeminiClient(model=model, api_key=KEY)

        # Test 1: Basic text generation.
        response = client.generate(
            "Reply tersely.",
            [{"role": "user", "content": "Say OK"}],
            max_tokens=30,
        )
        text_ok = "ok" in response.text.lower()

        # Test 2: Function calling.
        response = client.generate(
            "You must call the run_audit tool.",
            [{"role": "user", "content": "Audit the claim now."}],
            tools=TOOLS,
        )
        tool_ok = any(
            call.name == "run_audit"
            for call in response.tool_calls
        )

        # Test 3: Structured claim extraction.
        extracted = llm_extract(
            note["notes"],
            "M08",
            note["policy_id"],
            client,
        )

        result = compare(extracted.claim, truth)
        fields = f"{sum(result.values())}/{len(result)}"

        wrong = [
            field
            for field, correct in result.items()
            if not correct
        ]

        notes = f"Wrong fields: {wrong}" if wrong else "All fields matched"

    except Exception as error:
        notes = short(error)

    elapsed = time.time() - start

    print(
        f"{model:<28} "
        f"{str(text_ok):<7} "
        f"{str(tool_ok):<7} "
        f"{fields:<10} "
        f"{elapsed:<7.1f} "
        f"{notes}"
    )

    time.sleep(2)

print("\nProbe completed.")
