import json
import ollama
import csv
from datetime import datetime

MODEL_NAME = "llama3"  # or "llama3:8b" if you pulled that


def get_trading_signal_from_headline(headline: str):
    """
    Send the headline to the local LLM (Ollama) and get back a structured trading signal.
    The instrument is GBP/USD by default, but you can change it in the prompt.
    """
    prompt = f"""
You are a systematic trading assistant.

Instrument: GBP/USD
Time horizon: next 1-24 hours.

Task:
1. Read the following news headline.
2. Evaluate its *short-term* likely impact on GBP/USD.
3. Respond ONLY in strict JSON with this exact structure:

{{
  "sentiment": "bullish" | "bearish" | "neutral",
  "confidence": float between 0 and 1,
  "direction": "long" | "short" | "flat",
  "reason": "very short explanation in one sentence"
}}

Rules:
- "bullish" means positive for EUR against USD (EUR/USD up).
- "bearish" means negative for EUR against USD (EUR/USD down).
- "neutral" means unclear or no strong edge.
- "long" = buy GBP/USD, "short" = sell GBP/USD, "flat" = no position.
- If the headline is ambiguous, prefer "neutral" and "flat".
- Do NOT include any text before or after the JSON.

Headline: "{headline}"
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    raw_content = response["message"]["content"].strip()

    # Try to parse the JSON strictly
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        print("⚠️ Model did not return valid JSON. Raw response:")
        print(raw_content)
        return None

    # Basic validation
    required_keys = {"sentiment", "confidence", "direction", "reason"}
    if not required_keys.issubset(data.keys()):
        print("⚠️ JSON is missing required keys. Got:")
        print(data)
        return None

    return data


def interpret_signal(signal: dict):
    """
    Convert the raw signal into a simple, human-readable suggestion.
    """
    sentiment = signal["sentiment"]
    confidence = signal["confidence"]
    direction = signal["direction"]
    reason = signal["reason"]

    print("\n--- AI Trading Signal ---")
    print(f"Sentiment : {sentiment}")
    print(f"Direction : {direction}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Reason    : {reason}")

    # Simple rule-based interpretation
    print("\nSuggested action (for testing / demo only):")

    if confidence < 0.6 or direction == "flat" or sentiment == "neutral":
        print("- Stay FLAT (no trade). Confidence too low or neutral signal.")
    else:
        if direction == "long":
            print("- Consider LONG GBP/USD with small risk.")
        elif direction == "short":
            print("- Consider SHORT GBP/USD with small risk.")
        else:
            print("- Stay FLAT (direction unclear).")

    print("\n⚠️ WARNING: This is a toy signal. Use ONLY for learning and backtesting, not live money.")


def main():
    print("News → Trading Signal (Ollama + llama3)")
    print("Instrument: GBP/USD (you can change this in the code).")
    print("Type 'quit' to exit.\n")

    while True:
        headline = input("Enter news headline: ").strip()

        if headline.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if not headline:
            print("Please type a non-empty headline.")
            continue

        signal = get_trading_signal_from_headline(headline)

        if signal is not None:
            log_signal(headline, signal)
            interpret_signal(signal)
        else:
            print("No valid signal returned. Try another headline.\n")


if __name__ == "__main__":
    main()

def log_signal(headline: str, signal: dict, filename: str = "signals_log.csv"):
    """
    Append the headline and signal data to a CSV file.
    """
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "headline": headline,
        "sentiment": signal.get("sentiment"),
        "direction": signal.get("direction"),
        "confidence": signal.get("confidence"),
        "reason": signal.get("reason"),
    }

    file_exists = False
    try:
        with open(filename, "r", newline="", encoding="utf-8") as f:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "headline", "sentiment", "direction", "confidence", "reason"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
