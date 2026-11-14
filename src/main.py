"""
Interactive Terminal Chat with Fine-Tuned Local Models
======================================================
Provides an interactive conversation session with any local Ollama model
(defaults to the domain-specialized `banking-assistant`).

Usage:
    python src/main.py
    python src/main.py --model banking-assistant
    python src/main.py --model mistral:latest
"""

import argparse
import sys
import ollama


DEFAULT_MODEL = "banking-assistant"


def chat(model: str = DEFAULT_MODEL):
    """Interactive command-line chat session."""

    print("=" * 65)
    print(f"🤖 LOCAL LLM INTERACTIVE TERMINAL")
    print(f"   Active Model: {model}")
    print("=" * 65)
    print("Commands:")
    print("  • Type your question or prompt and press Enter.")
    print("  • Type 'exit', 'quit', or 'q' to end the session.\n")

    while True:
        try:
            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() in {"quit", "exit", "q"}:
                print("\nSession ended. Goodbye!")
                break

            print("\nAssistant: ", end="", flush=True)

            # Stream or generate response directly from local Ollama model
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": question}],
            )

            answer = response.get("message", {}).get("content", "").strip()
            print(answer)

        except KeyboardInterrupt:
            print("\n\nSession interrupted by user. Goodbye!")
            sys.exit(0)
        except Exception as error:
            print(f"\n[Error communicating with model '{model}']: {error}")
            print("Please ensure Ollama is running (`ollama list` or `ollama serve`).\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with local Ollama fine-tuned models")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model tag to load (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    chat(args.model)


if __name__ == "__main__":
    main()
