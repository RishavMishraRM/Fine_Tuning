"""
Interactive chat with the custom banking-assistant model.

Usage:
    python src/main.py
    python src/main.py --model mistral:latest
"""

import argparse
import ollama


DEFAULT_MODEL = "banking-assistant"


def chat(model):
    """Interactive chat loop."""

    print(f"\nBanking Assistant ({model})")
    print("Type your question or 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": question}],
            )

            answer = response["message"]["content"].strip()
            print(f"\nAssistant: {answer}\n")

        except Exception as error:
            print(f"\nError: {error}\n")


def main():
    parser = argparse.ArgumentParser(description="Chat with banking assistant")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use")
    args = parser.parse_args()

    chat(args.model)


if __name__ == "__main__":
    main()
