"""Проверка LLM-клиента: GigaChat или OpenRouter."""
from __future__ import annotations

import logging

from src.config import settings
from src.llm_client import get_llm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main() -> None:
    print(f"Провайдер: {settings.LLM_PROVIDER}")
    llm = get_llm()

    print("\nСтатус моделей до запроса:")
    for m in llm.status():
        flag = "active" if m["is_active"] else "idle"
        print(f"  {m['model']} | доступна: {m['available']} | {flag}")

    print("\nОтправляю тестовый запрос...")
    answer = llm.complete(
        "Напиши одно предложение о защите критической информационной "
        "инфраструктуры в энергетике."
    )
    print("\nОтвет LLM:")
    print(answer)

    print("\nСтатус моделей после запроса:")
    for m in llm.status():
        state = "доступна" if m["available"] else f"остывает {m['cooldown_in_sec']} сек"
        flag = "active" if m["is_active"] else "idle"
        print(f"  {m['model']} | {state} | {flag}")


if __name__ == "__main__":
    main()