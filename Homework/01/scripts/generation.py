import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional
from scripts.model import Model
from scripts.tokenizer import ByteTokenizer


def generate(
        model: Model,
        tokenizer: ByteTokenizer,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        max_length: int = 1024
) -> str:
    """
    Функция для генерации текста с использованием модели и токенизатора.

    Параметры:
    -----------
    model : Model
        Обученная модель для генерации текста. Ожидается, что модель будет поддерживать
        режим предсказаний с возвратом логитов и скрытых состояний (hx).
    tokenizer : ByteTokenizer
        Токенизатор, который преобразует текст в токены и наоборот. Используется для
        кодирования начального и декодирования сгенерированного текста.
    temperature : float, по умолчанию 1.0
        Параметр "температуры", который регулирует степень случайности при генерации.
        При значении 1.0 модель использует стандартное распределение вероятностей.
        При значении меньше 1.0 выбор становится более детерминированным, при значении
        больше 1.0 - более случайным.
    top_k : Optional[int], по умолчанию None
        Если задано, то модель будет выбирать следующий токен из top_k наиболее вероятных.
        Это ограничивает выбор, делая генерацию текста более управляемой.
    max_length : int, по умолчанию 1024
        Максимальное количество токенов, которое будет сгенерировано моделью.

    Возвращает:
    -----------
    str
        Сгенерированная строка текста, декодированная с помощью токенизатора.

    Пример:
    --------
    >>> model = Model()
    >>> tokenizer = ByteTokenizer()
    >>> text = generate(model, tokenizer, temperature=0.7, top_k=10, max_length=100)
    >>> print(text)
    "Пример сгенерированного текста..."

    Логика работы:
    --------------
    - Если температура > 0, производится выбор сэмплированием токенов в зависимости от их
      вероятностей, если же температура = 0, выбирается наиболее вероятный токен.
    - Если указан параметр top_k, модель выбирает следующий токен только из первых k наиболее
      вероятных токенов.
    - Процесс генерации завершается при достижении максимальной длины или если встречен токен
      окончания последовательности (eos_token_id).
    """
    do_sample = temperature > 0
    gen_ids = []
    # Изначально подидим в модель токен начала текста и нулевое состояние
    hx = None
    tokens = torch.tensor([tokenizer.bos_token_id], dtype=torch.long).unsqueeze(0)  # Добавляем размер для батча

    model.eval()
    while len(gen_ids) < max_length:
        with torch.no_grad():
            # Получаем логиты следующего токена и следующее состояние
            logits, hx = model(tokens, hx)

            # Берем только логиты для последнего токена
            logits = logits[:, -1, :]  # Убедитесь, что logits имеют форму [batch_size, vocab_size]

            if not do_sample:
                # Выбираем наиболее вероятный токен
                new_token = torch.argmax(logits, dim=-1).item()  # .item() преобразует тензор в число
            else:
                logits /= temperature
                # Получаем вероятностное распределение следующего токена
                p = F.softmax(logits, dim=-1).cpu().numpy().flatten()  # Приводим к 1D

                ids = np.arange(len(p))
                if top_k is not None:
                    # Выбираем top-k наиболее вероятных токенов
                    top_k_ids = np.argpartition(p, -top_k)[-top_k:]  # Индексы top-k
                    top_k_ids = top_k_ids[np.argsort(-p[top_k_ids])]  # Сортируем по вероятностям
                    p = p[top_k_ids] / p[top_k_ids].sum()  # Нормализуем вероятности
                    new_token = np.random.choice(top_k_ids, p=p)  # Выбираем токен из top-k
                else:
                    new_token = np.random.choice(ids, p=p)  # Общий выбор

        if new_token == tokenizer.eos_token_id:
            break
        gen_ids.append(new_token)
        tokens = torch.tensor([[new_token]], dtype=torch.long)  # Обновляем входные токены, добавляем размер для батча

    return tokenizer.decode(gen_ids)
