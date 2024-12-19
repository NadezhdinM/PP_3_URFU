import streamlit as st
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================ Конфигурация страницы ================
st.set_page_config(page_title="Генератор отзывов", initial_sidebar_state="expanded")
st.title("Генератор отзывов на основе ИИ")
st.write("Создавайте текстовые отзывы с помощью нейросети на основе категорий, рейтинга и ключевых слов.")

# ================ Загрузка модели ================
@st.cache_resource
def load_model():
    """Загрузка модели и токенизатора (кэшируется для предотвращения перезагрузки)."""
    model_path = "./model"
    if not os.path.exists(model_path):
        st.error(f"Модель не найдена по пути {model_path}. Проверьте путь.")
        st.stop()
    try:
        with st.spinner("Загрузка модели..."):
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        st.stop()

# ================ Рейтинг звёздочками ================
def star_rating():
    """Выбор рейтинга с отображением звёздочек."""
    stars = ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]
    selected_star = st.radio("Рейтинг:", range(1, 6), format_func=lambda x: stars[x - 1], help="Выберите оценку от 1 до 5 звёзд.")
    return selected_star

# ================ Форматирование текста ================
def format_text(text):
    """Форматирует текст: заглавные буквы и точки."""
    sentences = text.split(". ")
    formatted = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            if not sentence[0].isupper():
                sentence = sentence.capitalize()
            if not sentence.endswith((".", "!", "?")):
                sentence += "."
            formatted.append(sentence)
    return " ".join(formatted)

# ================ Потоковая генерация текста ================
def stream_generate(model, tokenizer, prompt, params, max_length, batch_size):
    """Генерирует текст блоками с обновлением контекста."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = input_ids

    for _ in range(max_length // batch_size):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=output_ids,
                max_new_tokens=batch_size,
                temperature=params["temperature"],
                top_p=params["top_p"],
                top_k=params["top_k"],
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=params["no_repeat_ngram_size"],
            )
            new_tokens = outputs[0, output_ids.shape[1]:]
            output_ids = torch.cat([output_ids, new_tokens.unsqueeze(0)], dim=-1)  # Обновляем контекст
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            yield new_text.strip()

            if tokenizer.eos_token_id in new_tokens:
                break

# ================ Основное приложение ================
def main():
    model, tokenizer = load_model()

    # Основные параметры
    category = st.selectbox("Категория:", ["Кафе", "Ресторан", "Парк", "Музей"])
    rating = star_rating()
    key_words = st.text_input("Ключевые слова", "вкусно, уютно, быстро", help="Введите ключевые слова для использования в отзыве.")

    # Стили генерации
    style = st.selectbox("Стиль генерации:", ["Строгий", "Умеренный", "Безумный"],
                         help="Выберите стиль текста: Строгий – формальный и точный; Безумный – творческий и креативный.")
    style_params = {"Строгий": (0.3, 30), "Умеренный": (0.7, 50), "Безумный": (1.5, 100)}
    temperature, top_k = style_params[style]

    # Дополнительные настройки
    with st.expander("Дополнительные настройки"):
        max_length = st.slider("Максимальная длина", 50, 300, 150, 
                               help="Определяет максимальное количество токенов в тексте. Большее значение увеличивает длину отзыва.")
        num_variants = st.number_input("Количество вариантов", 1, 5, 1, 
                                       help="Сколько вариантов текста будет сгенерировано.")
        batch_size = st.slider("Размер батча токенов", 1, 20, 5, 
                               help="Количество токенов, генерируемых за один шаг. Больше – быстрее, но менее плавный вывод.")
        temperature = st.slider("Температура", 0.01, 2.0, temperature, 0.05, 
                                help="Регулирует степень случайности текста. Низкие значения делают текст более предсказуемым.")
        top_p = st.slider("Top-p", 0.01, 1.0, 0.9, 0.05, 
                          help="Фильтрует токены с низкой вероятностью. Меньшие значения делают текст более логичным.")
        top_k = st.number_input("Top-k", 1, 100, top_k, 
                                help="Ограничивает количество рассматриваемых токенов на каждом шаге. Меньшие значения делают текст точнее.")
        no_repeat_ngram_size = st.slider("Размер n-грамм для предотвращения повторов", 1, 10, 3, 
                                         help="Предотвращает повторение фраз длиной n-грамм.")

    # Генерация текста
    if st.button("Сгенерировать"):
        if not key_words:
            st.warning("Введите ключевые слова.")
            return

        input_prompt = f"Категория: {category}; Рейтинг: {rating}; Ключевые слова: {key_words} -> Отзыв:"
        st.info("Генерация текстов...")

        placeholders = [st.empty() for _ in range(num_variants)]
        texts = [""] * num_variants

        for i in range(num_variants):
            for chunk in stream_generate(
                model, tokenizer, input_prompt, 
                {"temperature": temperature, "top_p": top_p, "top_k": top_k, "no_repeat_ngram_size": no_repeat_ngram_size},
                max_length, batch_size
            ):
                texts[i] += chunk + " "
                placeholders[i].write(f"**Вариант {i + 1}:**\n{format_text(texts[i])}")

        st.success("Генерация завершена!")

        # Скачивание всех вариантов
        all_texts = "\n\n".join([f"Вариант {i + 1}:\n{format_text(text)}" for i, text in enumerate(texts)])
        st.download_button("Скачать все отзывы", all_texts, "generated_reviews.txt")

if __name__ == "__main__":
    main()
