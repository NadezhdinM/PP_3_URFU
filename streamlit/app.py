import streamlit as st
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer

# Конфигурация интерфейса Streamlit
st.set_page_config(
    page_title="AI Review Creator",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("Генератор отзывов с использованием AI")
st.write("Генерируйте оригинальные отзывы о местах по заданным параметрам: категория, рейтинг и ключевые фразы.")
st.sidebar.title("Настройки генерации")

# Кэшируем модель и токенизатор для быстрой загрузки
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained('model')  # Загрузка обученной модели
    tokenizer = AutoTokenizer.from_pretrained('model')  # Загрузка токенизатора
    return model, tokenizer


# Генерация текста на основе входных данных
def generate_review(prompt, model, tokenizer, options):  
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=options['max_length'],
        num_return_sequences=options['num_return_sequences'],
        no_repeat_ngram_size=options['no_repeat_ngram_size'],
        do_sample=options['do_sample'],
        top_p=options['top_p'],
        top_k=options['top_k'],
        temperature=options['temperature'],
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def clean_and_format(text):
    """Форматирование текста: разделение на предложения и добавление заглавных букв."""
    content = text.split(":")[-1].strip()
    sentences = []
    current_sentence = []
    
    for char in content:
        current_sentence.append(char)
        if char in '.!?':  # Конец предложения
            sentences.append(''.join(current_sentence).strip())
            current_sentence = []
    
    if current_sentence:  # Добавляем остаток текста
        sentences.append(''.join(current_sentence).strip())

    # Исправляем каждое предложение
    formatted_sentences = []
    for sentence in sentences:
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]  # Заглавная буква
            if not sentence.endswith('.'):
                sentence += '.'  # Завершаем точкой
            formatted_sentences.append(sentence)
    
    return ' '.join(formatted_sentences)


# Основная функция приложения
def main():
    model, tokenizer = load_model()

    if 'is_generated' not in st.session_state:
        st.session_state['is_generated'] = False

    # Настройки модели
    options = {}
    options['max_length'] = st.sidebar.slider('Максимальная длина текста', 50, 300, 150)
    options['num_return_sequences'] = st.sidebar.number_input('Количество текстов', 1, 5, 1)
    options['no_repeat_ngram_size'] = st.sidebar.number_input('N-граммы (без повторений)', 1, 10, 2)
    options['do_sample'] = st.sidebar.checkbox('Случайная генерация', True)
    options['top_p'] = st.sidebar.slider('Top-p вероятность', 0.01, 1.00, 0.95, 0.05)
    options['top_k'] = st.sidebar.number_input('Top-k выборка', 1, 100, 50)
    options['temperature'] = st.sidebar.slider('Температура', 0.01, 2.00, 0.90, 0.05)

    # Ввод параметров пользователя
    place_category = st.text_input("Укажите категорию:", "Ресторан")
    user_rating = st.slider("Выберите рейтинг", 1, 5, 4)
    keywords = st.text_input("Ключевые фразы", "цена, обслуживание, меню")

    # Сбор данных в единый запрос
    user_prompt = f"Категория: {place_category}; Рейтинг: {user_rating}; Ключевые слова: {keywords} -> Отзыв:"

    # Кнопка генерации отзыва
    if st.button('Сгенерировать отзыв'):
        with st.spinner('Подождите, текст генерируется...'):
            generated_review = generate_review(user_prompt, model, tokenizer, options)
            formatted_review = clean_and_format(generated_review)
        st.success("Ваш отзыв готов!")
        st.text_area("Сгенерированный отзыв", formatted_review, height=200)


if __name__ == "__main__":  
    main()
