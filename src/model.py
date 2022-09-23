from tensorflow import keras
from tensorflow.keras import layers
title_features = layers.Embedding(num_words, 64)(title_input)
# Вложим каждое слово текста в 64-мерный вектор
body_features = layers.Embedding(num_words, 64)(body_input)

# Сокращаем последовательность вложенных слов заголовка до одного 128-мерного вектора
title_features = layers.LSTM(128)(title_features)
# Сокращаем последовательность вложенных слов заголовка до одного 32-мерного вектора
body_features = layers.LSTM(32)(body_features)

# Объединим все признаки в один вектор с помощью конкатенации
x = layers.concatenate([title_features, body_features, tags_input])

# Добавим логистическую регрессию для прогнозирования приоритета по признакам
priority_pred = layers.Dense(1, activation='sigmoid', name='priority')(x)
# Добавим классификатор отделов прогнозирующий на признаках
department_pred = layers.Dense(num_departments, activation='softmax', name='department')(x)

# Создание сквозной модели, предсказывающей приоритет и отдел
model = keras.Model(inputs=[title_input, body_input, tags_input],
                    outputs=[priority_pred, department_pred])
