# Lunar Lander DQN - Dockerfile
# Для единообразия используемой среды и упрощения установки

FROM python:3.10-slim

# Установка системных зависимостей для Box2D и pygame
RUN apt-get update && apt-get install -y \
    swig \
    cmake \
    g++ \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-ttf-dev \
    libsdl2-gfx-dev \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libopenmpi-dev \
    openmpi-bin \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование requirements.txt
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY model.py .
COPY agent.py .
COPY main.py .

# Переменные окружения
ENV PYTHONUNBUFFERED=1

# Команда по умолчанию - запуск в режиме обучения
CMD ["python", "main.py", "--mode", "train", "--episodes", "500"]
