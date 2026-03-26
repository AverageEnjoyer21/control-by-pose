# === ЭТОТ СКРИПТ НУЖЕН ДЛЯ ПРОВЕРКИ ТОГО, ЧТО ВСЕ ЗАВИСИМОСТИ УСТАНОВИЛИСЬ ПРАВИЛЬНО ===
# === ЕГО ПОЛЕЗНО ЗАПУСКАТЬ, ЧТОБЫ ПРОВЕРИТЬ, ЧТО ВСЕ УСТАНОВИЛОСЬ ПРАВИЛЬНО =====

import cv2
from ultralytics import YOLO

# Загрузка предобученной модели YOLO
model = YOLO("yolov8n-pose.pt")  # Автоматически загрузит модель при первом запуске

# Открытие видеопотока с веб-камеры (0 — индекс камеры по умолчанию)
cap = cv2.VideoCapture(0)

# Проверка, удалось ли открыть камеру
if not cap.isOpened():
    print("Ошибка: не удалось подключиться к веб-камере.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр.")
        break

    # Выполнение детекции на текущем кадре
    results = model(frame, verbose=False)

    # Отображение результатов на кадре
    annotated_frame = results[0].plot()  # Метод .plot() рисует рамки и метки

    # Показ кадра
    cv2.imshow("YOLOv11 Обнаружение объектов", annotated_frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
