# === ЭТОТ КОД ДУБЛИРУЕТ КОД ОСНОВНОГО ПРОЕКТА ДЛЯ РАБОТЫ С ВЕБ-КАМЕРОЙ ===

from ultralytics import YOLO
import cv2
import time

# === ПАРАМЕТРЫ ОТСЛЕЖИВАНИЯ ===
DEAD_ZONE_X = 150  # Мёртвая зона по горизонтали (px)
DEAD_ZONE_Y = 100  # Мёртвая зона по вертикали (px)
TARGET_ID = None  # ID отслеживаемого человека (автовыбор при первом обнаружении)

# Индексы ключевых точек (формат COCO)
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

# Загрузка предобученной модели YOLO
model = YOLO("yolov8n-pose.pt")  # Автоматически загрузит модель при первом запуске

# Открытие видеопотока с веб-камеры (0 — индекс камеры по умолчанию)
cap = cv2.VideoCapture(0)

# Ждём первый валидный кадр
while True:
    ret, frame = cap.read()
    if ret and frame is not None and frame.size > 0:
        print("Первый кадр получен. Запуск основного цикла.")
        break
    else:
        print("Ожидание кадра...")
        time.sleep(0.1)

# Стабилизация после запуска потока
time.sleep(1)
print("Отслеживание запущено. Нажмите 'q' для выхода.")

# Остановить все движения при старте
print("Сбросить все скорости")


# === ФУНКЦИИ ОБРАБОТКИ ПОЗЫ ===


def calculate_body_center(keypoints):
    """
    Вычисляет центр тела как среднее между плечами и бёдрами.
    :param keypoints: список ключевых точек [x, y] для человека
    :return: (cx, cy) — координаты центра тела
    """
    points = [
        keypoints[i] for i in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    ]
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return int(sum(xs) / 4), int(sum(ys) / 4)


def calculate_body_area(keypoints):
    """
    Вычисляет площадь четырёхугольника по ключевым точкам (плечи + бёдра).
    :param keypoints: список ключевых точек
    :return: площадь в пикселях
    """
    x1, y1 = keypoints[LEFT_SHOULDER]
    x2, y2 = keypoints[RIGHT_SHOULDER]
    x3, y3 = keypoints[RIGHT_HIP]
    x4, y4 = keypoints[LEFT_HIP]

    area = (
        abs(
            (x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1)
            - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1)
        )
        / 2
    )

    return int(area)


def get_command(body_x, body_y, square, cx, cy):
    """
    Формирует команды управления дроном на основе положения тела.
    :param body_x: X-координата центра тела
    :param body_y: Y-координата центра тела
    :param square: площадь области тела (приближение расстояния)
    :param cx: центр кадра по X
    :param cy: центр кадра по Y
    """
    # Определяем команду по условиям
    if body_x > cx + DEAD_ZONE_X:
        command = "GO RIGHT"
    elif body_x < cx - DEAD_ZONE_X:
        command = "GO LEFT"
    elif body_y > cy + DEAD_ZONE_Y:
        command = "GO DOWN"
    elif body_y < cy - DEAD_ZONE_Y:
        command = "GO UP"
    elif square > 30000:
        command = "GO BACK"
    elif square < 7000:
        command = "GO FORWARD"
    else:
        command = "STOP"

    # Вывод в консоль
    print(command)

    # Один вызов putText вместо шести
    cv2.putText(
        annotated_frame,
        command,
        (w - 125, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )


# === ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ===
try:
    while True:

        # Получение кадра с вебкамеры
        ret, frame = cap.read()

        # Проверка на пустой кадр
        if not ret or frame is None or frame.size == 0:
            print("Пустой кадр — пропуск...")
            continue

        # Подготовка изображения
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Обнаружение и трекинг позы
        results = model(frame, verbose=False)
        result = results[0]
        keypoints = (
            result.keypoints.xy.cpu().tolist() if result.keypoints is not None else []
        )

        # Визуализация: рисуем скелет, но без bounding box
        annotated_frame = result.plot(boxes=False)

        # Отображаем центр кадра
        cv2.circle(annotated_frame, (center_x, center_y), 8, (255, 0, 0), -1)  # Красный

        # Создание переменных для поиска ближайшего  человека
        biggest_area = 0
        body_center = None

        # Обработка обнаруженных людей
        if len(keypoints) > 0:
            for kpts in keypoints:
                # Проверяем, что достаточно ключевых точек (COCO: минимум 13, чтобы прочитать kpts[12])
                if len(kpts) >= 13:
                    try:
                        area = calculate_body_area(kpts)
                        cx_body, cy_body = calculate_body_center(kpts)

                        if area < 5000:
                            continue  # Пропускаем слишком маленькие объекты

                        # Выбираем самого крупного ЧЕЛОВЕКА, который примерно в центре
                        if (
                            area > biggest_area * 1.2
                            and abs(cx_body - center_x) < w * 0.4
                        ):
                            biggest_area = area
                            body_center = (cx_body, cy_body)
                    except Exception as e:
                        continue  # Пропускаем повреждённые данные

        # --- Визуализация цели ---
        if body_center is not None:
            # Круг в центре тела
            cv2.circle(annotated_frame, body_center, 10, (0, 0, 255), -1)

            # Текст с координатами
            cv2.putText(
                annotated_frame,
                f"Body Center",
                (body_center[0] + 15, body_center[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            # Площадь тела (остаётся как есть)
            cv2.putText(
                annotated_frame,
                f"Body Area: {biggest_area}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        # Если человека нет — отправляем стоп-команду
        else:
            cv2.putText(
                annotated_frame,
                f"STOP",
                (w - 125, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        # --- Управление дроном ---
        if body_center is not None:
            get_command(
                body_center[0], body_center[1], biggest_area, center_x, center_y
            )

        # Отображение кадра
        cv2.imshow("Pose Tracking", annotated_frame)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Обработка исключений
except KeyboardInterrupt:
    print("\nПрерывание по Ctrl+C.")

# Завершение работы
finally:
    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()
