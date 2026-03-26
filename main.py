from ultralytics import YOLO
from djitellopy import Tello
import os
import logging
import cv2
import time

# === Настройки окружения ===
os.environ["OPENCV_FFMPEG_SKIP_FRAME_CHECK"] = "1"

# Отключаем логирование Tello (опционально)
logging.getLogger("djitellopy").setLevel(logging.WARNING)

# === ПАРАМЕТРЫ ОТСЛЕЖИВАНИЯ ===
DEAD_ZONE_X = 150  # Мёртвая зона по горизонтали (px)
DEAD_ZONE_Y = 100  # Мёртвая зона по вертикали (px)
TARGET_ID = None  # ID отслеживаемого человека (автовыбор при первом обнаружении)
FRAME_SKIP = 1  # Обрабатывать каждый N-й кадр (ускорение)

# Индексы ключевых точек (формат COCO)
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


# === ИНИЦИАЛИЗАЦИЯ ДРОНА ===
fly = Tello()
fly.connect()

# Остановить все движения при старте
fly.send_rc_control(0, 0, 0, 0)
print(f"Заряд батареи: {fly.get_battery()}%")

# Уменьшить битрейт видео для стабильности при слабом Wi-Fi
# fly.set_video_bitrate(1)  # 1 Мбит/с
# print("Битрейт видео установлен на 1 Мбит/с")


# === ЗАГРУЗКА МОДЕЛИ YOLOv8 POSE ===
model = YOLO("yolov8n-pose.pt")


# === ВКЛЮЧЕНИЕ ВИДЕОПОТОКА ===
fly.streamon()
print("Видеопоток включён. Ожидание первого кадра...")

frame_read = fly.get_frame_read()

# Ждём первый валидный кадр
while True:
    frame = frame_read.frame
    if frame is not None and frame.size > 0:
        print("Первый кадр получен. Запуск основного цикла.")
        break
    else:
        print("Ожидание кадра...")
        time.sleep(0.1)

# Стабилизация после запуска потока
time.sleep(1)
print("Отслеживание запущено. Нажмите 'q' для выхода.")

# fly.takeoff()

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
    if body_x > cx + DEAD_ZONE_X:
        fly.send_rc_control(0, 0, 0, 60)  # Поворот направо
    elif body_x < cx - DEAD_ZONE_X:
        fly.send_rc_control(0, 0, 0, -60)  # Поворот налево
    elif body_y > cy + DEAD_ZONE_Y:
        fly.send_rc_control(0, 0, -60, 0)  # Вниз
    elif body_y < cy - DEAD_ZONE_Y:
        fly.send_rc_control(0, 0, 60, 0)  # Вверх
    elif square > 30000:
        fly.send_rc_control(0, -40, 0, 0)  # Назад (если человек близко)
    elif square < 7000:
        fly.send_rc_control(0, 40, 0, 0)  # Вперёд (если далеко)
    else:
        fly.send_rc_control(0, 0, 0, 0)  # Стоп


# === ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ЦИКЛА ===
last_annotated_frame = None  # Кэшированный аннотированный кадр
frame_counter = 0  # Счётчик кадров для пропуска


# === ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ===
try:
    while True:
        start_time = time.time()

        # Получение кадра с камеры дрона
        frame = frame_read.frame
        frame_counter += 1

        if frame is None or frame.size == 0:
            print("Пустой кадр — пропуск...")
            cv2.waitKey(1)
            continue

        # Подготовка изображения
        my_frame = cv2.resize(frame, (640, 480))
        h, w = my_frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Пропуск кадров для оптимизации (обрабатываем каждый FRAME_SKIP-й)
        if frame_counter % FRAME_SKIP == 0:
            # Обнаружение и трекинг позы
            results = model.track(my_frame, persist=True, verbose=False, imgsz=320)
            result = results[0]

            # Визуализация: рисуем скелет, но без bounding box
            annotated_frame = result.plot(boxes=False)

            # Отображаем центр кадра
            cv2.circle(annotated_frame, (center_x, center_y), 8, (0, 0, 255), -1)

            # Извлечение данных
            boxes = result.boxes
            keypoints = result.keypoints

            if (
                boxes is not None
                and boxes.id is not None
                and len(boxes) > 0
                and keypoints is not None
            ):

                track_ids = boxes.id.int().cpu().tolist()
                kpts_list = keypoints.xy.cpu().tolist()

                # Автовыбор целевого человека (первый обнаруженный)
                if TARGET_ID is None and len(track_ids) > 0:
                    TARGET_ID = track_ids[0]
                    print(f"Выбран ID для отслеживания: {TARGET_ID}")

                # Если целевой человек найден
                if TARGET_ID in track_ids:
                    idx = track_ids.index(TARGET_ID)
                    person_kpts = kpts_list[idx]

                    body_x, body_y = calculate_body_center(person_kpts)
                    body_square = calculate_body_area(person_kpts)

                    # Визуализация центра тела
                    cv2.circle(annotated_frame, (body_x, body_y), 8, (255, 0, 255), -1)

                    # Отображение площади в кадре
                    cv2.putText(
                        annotated_frame,
                        f"S: {body_square}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )

                    # Отправка управляющих команд
                    get_command(body_x, body_y, body_square, center_x, center_y)
                else:
                    print("Целевой человек не найден. Сброс ID.")
                    TARGET_ID = None
                    annotated_frame = my_frame.copy()  # Без изменений
            else:
                print("Люди не обнаружены.")
                TARGET_ID = None
                annotated_frame = my_frame.copy()

            # Сохраняем аннотированный кадр для отображения
            last_annotated_frame = annotated_frame

        # Отображение: всегда показываем последний валидный кадр
        display_frame = (
            last_annotated_frame if last_annotated_frame is not None else my_frame
        )
        cv2.imshow("Pose Tracking", cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

        # Управление FPS (~30 FPS)
        elapsed = time.time() - start_time
        wait_time = max(1, int(33 - elapsed * 1000))  # ~30 мс
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("\nПрерывание по Ctrl+C.")

finally:
    # Завершение работы
    cv2.destroyAllWindows()
    fly.send_rc_control(0, 0, 0, 0)  # Остановить дрон
    # fly.land()
    print(f"Заряд батареи: {fly.get_battery()}%")
    print("Конец полёта.")
    fly.streamoff()
    fly.end()
