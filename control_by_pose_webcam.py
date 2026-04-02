# === ЭТОТ КОД ДУБЛИРУЕТ КОД ОСНОВНОГО ПРОЕКТА с ДРОНОМ===
# === ЕГО МОЖНО ИСПОЛЬЗОВАТЬ ДЛЯ ТЕСТИРОВАНИЯ И ОТЛАДКИ НОВЫХ ФУНКЦИЙ ===

from ultralytics import YOLO
import cv2
import time

# === ПАРАМЕТРЫ ОТСЛЕЖИВАНИЯ ===
DEAD_ZONE_X = 150
DEAD_ZONE_Y = 100

# Индексы ключевых точек (COCO)
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

# Загрузка модели
model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)


# === МОДУЛЬНЫЕ ФУНКЦИИ ===


def wait_for_valid_frame(cap):
    """Ждёт первый валидный кадр."""
    while True:
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            print("Первый кадр получен. Запуск основного цикла.")
            return frame
        else:
            print("Ожидание кадра...")
            time.sleep(0.1)


def calculate_body_center(keypoints):
    """Вычисляет центр тела по плечам и бёдрам."""
    points = [
        keypoints[i] for i in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    ]
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return int(sum(xs) / 4), int(sum(ys) / 4)


def calculate_body_area(keypoints):
    """Вычисляет площадь четырёхугольника по ключевым точкам."""
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


def select_target_person(keypoints_list, frame_width):
    """Выбирает самого крупного человека около центра кадра."""
    biggest_area = 0
    best_center = None

    for kpts in keypoints_list:
        if len(kpts) < 13:
            continue
        try:
            area = calculate_body_area(kpts)
            cx_body, cy_body = calculate_body_center(kpts)
            if area < 5000 or abs(cx_body - frame_width // 2) > frame_width * 0.4:
                continue
            if area > biggest_area * 1.2:
                biggest_area = area
                best_center = (cx_body, cy_body)
        except Exception:
            continue

    return best_center, biggest_area


def get_command(body_x, body_y, area, frame_center_x, frame_center_y):
    """Определяет команду управления дроном."""
    if body_x > frame_center_x + DEAD_ZONE_X:
        return "GO RIGHT"
    elif body_x < frame_center_x - DEAD_ZONE_X:
        return "GO LEFT"
    elif body_y > frame_center_y + DEAD_ZONE_Y:
        return "GO DOWN"
    elif body_y < frame_center_y - DEAD_ZONE_Y:
        return "GO UP"
    elif area > 30000:
        return "GO BACK"
    elif area < 7000:
        return "GO FORWARD"
    else:
        return "STOP"


def annotate_frame(frame, center, area, command, frame_center):
    """Добавляет визуализацию на кадр."""
    h, w = frame.shape[:2]

    # Рисуем центр кадра
    cv2.circle(frame, frame_center, 8, (0, 255, 0), -1)

    # Если человек найден
    if center is not None:
        cv2.circle(frame, center, 10, (0, 0, 255), -1)
        cv2.putText(
            frame,
            "Body Center",
            (center[0] + 15, center[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Body Area: {area}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    # Команда всегда отображается
    cv2.putText(
        frame, command, (w - 125, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
    )


def process_frame(frame):
    """Полная обработка одного кадра: детекция → выбор цели → команда → разметка."""
    h, w = frame.shape[:2]
    frame_center = (w // 2, h // 2)

    # Детекция позы
    results = model(frame, verbose=False)
    result = results[0]
    keypoints_list = (
        result.keypoints.xy.cpu().tolist() if result.keypoints is not None else []
    )

    # Получаем размеченный кадр без bounding box
    annotated_frame = result.plot(boxes=False)

    # Выбираем цель
    body_center, body_area = select_target_person(keypoints_list, w)

    # Определяем команду
    command = (
        get_command(*body_center, body_area, *frame_center) if body_center else "STOP"
    )

    # Добавляем надписи в кадре
    annotate_frame(annotated_frame, body_center, body_area, command, frame_center)

    return annotated_frame, command


def main():
    """Основной цикл приложения."""
    wait_for_valid_frame(cap)
    time.sleep(1)
    print("Отслеживание запущено. Нажмите 'q' для выхода.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("Пустой кадр — пропуск...")
                continue

            annotated_frame, command = process_frame(frame)
            print(command)

            cv2.imshow("Pose Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nПрерывание по Ctrl+C.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
