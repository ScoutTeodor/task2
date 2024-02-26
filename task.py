import cv2 as cv
import numpy as np
import math

def apply_canny(image):
    # Применение оператора Кэнни для выделения границ
    edges_image = cv.Canny(image, 50, 150)
    return edges_image

def apply_hough_transform(edges_image):
    # Применение преобразования Хафа для обнаружения прямых линий
    hough_lines_image = cv.cvtColor(edges_image, cv.COLOR_GRAY2BGR)
    lines_angles = []

    lines_hough = cv.HoughLinesP(edges_image, 1, np.pi / 180, 100, None, 20, 10)
    if lines_hough is not None:
        for line in lines_hough[:, 0]:
            if line[2] - line[0] == 0:
                continue
            angle = math.degrees(math.atan((line[3] - line[1]) / (line[2] - line[0])))
            if -40 < angle < 40:
                cv.line(hough_lines_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1, cv.LINE_AA)
                lines_angles.append(angle)

    return lines_angles

def rotate_image(input_filename, output_filename):
    # Загрузка изображения в оттенках серого
    original_image = cv.imread(input_filename, cv.IMREAD_GRAYSCALE)

    # Применение оператора Кэнни
    edges_image = apply_canny(original_image)

    # Применение преобразования Хафа и выделение линий
    lines_angles = apply_hough_transform(edges_image)

    # Вычисление среднего угла поворота на основе сохраненных углов
    rotation_angle = np.mean(lines_angles)
    print(f"Угол поворота: {rotation_angle}")

    # Применение аффинного преобразования для поворота изображения на найденный угол
    (h, w) = original_image.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv.warpAffine(original_image, rotation_matrix, (w, h), borderMode=cv.BORDER_REPLICATE)

    # Сохранение повернутого изображения в новом файле
    cv.imwrite(output_filename, rotated_image)

if __name__ == "__main__":
    # Ддля использования измените значения input_filename и output_filename
    input_filename = "2_2.jpg"
    output_filename = "result2.jpg"
    rotate_image(input_filename, output_filename)
