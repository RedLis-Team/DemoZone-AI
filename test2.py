import gradio as gr
import cv2


# Функция для захвата и обработки кадра в реальном времени
def process_frame():
    # Открываем поток с веб-камеры (0 — это индекс устройства веб-камеры)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Пример обработки кадра: добавление текста
        frame = cv2.putText(frame, "Real-time Stream", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Преобразование кадра в RGB для отображения в Gradio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Прерывание, чтобы остановить поток (по тайм-ауту, нажатием ESC, и т.д.)
        yield frame_rgb  # Возвращаем текущий кадр

    cap.release()


# Настройка интерфейса Gradio для стриминга
interface = gr.Interface(
    fn=process_frame,  # Функция стриминга
    inputs=None,  # Нет входных данных
    outputs=gr.Video()  # Вывод: видео
)

# Запуск стриминга
interface.launch()