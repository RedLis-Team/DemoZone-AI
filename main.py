import cv2
import gradio as gr
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

model = YOLO("yolo11n-seg.pt")

def video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    video_name = f'./video_gen/genvideo.mp4'
    out = cv2.VideoWriter(video_name, cv2.VideoWriter.fourcc(*"H264"), fps, (w, h))

    while True:
        ret, im0 = cap.read()
        if not ret:
            break

        annotator = Annotator(im0, line_width=2)
        results = model.track(im0, persist=True)

        if results[0].boxes.id is not None and results[0].masks is not None:
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for mask, track_id in zip(masks, track_ids):
                color = colors(int(track_id), True)
                txt_color = annotator.get_txt_color(color)
                annotator.seg_bbox(mask=mask, mask_color=color, label=str(track_id), txt_color=txt_color)

        out.write(im0)

    out.release()
    cap.release()

    return video_name


video = gr.Interface(
    fn=video_stream,
    inputs=gr.Video(),
    outputs=[gr.Video()],
)

if __name__ == "__main__":
    video.launch()