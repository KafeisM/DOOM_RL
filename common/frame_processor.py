# common/frame_processor.py
import cv2
import numpy as np

def default_frame_processor(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[0] in [1, 2, 3]:
        frame = np.transpose(frame, (1, 2, 0))  # de CHW a HWC si hace falta

    resized = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)

    if resized.shape[2] == 3:  # Asegurar RGB si VizDoom devuelve BGR
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    return resized.astype(np.uint8)
