import platform

import cv2


class VideoCaptureFactory:
    @staticmethod
    def open_capture(source):
        if isinstance(source, str) and not source.isdigit():
            cap = cv2.VideoCapture(source)
            return cap if cap.isOpened() else None

        cam_index = int(source)
        is_macos = platform.system() == "Darwin"
        backend_options = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if is_macos else [cv2.CAP_ANY]

        for backend in backend_options:
            cap = cv2.VideoCapture(cam_index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
        return None
