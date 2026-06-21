
import numpy as np
import pyrealsense2 as rs
import cv2
import pygame
import gymnasium as gym
import ale_py
from collections import deque

class Eyes:
    def __init__(self, n=4):
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            config.enable_device('814412071258')
            self.pipeline.start(config)
            self.frames = deque(maxlen=n)
            self.n = n
            self.screen_corner = np.array([
                    [[373, 260]],
                    [[895, 261]],
                    [[895, 551]],
                    [[373, 551]]
                ], dtype=np.float32)
            

    def watch(self):

        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            frame = np.ascontiguousarray(np.asanyarray(color_frame.get_data()))
            screen_crop = self.crop_to_screen(frame, self.screen_corner)
            cv2.imshow("streaming", screen_crop)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.pipeline.stop()
        cv2.destroyAllWindows()


    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84,84))
        img = img/255.0

        return img.astype(np.float32)

    def reset(self):
        proc = self.preprocess(self.get_frame())
        for _ in range(self.n):
            self.frames.append(proc)
        return self._get_state()

    def step(self):
        self.frames.append(self.preprocess(self.get_frame()))
        return self._get_state()

    def _get_state(self):
        return np.stack(self.frames, axis=0)
    
    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.ascontiguousarray(np.asanyarray(color_frame.get_data()))
        return self.crop_to_screen(frame, self.screen_corner)

    def find_screen(self, frame, min_width=495, max_width=575, min_height=271, max_height=351):
        #TODO: doesn't work properly need fixing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) != 4:
                continue
            
            x, y, w, h = cv2.boundingRect(approx)
            
            # filter by known approximate screen pixel dimensions
            if not (min_width < w < max_width):
                continue
            if not (min_height < h < max_height):
                continue
            
            # aspect ratio still worth checking — 16:9 ± tolerance
            aspect = w / h
            if not (1.2 < aspect < 2.2):
                continue
            
            candidates.append((cv2.contourArea(c), approx))
        
        if not candidates:
            return None
        
        # among valid candidates pick the one closest to expected size
        # rather than just the largest
        target_w = (min_width + max_width) / 2
        target_h = (min_height + max_height) / 2
        
        def size_score(self, candidate):
            _, approx = candidate
            x, y, w, h = cv2.boundingRect(approx)
            return abs(w - target_w) + abs(h - target_h)
        
        best = min(candidates, key=size_score)
        return best[1]
            
    def crop_to_screen(self, frame, corner):
        pts = corner.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        w, h = 640, 480
        dst = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        
        return cv2.warpPerspective(frame, M, (w, h))

class Frames():
    def __init__(self, n=4):
        self.frames = deque(maxlen=n)
        self.n = n

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84,84))
        img = img/255.0

        return img.astype(np.float32)
    
    def reset(self, img):
        proc = self.preprocess(img)
        for _ in range(self.n):
            self.frames.append(proc)
        return self._get_state()
    
    def step(self, img):
        self.frames.append(self.preprocess(img))
        return self._get_state()
    
    def _get_state(self):
        return np.stack(self.frames, axis=0)