import cv2
import numpy as np
import os
import random
from ultralytics import YOLO

class VideoProcessor:
    def __init__(self, model_path, source_folder, background_folder, overlay_folder, output_folder):
        self.model = YOLO(model_path)
        self.source_folder = source_folder
        self.background_folder = background_folder
        self.overlay_folder = overlay_folder
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def select_random_overlay_image(self):
        overlay_files = [f for f in os.listdir(self.overlay_folder) if f.endswith('.png')]
        if not overlay_files:
            print("No overlay images found in the specified directory.")
            return None
        selected_image = os.path.join(self.overlay_folder, random.choice(overlay_files))
        return cv2.imread(selected_image, cv2.IMREAD_UNCHANGED)

    def process_frame(self, frame):
        # Run inference on an image
        results = self.model(frame, stream=True)  # results list

        segmentation_polygons = []
        boxes = []

        # Check if any results are found and process them
        if results:
            for r in results:
                if r.boxes and len(r.boxes.xywh) > 0:
                    # Iterate over bounding boxes and class IDs together using enumerate for indexing
                    for index, (bbox, cls) in enumerate(zip(r.boxes.xywh, r.boxes.cls)):
                        # Check if the class ID is 0 and process only these cases
                        if cls.item() == 0:  # Convert tensor to Python scalar
                            # Convert bbox to CPU and numpy array
                            box_array = bbox.cpu().numpy()
                            boxes.append(box_array)
                            # Process corresponding masks using the same index
                            if r.masks and len(r.masks.xy) > index:
                                mask = r.masks.xy[index]
                                if mask is not None and len(mask) > 0:
                                    # Ensure that mask is appropriately reshaped for fillPoly
                                    segmentation_polygon = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                                    segmentation_polygons.append(segmentation_polygon)

        # Return a mask for each segmentation polygon found, or an empty mask if none
        if segmentation_polygons:
            return self.create_mask_from_polygon(frame.shape, segmentation_polygons[0]), boxes
        else:
            empty_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            return empty_mask, boxes

    def create_mask_from_polygon(self, image_shape, polygon):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        return mask


    def segment_and_combine(self, target_frame, background_frame, mask):
        background_frame_resized = cv2.resize(background_frame, (target_frame.shape[1], target_frame.shape[0]))
        inverse_mask = cv2.bitwise_not(mask)
        foreground = cv2.bitwise_and(target_frame, target_frame, mask=mask)
        background = cv2.bitwise_and(background_frame_resized, background_frame_resized, mask=inverse_mask)
        combined_frame = cv2.add(foreground, background)
        return combined_frame
    
    def blend_overlay_with_boxes(self, frame, overlay_image, boxes):
        if not boxes:
            return frame  # 바운딩 박스가 없으면 원본 프레임 반환
        
        for box in boxes:
            x, y, w, h = map(int, box)
            frame_height, frame_width = frame.shape[:2]

            # 배경 영역 추출 전에 범위 검증
            y_end, x_end = min(y + h, frame_height), min(x + w, frame_width)
            background_region = frame[y:y_end, x:x_end]
            
            # 오버레이 이미지 및 알파 채널의 크기를 background_region에 맞춰 조정
            overlay_image_resized = cv2.resize(overlay_image, (background_region.shape[1], background_region.shape[0]), interpolation=cv2.INTER_AREA)
            
            if overlay_image_resized.shape[2] == 4:  # 알파 채널이 있는 경우
                overlay_color = overlay_image_resized[..., :3]  # RGB 채널
                overlay_alpha = overlay_image_resized[..., 3] / 255.0  # 알파 채널 정규화
                overlay_alpha = overlay_alpha[..., np.newaxis]  # 차원 확장
                
                # 알파 블렌딩 적용
                blended_region = overlay_alpha * overlay_color + (1 - overlay_alpha) * background_region
                frame[y:y_end, x:x_end] = blended_region.astype(np.uint8)

        return frame
    
    def process_video(self, video_path, background_paths, overlay_images, output_folder):
            frame_width, frame_height, fps = None, None, None
            for background_video_path in background_paths:
                target_cap = cv2.VideoCapture(video_path)  # Open the target video capture inside the loop
                if frame_width is None or frame_height is None or fps is None:
                    fps = target_cap.get(cv2.CAP_PROP_FPS)
                    frame_width = int(target_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(target_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                background_cap = cv2.VideoCapture(background_video_path)
                video_base = os.path.basename(video_path).split('.')[0]
                background_base = os.path.basename(background_video_path).split('.')[0]
                output_video_path = os.path.join(output_folder, f"{video_base}_with_{background_base}.mp4")
                out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

                while True:
                    ret_target, target_frame = target_cap.read()
                    ret_background, background_frame = background_cap.read()
                    if not ret_target or not ret_background:
                        break
                    mask, boxes = self.process_frame(target_frame)
                    combined_frame = self.segment_and_combine(target_frame, background_frame, mask)
                    overlay_image = random.choice(overlay_images) if overlay_images else None
                    if boxes and overlay_image is not None:
                        combined_frame = self.blend_overlay_with_boxes(combined_frame, overlay_image, boxes)
                    out.write(combined_frame)

                target_cap.release()  # Release the target capture after finishing with each background
                background_cap.release()
                out.release()

    def process_videos(self):
        background_paths = self.get_files(self.background_folder, '.mp4')
        overlay_images = [self.select_random_overlay_image() for _ in range(len(background_paths))]  # Ensure enough overlay images are selected
        for entry in os.listdir(self.source_folder):
            path = os.path.join(self.source_folder, entry)
            if os.path.isfile(path) and path.endswith('.mp4'):
                self.process_video(path, background_paths, overlay_images, self.output_folder)
    
    def get_files(self, directory, extension):
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]



# Example usage
model_path = '/home/park/Desktop/code/yolov8n-seg.pt'
background_folder = '/home/park/Desktop/precessing_data/background'
overlay_folder = '/home/park/Desktop/precessing_data/overlay_images'
output_folder = '/media/park/Elements/synthetic_data/bad_back_round'
source_folder = '/home/park/Desktop/data/Video_Dataset/bad_back_round'

processor = VideoProcessor(model_path, source_folder, background_folder, overlay_folder, output_folder)
processor.process_videos()
