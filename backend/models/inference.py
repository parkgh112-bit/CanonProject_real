"""
모델 추론 통합 로직
YOLO, CNN 모델을 연결하여 Pass/Fail 판단
"""
import base64
import io
import time
import cv2 # Added cv2 import
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from models.cnn_model import CNNModel
from models.yolo_model import YOLOModel

# 전역 모델 인스턴스 (서버 시작 시 한 번만 로드)
yolo_model = None
cnn_model = None

# YOLO 클래스 이름 (yolo_model.py의 정의를 따름)
CLASS_NAMES = ['Btn_Home', 'Btn_Back', 'Btn_ID', 'Btn_Stat', 'Monitor', 'Text']


def convert_numpy_types(data):
    """
    분석 결과에 포함된 NumPy 타입을 Python 기본 타입으로 재귀적으로 변환
    """
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [convert_numpy_types(i) for i in data]
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data


def initialize_models(
    yolo_path: str = "models/YOLO.pt",
    cnn_path: str = "models/CNN_classifier.pt"
):
    """모델 초기화 (서버 시작 시 호출)"""
    global yolo_model, cnn_model
    
    if yolo_model is None:
        yolo_model = YOLOModel(model_path=yolo_path)
    
    if cnn_model is None:
        cnn_model = CNNModel(model_path=cnn_path)
    
    return yolo_model, cnn_model


def analyze_image(image: np.ndarray) -> Dict:
    """
    이미지 분석 및 시각화 메인 함수
    """
    # 모델 초기화 확인
    if yolo_model is None or cnn_model is None:
        initialize_models()
    
    start_time = time.time()
    
    try:
        # --- 이미지 준비 ---
        # 분석에는 원본 numpy 배열(RGB) 사용
        # 그리기를 위해 BGR로 변환된 cv2 이미지 준비
        draw_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # CNN을 위해 PIL 이미지 준비
        pil_img_gray = Image.fromarray(image).convert("L")

        # 1. YOLO 객체 검출
        yolo_results = yolo_model.detect(image)
        detected_classes = [d["class"] for d in yolo_results.get("detections", [])]
        
        # 2. CNN ROI 검증
        cnn_results = []
        roi_pass_list = []
        button_conditions = ['Btn_Back', 'Btn_Home', 'Btn_ID', 'Btn_Stat']
        
        for detection in yolo_results.get("detections", []):
            cls_name = detection["class"]
            if cls_name not in button_conditions:
                continue
            
            bbox = detection["bbox"]  # [x1, y1, x2, y2]
            crop = pil_img_gray.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            
            prob, is_pass = cnn_model.predict_roi(crop, cls_name)
            roi_pass_list.append(is_pass)
            
            cnn_results.append({
                "class": cls_name,
                "bbox": bbox,
                "probability": prob,
                "status": "Pass" if is_pass else "Fail"
            })
        
        # 3. YOLO 및 CNN 판정
        yolo_ok = (
            ('Btn_Home' in detected_classes and 'Btn_Stat' in detected_classes) and
            (('Btn_Back' in detected_classes) ^ ('Btn_ID' in detected_classes)) and
            ('Monitor' in detected_classes)
        )
        cnn_ok = all(roi_pass_list) if roi_pass_list else False
        
        # 4. 최종 판정 및 Fail 사유 수집
        final_status = "PASS" if (yolo_ok and cnn_ok) else "FAIL"
        reasons = []
        if not yolo_ok:
            reasons.append("YOLO: Required objects missing/invalid")
        if not cnn_ok:
            reasons.append("CNN: Button quality check failed")
        reason_str = "; ".join(reasons) if reasons else None
        
        # --- 이미지 시각화 (OpenCV) ---
        # CNN 결과를 bbox를 키로 하는 맵으로 변환 (튜플로 변환하여 해시 가능하게 함)
        cnn_results_map = {tuple(res["bbox"]): res["status"] for res in cnn_results}

        # 5. 모든 탐지된 객체에 대한 정보 그리기
        all_detections = yolo_results.get("detections", [])

        for det in all_detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = det["class"]
            
            # CNN 결과가 있는 버튼의 경우, 라벨에 상태 추가
            bbox_tuple = tuple(det["bbox"])
            if bbox_tuple in cnn_results_map:
                status = cnn_results_map[bbox_tuple]
                label += f" {status}"
                color = (0, 255, 0) if status == "Pass" else (0, 0, 255) # Green for Pass, Red for Fail
            else:
                color = (0, 200, 255) # Default color (yellow-ish)

            cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(draw_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA) # Black outline
            cv2.putText(draw_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA) # White text

        # 6. 최종 결과 및 실패 원인 그리기
        status_color = (0, 255, 0) if final_status == "PASS" else (0, 0, 255) # Green or Red
        cv2.putText(draw_img, final_status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA)
        
        if final_status == "FAIL":
            y_offset = 80
            for r in reasons:
                cv2.putText(draw_img, r, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
                y_offset += 30

        # 7. 시각화된 이미지를 Base64로 인코딩
        _, buffer = cv2.imencode('.jpg', draw_img)
        annotated_image_str = base64.b64encode(buffer).decode('utf-8')

        # 8. 최종 결과 객체 생성
        confidence_scores = [d.get("confidence", 0) * 100 for d in all_detections] + \
                            [c.get("probability", 0) * 100 for c in cnn_results]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        end_time = time.time()
        execution_time = round(end_time - start_time, 4)
        
        final_result = {
            "status": final_status,
            "reason": reason_str,
            "confidence": round(avg_confidence, 2),
            "details": {
                "yolo_status": "Pass" if yolo_ok else "Fail",
                "cnn_status": "Pass" if cnn_ok else "Fail",
                "execution_time": execution_time,
                "annotated_image": annotated_image_str,
                "yolo_detections": yolo_results.get("detections", []),
                "cnn_results": cnn_results,
                "detected_classes": detected_classes
            }
        }
        return convert_numpy_types(final_result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_result = {
            "status": "FAIL",
            "reason": f"분석 중 오류 발생: {str(e)}",
            "confidence": 0,
            "details": {}
        }
        return convert_numpy_types(error_result)

def analyze_frame(image: np.ndarray) -> Dict:
    """
    실시간 프레임 분석 (analyze_image와 동일)
    """
    return analyze_image(image)