import cv2
import dlib
import numpy as np
import os

# 初始化 dlib 检测器与预测器
predictor_path = "F:/DeepfakeDetection/models/pretrained/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def extract_geometric_features(image_path):
    """
    提取68个面部关键点后计算6个几何距离特征
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    landmarks = predictor(gray, faces[0])
    points = np.array([[p.x, p.y] for p in landmarks.parts()])

    def distance(p1, p2):
        return np.linalg.norm(points[p1] - points[p2])

    features = [
        distance(36, 39),  # 左眼宽度
        distance(42, 45),  # 右眼宽度
        distance(48, 54),  # 嘴巴宽度
        distance(51, 57),  # 嘴巴高度
        distance(27, 33),  # 鼻子高度
        distance(21, 22),  # 眉毛间距
    ]
    return np.array(features, dtype=np.float32)

def visualize_landmarks(image_path):
    """
    可视化图像中的68个面部关键点
    """
    img = cv2.imread(image_path)
    if img is None:
        print("无法加载图像:", image_path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(68):
            x, y = landmarks.part(n).x, landmarks.part(n).y
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Landmarks", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_landmark_dataset(data_dir):
    """
    加载图像数据集并提取每张图像的 dlib 几何特征（用于分类任务）
    """
    X, y = [], []
    classes = {'real': 0, 'fake': 1}

    for label in classes:
        label_dir = os.path.join(data_dir, label)
        for fname in os.listdir(label_dir):
            if fname.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(label_dir, fname)
                features = extract_geometric_features(img_path)
                if features is not None:
                    X.append(features)
                    y.append(classes[label])
    
    return np.array(X), np.array(y)
