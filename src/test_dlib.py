import os
import cv2
import dlib
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                           recall_score, f1_score, precision_score,
                           confusion_matrix, roc_curve, precision_recall_curve)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# 初始化dlib模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models\pretrained\dlib\shape_predictor_68_face_landmarks.dat")

def extract_landmarks(image_path):
    """提取68个关键点并计算几何特征"""
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None
    
    landmarks = predictor(gray, faces[0])
    points = np.array([[p.x, p.y] for p in landmarks.parts()])
    
    # 计算关键几何特征
    def distance(p1, p2):
        return np.linalg.norm(points[p1] - points[p2])
    
    features = [
        distance(36, 39),  # 左眼宽度
        distance(42, 45),  # 右眼宽度
        distance(48, 54),  # 嘴巴宽度
        distance(51, 57),  # 嘴巴高度
        distance(27, 33),  # 鼻子高度
        distance(21, 22),  # 眉毛间距
        # 新增面部比例特征
        distance(36, 45) / distance(48, 54),  # 眼嘴宽度比
        distance(39, 42) / distance(33, 51),  # 眼鼻比例
    ]
    return np.array(features)

def load_dataset(data_dir):
    """加载数据集并提取特征"""
    X, y = [], []
    classes = {'real': 0, 'fake': 1}
    
    for label in classes:
        label_dir = os.path.join(data_dir, label)
        if not os.path.exists(label_dir):
            continue
            
        for img_file in os.listdir(label_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(label_dir, img_file)
                features = extract_landmarks(img_path)
                if features is not None:
                    X.append(features)
                    y.append(classes[label])
    
    return np.array(X), np.array(y)

def evaluate_model(y_true, y_pred, y_score=None, model_name=""):
    """完整评估模型性能"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'Confusion_Matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_score is not None:
        metrics['AUC-ROC'] = roc_auc_score(y_true, y_score)
    
    # 打印报告
    print(f"\n{model_name} Evaluation Report:")
    print("="*50)
    for k, v in metrics.items():
        if k != 'Confusion_Matrix':
            print(f"{k}: {v:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['Confusion_Matrix'])
    
    # 可视化
    plot_metrics(y_true, y_pred, y_score, model_name)
    
    return metrics

def plot_metrics(y_true, y_pred, y_score, model_name):
    """绘制评估指标可视化"""
    plt.figure(figsize=(15, 5))
    
    # 1. 混淆矩阵
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Pred Real', 'Pred Fake'],
               yticklabels=['True Real', 'True Fake'])
    plt.title(f'{model_name} Confusion Matrix')
    
    # 2. ROC曲线
    if y_score is not None:
        plt.subplot(1, 3, 2)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, y_score):.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
    
    # 3. 指标对比
    plt.subplot(1, 3, 3)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    ]
    plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    plt.ylim(0, 1)
    plt.title('Performance Metrics')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_metrics.png')
    plt.close()

# 主流程
def main():
    # 加载数据
    print("Loading datasets...")
    X_train, y_train = load_dataset("data/train")
    X_val, y_val = load_dataset("data/val")
    X_test, y_test = load_dataset("data/test")
    
    # 检查数据平衡性
    print(f"\nTrain samples: {len(y_train)} (Real: {sum(y_train==0)}, Fake: {sum(y_train==1)})")
    print(f"Val samples: {len(y_val)} (Real: {sum(y_val==0)}, Fake: {sum(y_val==1)})")
    print(f"Test samples: {len(y_test)} (Real: {sum(y_test==0)}, Fake: {sum(y_test==1)})")
    
    # 标准化特征
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # 方法1: SVM分类器
    print("\nTraining SVM...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    svm.fit(X_train, y_train)
    
    # 评估SVM
    val_pred = svm.predict(X_val)
    val_score = svm.predict_proba(X_val)[:, 1]
    test_pred = svm.predict(X_test)
    test_score = svm.predict_proba(X_test)[:, 1]
    
    evaluate_model(y_val, val_pred, val_score, "SVM Validation")
    evaluate_model(y_test, test_pred, test_score, "SVM Test")

    # 方法2: 神经网络分类器
    print("\nTraining Neural Network...")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 评估神经网络
    val_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
    val_score = model.predict(X_val).flatten()
    test_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    test_score = model.predict(X_test).flatten()
    
    evaluate_model(y_val, val_pred, val_score, "NN Validation")
    evaluate_model(y_test, test_pred, test_score, "NN Test")
    
    # 绘制训练历史
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    main()