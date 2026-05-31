import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

st.set_page_config(page_title="人脸检测系统", layout="wide", page_icon="👤")

st.title("👤 人脸AI智能检测系统")
st.markdown("> 上传一张图片，AI将自动检测并标记所有人脸位置")
st.markdown("---")

def area_of(left_top, right_bottom):
    """计算边界框面积"""
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """计算 IoU（交并比）"""
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold=0.3, top_k=-1, candidate_size=200):
    """
    硬非极大值抑制，完全匹配原题 box_utils.hard_nms 逻辑
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)[::-1]
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current)
        if len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(rest_boxes, np.expand_dims(current_box, axis=0))
        indexes = indexes[iou <= iou_threshold]
    
    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    """预测函数 - 完全匹配原题逻辑"""
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        # 使用 hard_nms 进行非极大值抑制
        box_probs = hard_nms(box_probs, iou_threshold=iou_threshold, top_k=top_k)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    
    picked_box_probs = np.concatenate(picked_box_probs)
    # 还原坐标到原图尺寸
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

# 加载模型
@st.cache_resource
def load_model():
    return ort.InferenceSession('version-RFB-320.onnx')

# 加载标签
@st.cache_data
def load_labels():
    return [name.strip() for name in open('voc-model-labels.txt').readlines()]

# 侧边栏
st.sidebar.header("📋 使用说明")
st.sidebar.markdown("""
1. 上传一张包含人脸的图片
2. AI自动检测所有人脸
3. 在原图上画出方框并标记

**调整检测灵敏度：**
""")

threshold = st.sidebar.slider("置信度阈值", 0.1, 0.9, 0.5, 0.05)
st.sidebar.caption(f"当前阈值: {threshold}（原题默认 0.7）")

st.sidebar.markdown("---")
st.sidebar.caption("模型：RFB-320 · 轻量级人脸检测")

uploaded_file = st.file_uploader("📤 上传图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 原图")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🔍 检测结果")
        
        with st.spinner("🤖 AI正在检测人脸..."):
            # 完全按照原题的处理流程
            orig_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            h, w = orig_image.shape[:2]
            
            # 预处理
            image_resized = cv2.resize(orig_image, (320, 240))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_mean = np.array([127, 127, 127])
            image_normalized = (image_rgb - image_mean) / 128
            image_transposed = np.transpose(image_normalized, [2, 0, 1])
            image_input = np.expand_dims(image_transposed, axis=0).astype(np.float32)
            
            # 推理
            session = load_model()
            input_name = session.get_inputs()[0].name
            confidences, boxes = session.run(None, {input_name: image_input})
            
            # 后处理
            boxes_result, labels, probs = predict(w, h, confidences, boxes, threshold)
            
            # 绘制结果
            result_img = orig_image.copy()
            class_names = load_labels()
            
            if len(boxes_result) > 0:
                st.success(f"✅ 检测到 {len(boxes_result)} 张人脸")
                
                for i in range(boxes_result.shape[0]):
                    box = boxes_result[i, :]
                    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
                    cv2.rectangle(result_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                    cv2.putText(result_img, label, (box[0], box[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # 显示详情
                st.markdown("---")
                st.subheader("📊 检测详情")
                for i, (box, conf) in enumerate(zip(boxes_result, probs)):
                    conf_value = float(conf)
                    st.write(f"**人脸 {i+1}** : 置信度 {conf_value:.2%}")
                    st.progress(conf_value)
                
                # 显示调试信息
                with st.expander("🔧 调试信息"):
                    st.write(f"检测到 {len(boxes_result)} 个人脸框")
                    st.write(f"使用的阈值: {threshold}")
                    st.write(f"模型输出形状: confidences {confidences.shape}, boxes {boxes.shape}")
            else:
                st.warning("⚠️ 未检测到人脸")
                st.info(f"建议：当前阈值 {threshold}，尝试降低到 0.3 以下或使用更清晰的人脸图片")
    
    st.markdown("---")
    st.info("💡 提示：降低置信度阈值（左侧滑块）可以检测到更多人脸")

else:
    st.info("👆 请上传一张包含人脸的图片开始检测")
    
    # 示例展示
    st.markdown("---")
    st.subheader("🎯 效果示意")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    with col_ex1:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: #f0f0f0; border-radius: 15px;">
            <span style="font-size: 48px;">👤</span>
            <p>单人检测</p>
        </div>
        """, unsafe_allow_html=True)
    with col_ex2:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: #f0f0f0; border-radius: 15px;">
            <span style="font-size: 48px;">👥👥</span>
            <p>多人检测</p>
        </div>
        """, unsafe_allow_html=True)
    with col_ex3:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: #f0f0f0; border-radius: 15px;">
            <span style="font-size: 48px;">⚡</span>
            <p>实时检测</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("🤖 基于 RFB-320 ONNX 模型 · 轻量级人脸检测系统")