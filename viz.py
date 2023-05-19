import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

from get_model import get_model
from transform import get_transform
from collections import Counter, OrderedDict


@torch.no_grad()
def viz(model_paths, output_dir_names, img_fp, static_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir_paths = []
    for dname in output_dir_names:
        out_path = os.path.join(static_root, dname)
        output_dir_paths.append(out_path)
        os.makedirs(out_path, exist_ok=True)

    models = []
    for model_path in model_paths:
        model = get_model()
        model.eval()
        model.to(device)
        model.load_state_dict(torch.load(model_path)["model"])
        models.append(model)

    transform = get_transform()
    img = Image.open(img_fp)
    img_tensor = transform(img).unsqueeze(0)

    samples = img_tensor.to(device)

    output_filepaths = []
    for out_dir, model in zip(output_dir_paths, models):
        output = model(samples)
        out_path = os.path.join(out_dir, f'res_{img_fp.split("/")[-1].replace(".png", "")}.jpg')
        out_img = compute_boxes(img, output)
        cv2.imwrite(out_path, out_img)
        output_filepaths.append(out_path)

    return output_filepaths


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', \
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
              'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
              'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
              'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def compute_boxes(input_image, output_dict):
    scores = torch.sigmoid(output_dict['pred_logits'][..., 1])
    _, predicted =  torch.max(output_dict['pred_logits'],2)
    pred_boxes = output_dict['pred_boxes']
    scores = scores.cpu()
    pred_boxes = pred_boxes.cpu()
    img_w, img_h = input_image.size
    pred_boxes_ = box_cxcywh_to_xyxy(pred_boxes) * torch.Tensor([img_w, img_h, img_w, img_h])
    I = scores.argsort(descending = True) # sort by model confidence
    predicted_ = predicted[0, I[0,:]]
    pred_boxes_ = pred_boxes_[0, I[0,:]] # pick top 3 proposals
    scores_ = scores[0, I[0,:]]
    predicted_class = np.array(predicted_.cpu())
    cls_cnt = Counter(predicted_class)
    sorted_dict = OrderedDict(sorted(cls_cnt.items(), key = lambda kv : kv[1], reverse=True))
    used_classes = []
    mean_p_class = []
    len_th = 70
    for m_class in sorted_dict: 
        print(f"Class no.: {m_class}, Count: {sorted_dict[m_class]}")
        if sorted_dict[m_class] > len_th:
            used_classes.append(m_class)
            idx = predicted_ == m_class
            p_values = scores_[idx]
            measure_th = int(sorted_dict[m_class] * 0.15)
            th_p = torch.mean(p_values[0:measure_th])
            mean_p_class.append(th_p.item())
    rgb_img = np.float32(np.array(input_image))
    scores_np = np.array(scores_.data)
    used_classes_np =list(used_classes)
    mean_p_class_np = list(mean_p_class)
    th_p = 0.35
    draw_brg_img = draw_boxes(pred_boxes_, predicted_class, scores_np, rgb_img, th_p , used_classes_np, mean_p_class_np)
    return draw_brg_img

def draw_boxes(boxes, labels, scores, image, th, used_classes, mean_p_class ):
    img_det = np.array(image)
    img_det_bgr = cv2.cvtColor(img_det, cv2.COLOR_RGB2BGR)
    num_boxes = 0
    for i, box in enumerate(boxes):
        #color = COLORS[labels[i]]        
        class_m = labels[i]
        if class_m in used_classes:
            th_idx = used_classes.index(class_m)
            if scores[i] > mean_p_class[th_idx]:
                num_boxes +=1
                color = COLORS[class_m]
                cv2.rectangle(
                    img_det_bgr,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color, 2
                )
                cv2.putText(img_det_bgr,'{:0.2f}'.format(scores[i]), (int(box[0]), int(box[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                            lineType=cv2.LINE_AA)


    if num_boxes < 2 :
        for i, box in enumerate(boxes):
            if scores[i] >th:
                num_boxes +=1
                color = COLORS[class_m]
                cv2.rectangle(
                    img_det_bgr,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color, 2
                )
                cv2.putText(img_det_bgr,'{:0.2f}'.format(scores[i]), (int(box[0]), int(box[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                            lineType=cv2.LINE_AA)


    for i, class_m in enumerate(used_classes):
        color = COLORS[class_m]
        cv2.putText(img_det_bgr, coco_names[class_m], (int(10), int(10+i*15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                            lineType=cv2.LINE_AA)

    print('No. bboxes: {:d}'.format(num_boxes))
    return img_det_bgr


if __name__ == "__main__":
    viz("/data/weights/deformable_detr/checkpoint_fcs_source.pth", "/data/weights/deformable_detr/checkpoint_fcs.pth", "/workspace/static/fcs_berlin_samples/berlin_000000_000019_leftImg8bit_foggy_beta_0.01.png", "/workspace/static")