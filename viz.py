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
def viz(model_fp_s, model_fp_d, img_fp, output_dir):

    out_s = os.path.join(output_dir, "source")
    out_d = os.path.join(output_dir, "domain")
    os.makedirs(out_s, exist_ok=True)
    os.makedirs(out_d, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_s = get_model()
    model_s.eval()
    model_s.to(device)
    model_s.load_state_dict(torch.load(model_fp_s)["model"])
    model_d = get_model()
    model_d.eval()
    model_d.to(device)
    model_d.load_state_dict(torch.load(model_fp_d)["model"])

    transform = get_transform()
    img = Image.open(img_fp)
    img_tensor = transform(img).unsqueeze(0)

    samples = img_tensor.to(device)
    top_k = 10

    output_s = model_s(samples)
    output_d = model_d(samples)
    indices_s = output_s['pred_logits'][0].softmax(-1)[..., 1].sort(descending=True)[1][:top_k]
    # indices_d = output_d['pred_logits'][0].softmax(-1)[..., 1].sort(descending=True)[1][:top_k]
    predictied_boxes_s = torch.stack([output_s['pred_boxes'][0][i] for i in indices_s]).unsqueeze(0)
    # predictied_boxes_d = torch.stack([output_d['pred_boxes'][0][i] for i in indices_d]).unsqueeze(0)
    logits_s = torch.stack([output_s['pred_logits'][0][i] for i in indices_s]).unsqueeze(0)
    # logits_d = torch.stack([output_d['pred_logits'][0][i] for i in indices_d]).unsqueeze(0)

    fp_s = os.path.join(out_s, f'img_{time.time()}.jpg')
    fig, ax = plt.subplots(1, figsize=(10,3), dpi=200)
    plot_prediction(samples[0:1], predictied_boxes_s, logits_s, ax, plot_prob=False)
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.savefig(fp_s, bbox_inches='tight')
    plt.close()

    fp_d = os.path.join(out_d, f'img_{time.time()}.jpg')
    out_img_d = compute_boxes(img, output_d)

    cv2.imwrite(fp_d, out_img_d)
    
    return fp_s, fp_d

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(out_bbox)
    return b

def plot_prediction(image, boxes, logits, ax=None, plot_prob=True):
    bboxes_scaled0 = rescale_bboxes(boxes[0], list(image.shape[2:])[::-1])
    probas = logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.01
    if ax is None:
        ax = plt.gca()
    plot_results(image[0].permute(1, 2, 0).detach().cpu().numpy(), probas[keep], bboxes_scaled0[keep], ax, plot_prob=plot_prob)

def plot_results(pil_img, prob, boxes, ax, plot_prob=True, norm=True):
    from matplotlib import pyplot as plt
    image = plot_image(ax, pil_img, norm)
    if prob is not None and boxes is not None:
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color='r', linewidth=1))
            if plot_prob:
                text = f'{p:0.2f}'
                ax.text(xmin, ymin, text, fontsize=15,
                        bbox=dict(facecolor='yellow', alpha=0.5))
    ax.grid('off')

def plot_image(ax, img, norm):
    if norm:
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = (img * 255)
    img = img.astype('uint8')
    ax.imshow(img)

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
        print (m_class, sorted_dict[m_class])
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

    print('{:d}'.format(num_boxes))
    return img_det_bgr


if __name__ == "__main__":
    viz("/data/weights/deformable_detr/checkpoint_fcs_source.pth", "/data/weights/deformable_detr/checkpoint_fcs.pth", "/workspace/static/fcs_berlin_samples/berlin_000000_000019_leftImg8bit_foggy_beta_0.01.png", "/workspace/static")