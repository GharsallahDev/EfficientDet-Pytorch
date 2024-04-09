import argparse
import os
import yaml
import time
import torch
from torch.backends import cudnn
import matplotlib.pyplot as plt
from backbone import EfficientDetBackbone
import cv2
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
from matplotlib import colors

def get_args():
    parser = argparse.ArgumentParser('EfficientDet Pytorch: SOTA object detection network')
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--cuda', action='store_true', help='use cuda')
    parser.add_argument('-f', '--float16', action='store_true', help='use float')
    parser.add_argument('-w', '--weight', type=str, default=None, help='trained model weight')
    parser.add_argument('-i', '--image_path', type=str, default=None, help='image to infer')
    parser.add_argument('-s', '--force_input_size', type=str, default=None, help='force image size')
    parser.add_argument('-m', '--imshow', type=bool, default=False, help='imshow')
    parser.add_argument('-r', '--imwrite', type=bool, default=False, help='imwrite')

    args = parser.parse_args()
    return args

def validate_args(opt):
    if not opt.project:
        print(f"Warning: Project file is not provided: {opt.project}")
        return False
    elif not os.path.exists(os.path.join('./projects', opt.project + '.yml')):
        print(f"Warning: Project path is invalid: {opt.project}")
        return False

    if not opt.weight:
        print("Warning: Weight path is not provided")
        return False
    elif not os.path.exists(opt.weight):
        print(f"Warning: Weight path is invalid: {opt.weight}")
        return False
    
    if not opt.image_path:
        print("Warning: Image path is not provided")
        return False
    elif not os.path.exists(opt.image_path):
        print(f"Warning: Image path is invalid: {opt.image_path}")
        return False
    else:
        valid_image_extensions = ['.jpg', '.jpeg', '.png']
        if not any(opt.image_path.lower().endswith(ext) for ext in valid_image_extensions):
            print("Warning: Image format is invalid")
            return False

    print("All checks passed. Ready for inference.")
    return True

def infer(opt, config):

    image_path = opt.image_path
    use_cuda = opt.cuda and torch.cuda.is_available()
    use_float16 = opt.float16 and torch.cuda.is_available() and 'cuda' in use_cuda and torch.cuda.get_device_capability()[0] >= 7

    compound_coef = config.get('compound_coef', 0)
    anchor_ratios = eval(config['anchors_ratios'])
    anchor_scales = eval(config['anchors_scales'])
    threshold = config.get('threshold', 0.2)
    iou_threshold = config.get('iou_threshold', 0.2)
    obj_list = config['obj_list']

    cudnn.fastest = True
    cudnn.benchmark = True

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if opt.force_input_size is None else opt.force_input_size

    ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_size)

    if use_cuda:
        image = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        image = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    image = image.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(opt.weight, map_location='cpu'))
    model.requires_grad_(False)
    model.eval() 

    if use_cuda:
        model = model.cuda()

    if use_float16:
        model = model.half()

    with torch.no_grad():
        _, regression, classification, anchors = model(image)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(image,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)
        
        out = invert_affine(framed_metas, out)

    with torch.no_grad():
        print('Model Inferring & Postprocessing')
        print('Inferring Image x 10')
        t1 = time.time()
        for _ in range(10):
            _, regression, classification, anchors = model(image)

            out = postprocess(image,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)
            out = invert_affine(framed_metas, out)
            display(out, ori_imgs, obj_list, compound_coef, image_path=image_path, imshow=opt.imshow, imwrite=opt.imwrite)

        t2 = time.time()
        tact_time = (t2 - t1) / 10
        print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

def display(preds, imgs, obj_list, compound_coef, image_path, imshow=False, imwrite=False):
    color_list = standard_to_bgr(STANDARD_COLORS)
    save_dir = 'test'
    image_name = os.path.basename(image_path)  # Extract img.png from /something/img.png
    inferred_image_name = f"inferred_{image_name}"  # Create inferred_img.png
    save_path = os.path.join(save_dir, inferred_image_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue  # Skip images without detections

        img_copy = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = preds[i]['scores'][j]
            score = 0.0 if score is None else float(score)
            label = f"{obj} {score:.2f}"  # Display score with label

            color = color_list[get_index_label(obj, obj_list)]
            plot_one_box(img_copy, [x1, y1, x2, y2], label=label, color=color)

        if imwrite:
            cv2.imwrite(save_path, img_copy)

        if imshow:
            print("INFERRED IMAGE")
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()


def load_config(project_file):
    with open(project_file, 'r') as file:
        config = yaml.safe_load(file)

    required_variables = ['obj_list', 'anchors_ratios', 'anchors_scales']
    for var in required_variables:
        if var not in config or config[var] is None:
            print(f"Warning: '{var}' is missing or empty in the YAML file")
            return None
    return config


if __name__ == '__main__':
    opt = get_args()
    if validate_args(opt):
        config = load_config(os.path.join('./projects', opt.project + '.yml'))
        if config is not None:
            infer(opt, config)