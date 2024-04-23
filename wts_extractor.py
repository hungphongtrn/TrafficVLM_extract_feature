"""Extracting features from the video frames using CLIP model."""
import os

from PIL import Image
import cv2 as cv
import yaml
import numpy as np
import torch
import clip

from bbox import get_square_box


'''                     SECTION: SETUPS
Instructions: Fill in the paths and configurations in config.yaml file.'''

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yaml'), 'r', encoding='UTF-8') as file:
    configs = yaml.safe_load(file)
video_path = configs['paths']['video_path']  # Path to the input folder
anno_path = configs['paths']['anno_path']   # Path to the annotation folder
output_path = configs['paths']['output_path']  # Path to the output folder
feature_type = configs['feature_type']   # Choices: 'local', 'semi-global', 'global'

CLIP_MODEL_DIR = configs['paths']['CLIP_PATH'] # Path to the CLIP model directory

split = 'train' if 'train' in video_path else ('val' if 'val' in video_path else 'test')
is_external = 'BDD' in video_path

model, preprocess = clip.load("ViT-L/14", download_root=CLIP_MODEL_DIR)
model.eval()
model.cuda()

if not os.path.exists(output_path):
    os.mkdir(output_path)




'''                     SECTION: EXTRACTOR                          '''

file_list = os.listdir(video_path)

# Extract: external
if is_external:
    for filename in file_list:

        # initial configurations
        filename = filename[:filename.rfind('.')] #e.g. vid.mp4->vid
        file_vid_path = os.path.join(video_path,filename+'.mp4')
        file_bbox_path = os.path.join(anno_path,filename+'_bbox.json')
        bbox_to_cut = get_square_box(file_vid_path, file_bbox_path, feature_type)
        if feature_type == 'local':
            os.mkdir(os.path.join(output_path,filename))
        else:
            imfeat = []
        frame_count = 0

        # extract features
        cap = cv.VideoCapture(file_vid_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if feature_type == 'local':
                if frame_count in bbox_to_cut:
                    x1, y1, x2, y2 = bbox_to_cut[frame_count]
                    frame = frame[y1:y2, x1:x2]
                    frame = cv.cvtColor(cv.resize(frame, (224,224)),cv.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                    image = preprocess(image).unsqueeze(0).cuda()
                    with torch.no_grad():
                        image_features = model.encode_image(image).cpu().numpy()
                    np.save(os.path.join(output_path,filename,f'frame{frame_count}.npy'),image_features)
                else:
                    pass
            else:
                x1, y1, x2, y2 = bbox_to_cut[-1]
                frame = frame[y1:y2, x1:x2]
                frame = cv.cvtColor(cv.resize(frame, (224,224)),cv.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                image = preprocess(image).unsqueeze(0).cuda()
                with torch.no_grad():
                    image_features = model.encode_image(image).cpu().numpy()
                imfeat.append(image_features)
            frame_count += 1

        cap.release()
        if feature_type != 'local':
            imfeat = np.concatenate(imfeat, axis=0)
            np.save(os.path.join(output_path,filename+'.npy'),imfeat)
    
# Extract: internal
if not is_external:
    for view_name in file_list:

        # create a new folder as each scene contains multiple views
        if not os.path.exists(os.path.join(output_path,view_name)):
            os.mkdir(os.path.join(output_path,view_name))
    
        l_overhead_files = os.listdir(os.path.join(video_path,view_name,'overhead_view'))
        l_vehicle_files  = os.listdir(os.path.join(video_path,view_name,'vehicle_view'))

        # Extract: OVERHEAD
        for overhead_file in l_overhead_files:            
            # initial configurations
            overhead_file = overhead_file[:overhead_file.rfind('.')]
            overhead_vid_path = os.path.join(video_path,view_name,'overhead_view',overhead_file+'.mp4')
            overhead_bbox_path = os.path.join(anno_path,view_name,'overhead_view',overhead_file+'_bbox.json')
            bbox_to_cut = get_square_box(overhead_vid_path, overhead_bbox_path, feature_type)

            if feature_type == 'local':
                os.mkdir(os.path.join(output_path,view_name,overhead_file))
            else:
                imfeat = []
            frame_count = 0
            
            # extract features
            cap = cv.VideoCapture(overhead_vid_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if feature_type == 'local':
                    if frame_count in bbox_to_cut:
                        x1, y1, x2, y2 = bbox_to_cut[frame_count]
                        frame = frame[y1:y2, x1:x2]
                        frame = cv.cvtColor(cv.resize(frame, (224,224)),cv.COLOR_BGR2RGB)
                        image = Image.fromarray(frame)
                        image = preprocess(image).unsqueeze(0).cuda()
                        with torch.no_grad():
                            image_features = model.encode_image(image).cpu().numpy()
                        np.save(os.path.join(output_path,view_name,'overhead_view',overhead_file,f'frame{frame_count}.npy'),image_features)
                    else:
                        pass
                else:
                    x1, y1, x2, y2 = bbox_to_cut[-1]
                    frame = frame[y1:y2, x1:x2]
                    frame = cv.cvtColor(cv.resize(frame, (224,224)),cv.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                    image = preprocess(image).unsqueeze(0).cuda()
                    with torch.no_grad():
                        image_features = model.encode_image(image).cpu().numpy()
                    imfeat.append(image_features)
                frame_count += 1
            cap.release()
            if feature_type != 'local':
                imfeat = np.concatenate(imfeat, axis=0)
                np.save(os.path.join(output_path,view_name,'overhead_view',overhead_file+'.npy'),imfeat)

        # Extract: VEHICLE
        for vehicle_file in l_vehicle_files:
            # initial configurations
            vehicle_file = vehicle_file[:vehicle_file.rfind('.')]
            vehicle_vid_path = os.path.join(video_path,view_name,'vehicle_view',vehicle_file+'.mp4')
            vehicle_bbox_path = os.path.join(anno_path,view_name,'vehicle_view',vehicle_file+'_bbox.json')
            bbox_to_cut = get_square_box(vehicle_vid_path, vehicle_bbox_path, feature_type)

            if feature_type == 'local':
                os.mkdir(os.path.join(output_path,view_name,vehicle_file))
            else:
                imfeat = []
            frame_count = 0

            # extract features
            cap = cv.VideoCapture(vehicle_vid_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if feature_type == 'local':
                    if frame_count in bbox_to_cut:
                        x1, y1, x2, y2 = bbox_to_cut[frame_count]
                        frame = frame[y1:y2, x1:x2]
                        frame = cv.cvtColor(cv.resize(frame, (224,224)),cv.COLOR_BGR2RGB)
                        image = Image.fromarray(frame)
                        image = preprocess(image).unsqueeze(0).cuda()
                        with torch.no_grad():
                            image_features = model.encode_image(image).cpu().numpy()
                        np.save(os.path.join(output_path,view_name,'vehicle_view',vehicle_file,f'frame{frame_count}.npy'),image_features)
                    else:
                        pass
                else:
                    x1, y1, x2, y2 = bbox_to_cut[-1]
                    frame = frame[y1:y2, x1:x2]
                    frame = cv.cvtColor(cv.resize(frame, (224,224)),cv.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                    image = preprocess(image).unsqueeze(0).cuda()
                    with torch.no_grad():
                        image_features = model.encode_image(image).cpu().numpy()
                    imfeat.append(image_features)
                frame_count += 1
            cap.release()
            if feature_type != 'local':
                imfeat = np.concatenate(imfeat, axis=0)
                np.save(os.path.join(output_path,view_name,'vehicle_view',vehicle_file+'.npy'),imfeat)