import os
import numpy as np
import json
import cv2

def main():

    path = '/media/datasets/pose_estimation/PoseTrack_Challenge_2018_v2.2/posetrack_data/'
    save_path = '/media/work2/doering/2020/output/DetectAndTrack/posetrack_2018_files/'

    splits = ['train', 'val']

    for split in splits:
        new_images = []
        annotations = []
        categories = []

        split_path = os.path.join(path, 'annotations', split)

        files = os.listdir(split_path)
        for idx, file in enumerate(files):
            with open(os.path.join(split_path, file), 'r') as f:
                data = json.load(f)

            if idx == 0:
                categories = data['categories']

            images = {img['id']: {**img, 'annotations': []} for img in data['images']}
            for anno in data['annotations']:
                images[anno['image_id']]['annotations'].append(anno)

            img_ctr = 0
            for i_idx, img in enumerate(data['images']):
                if i_idx == 0:
                    image = cv2.imread(os.path.join(path, img['file_name']))
                    height, width = image.shape[:2]

                new_img = {
                    'is_labeled': img['is_labeled'],
                    'height': height,
                    'width': width,
                    'file_name': img['file_name'],
                    'frame_id': img['id'] % 10000,
                    'original_file_name': img['file_name'],
                    'nframes': img['nframes'],
                    'id': img_ctr
                }
                new_images.append(new_img)

                for anno in images[img['id']]['annotations']:
                    if np.sum(anno['keypoints']) == 0:
                        continue

                    bbox = anno['bbox']
                    area = bbox[2] * bbox[3]
                    keypoints = np.array(anno['keypoints']).reshape([-1, 3])
                    for j in range(17):
                        if keypoints[j, 2] == 1:
                            keypoints[j, 2] = 2

                    anno['keypoints'] = keypoints.reshape([-1]).tolist()

                    new_anno = {
                        'id': img_ctr,
                        'segmentations': [],
                        'num_keypoints': 17,
                        'area': area,
                        'iscrowd': 0,
                        'keypoints': anno['keypoints'],
                        'track_id': anno['track_id'],
                        'head_box': anno['bbox_head'],
                        'image_id': img['id'],
                        'bbox': anno['bbox'],
                        'category_id': 1
                    }

                    annotations.append(new_anno)

                img_ctr += 1

        split_info = {
            'images': new_images,
            'annotations': annotations,
            'categories': categories
        }

        with open(os.path.join(save_path, f'{split}.json'), 'w') as f:
            json.dump(split_info, f, indent=4)


if __name__ == '__main__':
    main()