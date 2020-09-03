import torch
import os
import cv2
from torch.utils.data.dataloader import default_collate
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import numpy as np
from PIL import Image

device = torch.device("cpu")
transform_test = transforms.Compose([transforms.Resize((512, 512)),
                                     transforms.ToTensor()])
if __name__ == "__main__":
    dataset_path = 'E:\Dataset\DR\DeepDr\Onsite-Challenge1-2-Evaluation'
    csv_file_path = 'Onsite-Challenge1-2-Evaluation_full.csv'

    n_csv_path = dataset_path + '/' + csv_file_path
    if not os.path.exists(n_csv_path):
        raise EOFError('Input CSV file (' + str(n_csv_path) + ') not found')

    main_db = pd.read_csv(n_csv_path, keep_default_na=False)
    ''' Read the CSV file '''
    patient_id = main_db['patient_id']
    image_id = main_db['image_id']
    dr_level = main_db['patient_DRLevel']
    size = len(patient_id)
    result_list = []

    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=5)
    ''' Loaded pretrained weights for efficientnet-b- '''
    model.to(device)
    model.load_state_dict(torch.load('checkpoint_fold_2.pt'))
    ''' Load the last checkpoint with the best model '''
    model.eval()
    if True:
        for i in range(size):
            image_path = os.path.join(dataset_path, str(patient_id[i]), str(image_id[i]) + '.jpg')

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = Image.fromarray(image)
            pred_img = transform_test(image)
            pred_img = default_collate([pred_img])
            grade_result = model(pred_img).detach().cpu().numpy()

            grade_result = grade_result.argmax()
            result_list.append(grade_result)
            print('Image ' + str(i) + ' ---- Predict: ' + str(grade_result) + ' , GT: ' + str(dr_level[i]))

        np.save('result.npy', result_list)
    else:
        result_list = np.load('result.npy')
    # for i in range(0, len(result_list), 4):
    #     m = max([result_list[i], result_list[i + 1], result_list[i + 2], result_list[i + 3]])
    #     result_list[i] = result_list[i + 1] = result_list[i + 2] = result_list[i + 3] = m

    dr_list = dr_level.values[:size]

    acc = sum(result_list == dr_list) / size
    kappa = cohen_kappa_score(result_list, dr_list, weights='quadratic')
    print('Acc: %f' % acc)
    print('Kappa: %f' % kappa)
