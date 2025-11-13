import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2 as cv
from torchvision import transforms
import numpy as np

class Vocalbulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.PAD_idx = 0
        self.SOS_idx = 1
        self.EOS_idx = 2
        self.UNK_idx = 3

    def __len__(self):
        return len(self.word2idx)
    
    def build_vocab(self, all_captions_list):
        word_list = set()
        for caption in all_captions_list:
            word_list.update(caption.split())
        
        for word in sorted(list(word_list)):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

class COCODataset(Dataset):
    def __init__(self, image_dir, features_dir, captions_file, vocabulary):
        self.image_dir = image_dir
        self.features_dir = features_dir
        self.vocabulary = vocabulary

        try:
            df = pd.read_csv(captions_file)
            # Nhóm caption tiếng Việt theo image_name
            self.annotations = df.groupby('image_name')['translate'].apply(list).to_dict()
            
        except Exception as e:
            print(f"Lỗi khi tải hoặc xử lý file CSV: {e}")
            self.annotations = {}

        self.image_ids = list(self.annotations.keys())
        self.feature_paths = {img_id: os.path.join(features_dir, f"{os.path.splitext(img_id)[0]}.npz") 
                              for img_id in self.image_ids}

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # ĐỌC ẢNH
        image_file = os.path.join(self.image_dir, f"{image_id}")
        image = cv.imread(image_file)
        if image is None:
            image_tensor = torch.zeros((3, 224, 224))  # Trả về tensor rỗng nếu không đọc được ảnh\
        else:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = cv.resize(image, (224, 224), interpolation=cv.INTER_LINEAR)
            image_tensor = self.transform(image)
        
        # ĐỌC ĐẶC TRƯNG
        feature_path = self.feature_paths.get(image_id)
        try:
            features = np.load(feature_path)
            V_raw = torch.tensor(features['V_features'], dtype=torch.float32)
            g_raw = torch.tensor(features['g_raw'], dtype=torch.float32)
        except:
            # Placeholder nếu file đặc trưng chưa được tạo (CHỈ DÙNG KHI TEST)
            V_raw = torch.randn(36, 2048)
            g_raw = torch.randn(2048)
        
        # XỬ LÝ CAPTION
        caption = self.annotations[image_id][0]  # Lấy caption đầu tiên
        tokens = [self.vocabulary.word2idx.get(word, self.vocabulary.UNK_idx) for word in caption.split()]
        tokens = [self.vocabulary.SOS_idx] + tokens + [self.vocabulary.EOS_idx]
        caption_tensor = torch.tensor(tokens)

        return image_tensor, V_raw, g_raw, caption_tensor, len(tokens)
    
def collate_fn(data):
    image_bacth, V_batch, g_batch, captions, lengths = zip(*data)
    max_len = max(lengths)
    padded_captions = torch.zeros(len(captions), max_len).long()

    for i, caption in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = caption[:end]

    image_bacth = torch.stack(image_bacth)
    V_batch = torch.stack(V_batch)
    g_batch = torch.stack(g_batch)

    return image_bacth, V_batch, g_batch, padded_captions, torch.tensor(lengths)


        
        