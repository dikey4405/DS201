import numpy as np
import pandas as pd
import os
from Extractor.real_feature_extractor import FeatureExtractor
from tqdm import tqdm

# Configurations
image_name_col = "image_name"
image_dir = "D:\\Python\\Nam3\\DL\\DoAn\\MS_COCO\\dataset\\train_split\\image"
captions_file = "D:\\Python\\Nam3\\DL\\DoAn\\MS_COCO\\annotations\\captions_translated_Khiem.csv"
features_dir = "D:\\Python\\Nam3\\DL\\DoAn\\MS_COCO\\dataset\\features_real_2048d"

D_MODEL = 2048
D_REGION = 36

def extract_and_save_features():
    print("Bắt đầu trích xuất đặc trưng...")
    os.makedirs(features_dir, exist_ok=True)

    try:
        df = pd.read_csv(captions_file)
        image_names = df[image_name_col].unique()
    except Exception as e:
        print(f"Lỗi khi đọc file captions: {e}")
        return
    
    try:
        extractor = FeatureExtractor(d_model=D_MODEL, n_regions=D_REGION)
    except Exception as e:
        print(f"Lỗi khởi tạo FeatureExtractor: {e}. Vui lòng kiểm tra kết nối mạng và thư viện.")
        return
    
    for image_name in tqdm(image_names, desc="Trích xuất đặc trưng cho ảnh"):
        image_file = os.path.join(image_dir, image_name)
        output_name = os.path.splitext(image_name)[0]
        output_file = os.path.join(features_dir, output_name + ".npz")

        if os.path.exists(output_file):
            continue
        try:
            V_features, g_raw = extractor.extract_features(image_file)
            np.savez_compressed(
                output_file, 
                V_features=V_features, 
                g_raw=g_raw
            )
        except Exception as e:
            print(f"\nLỗi xử lý ảnh {image_name}: {e}. Bỏ qua.")
            continue
    
    print("Hoàn tất trích xuất đặc trưng.")

if __name__ == "__main__":
    extract_and_save_features()