import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset
import nibabel as nib
from num2vect import num2vect

# %% 定義dataset
class MRIDataset_SM(Dataset):
    def __init__(self, data_folders_set, labels_df, bin_range, augment = False, crop_coords = None):
        """
        物件:
            data_folders_set: 影像資料夾的路徑列表，list的資料型態
            labels_df: 包含 [Filename, Age]的DataFrame型態，這個在csv已經處理好了
            bin_range: 年齡範圍，例如(5, 100)，是tuple的資料型態，為了要給 num2vect 使用
            bin_step: 每個 bin 的間隔，資料型態是int，這邊固定使用1(我們都是以1歲當成一個年齡區間)，一樣要給 num2vect 使用
            sigma: 與論文一樣設為1，資料型態為float，一樣要給 num2vect 使用
            augment: 是否要做資料增量，預設是關掉，只有在訓練集要打開
            crop_coords: 是否要做影像裁切，如果沒給數值會自動判定不裁切，如果給定裁切座標，會依照裁切座標進行裁切
            file_paths: 存放找到影像
            ages: file_paths找到的影像，該影像年齡
        """
        self.data_folders_set = data_folders_set
        self.labels_df = labels_df
        self.bin_range = bin_range
        self.bin_step = 1
        self.sigma = 1
        self.augment = augment
        self.crop_coords = crop_coords 
        self.file_paths = []  # 建立完整檔案清單，包含檔名和完整路徑
        self.ages = []

        
        for _, row in labels_df.iterrows():  # row 變數會列出（Filename HCA9992517.nii、Age 54.0、Age Bin 45~55）
            filename = row["Filename"]       # 取出row內的Filename欄
            age = float(row["Age"])          # 取出row內的Age欄
            for folder in data_folders_set:  # 從data_folders_set中找這個資料的檔案路徑
                full_path = os.path.join(folder, filename)  # 找指定檔案後建立路徑
                if os.path.exists(full_path):
                    self.file_paths.append(full_path)   #將指定檔案路徑存放file_paths，年齡存放於ages
                    self.ages.append(age)
                    break
            else:
                raise FileNotFoundError(f"❌ 找不到 {filename} 在指定資料夾中")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        image = nib.load(image_path).get_fdata().astype('float32')  # 載入.nii檔
        
        # ======= 裁切影像 =======
        if self.crop_coords is not None:
            min_coords, max_coords = self.crop_coords
            image = image[
                min_coords[0]:max_coords[0]+1,
                min_coords[1]:max_coords[1]+1,
                min_coords[2]:max_coords[2]+1
            ]
        
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W, D)

        # ======= Data Augmentation =======
        if self.augment:
            # 1. 隨機平移 (沿每個軸 0, 1, 2 個體素)
            shift_vals = [random.choice([0, 1, 2]) for _ in range(3)]
            image = torch.roll(image, shifts=shift_vals, dims=(1, 2, 3))

            # 2. 50% 機率沿 x 軸翻轉
            if random.random() < 0.5:
                image = torch.flip(image, dims=[1])  # x 軸翻轉

        age_label = self.ages[idx]
        label_soft, _ = num2vect(age_label, self.bin_range, self.bin_step, self.sigma)  #轉成soft label
        label_soft = torch.tensor(label_soft, dtype=torch.float32)  #轉成 pytorch 的tensor 型態
        return image, age_label, label_soft

# 動態抽樣
def sample_balanced_epoch(df, selected_counts, sample_per_decade=210):
    """
    保證每個樣本至少被抽一次。
    5~75 歲 decade：先抽未抽過樣本，再補缺額，補缺額時選擇被抽次數最少的樣本
    75~85、85~100：全部取
    """
    sampled_list = []

    # 5~75 歲 decade
    for start in range(5, 75, 10):
        end = start + 10
        subset = df[(df['Age'] >= start) & (df['Age'] < end)].copy()
        if subset.empty:
            continue

        # 記錄每筆 sample 被抽過的次數
        subset['select_count'] = subset['Filename'].map(selected_counts).fillna(0)

        # 剩餘未抽樣的樣本（select_count = 0）
        remaining_unselected = subset[subset['select_count'] == 0].copy()

        # 先取未抽過的
        chosen = remaining_unselected.copy()

        # 計算還需要多少樣本補足 sample_per_decade
        n_needed = sample_per_decade - len(chosen)

        if n_needed > 0:
            # 從已抽過樣本中選 select_count 最少的補足
            already_selected = subset[subset['select_count'] > 0].copy()
            already_selected = already_selected.sort_values('select_count', ascending=True)
            chosen2 = already_selected.head(n_needed)
            chosen = pd.concat([chosen, chosen2], ignore_index=True)

        # 如果剩餘未抽樣數量已經超過 sample_per_decade，隨機抽取 sample_per_decade
        if len(chosen) > sample_per_decade:
            chosen = chosen.sample(n=sample_per_decade, random_state=None)

        sampled_list.append(chosen)

    # 75~85 歲全部加入
    subset_75_85 = df[(df['Age'] >= 75) & (df['Age'] < 85)].copy()
    if not subset_75_85.empty:
        sampled_list.append(subset_75_85)

    # 85~100 歲全部加入
    subset_85_100 = df[(df['Age'] >= 85) & (df['Age'] <= 100)].copy()
    if not subset_85_100.empty:
        sampled_list.append(subset_85_100)

    # 合併
    epoch_df = pd.concat(sampled_list, ignore_index=True)

    # 更新 selected_counts
    for filename, count in epoch_df['Filename'].value_counts().items():
        selected_counts[filename] = selected_counts.get(filename, 0) + count

    return epoch_df, selected_counts


