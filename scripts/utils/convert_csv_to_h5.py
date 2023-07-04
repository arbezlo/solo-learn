# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import io
import json

def get_cropped_bin_img(folder_path, row):
    im = Image.open(folder_path+row['img_path'])
    cropped_im = im.crop(row[['x1','y1','x2','y2']].values.tolist())
    buf = io.BytesIO()
    cropped_im.save(buf, format='JPEG')
    return buf.getvalue()

def get_class_to_idx(path:str) -> dict:
            with open(path) as json_file:
                data = json.load(json_file)
            
            return {v: k for k, v in data.items()}

def convert_csv_to_h5(csv_path: str, idx_to_class_path: str, folder_path: str, h5_path: str):
    """Converts csv dataset file to a h5 dataset.

    Args:
        csv_path (str): path to the csv dataset file.
        idx_to_class_pass (str): path a json idx to path file.  
        folder_path (str): path to the image folder.
        h5_path (str): output path of the h5 file.
    """

    df = pd.read_csv(csv_path)
    classes = np.unique(df['target'].values)
    idx_to_class = get_class_to_idx(idx_to_class_path)
    classes_name = list(idx_to_class.keys())

    with h5py.File(h5_path, "w") as h5:
        for class_name in tqdm(classes,desc="Processing classes"):
            
            cur_folder = df.loc[df["target"]==class_name]
            cur_folder_bis_ = cur_folder.copy()['img_path'].str.replace(' ','_') 
            class_group = h5.create_group(str(classes_name[int(class_name)]).replace(' ','_'))
            
            for i, row in cur_folder.iterrows():
                
                byte_im = get_cropped_bin_img(folder_path, row)
                data = np.frombuffer(byte_im, dtype="uint8")
                
                aaa = cur_folder_bis_.loc[cur_folder_bis_.index == i].values.tolist()[0]
                aaa = aaa.replace("/","\\")
                try: 
                    class_group.create_dataset(
                        aaa,
                        data=data,
                        shape=data.shape,
                        compression="gzip",
                        compression_opts=9,
                    )
                except Exception as e:
                    print(f"Error {e} on image path {aaa}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required = True)
    parser.add_argument('--idx_to_class_path', type=str, required = True)
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--h5_path", type=str, required=True)
    args = parser.parse_args()
    convert_csv_to_h5(args.csv_path , args.idx_to_class_path ,args.folder_path, args.h5_path)
