import os
from tqdm import tqdm
import argparse

def main(data_path):
    txt_train = open("data_txts/bair_train.txt", "w")

    for v in tqdm(os.listdir(data_path)):
        for f in os.listdir(os.path.join(data_path, v)):
            if f.split('.')[-1] == 'png':
                path = os.path.join(data_path, v, f)
                txt_train.write(path + '\n')

    txt_train.close()

    txt_test = open("data_txts/bair_test.txt", "w")

    for v in tqdm(os.listdir(data_path)):
        for f in os.listdir(os.path.join(data_path, v)):
            if f.split('.')[-1] == 'png':
                path = os.path.join(data_path, v, f)
                txt_train.write(path + '\n')

    txt_test.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='your data path')
    args = parser.parse_args()
    data_path = args.data_path
    main(data_path)