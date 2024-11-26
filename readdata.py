import numpy as np
import os


def read_npz_file(file_path):
    data = np.load(file_path)
    for key, value in data.items():
        print(f"{key}: {value}")


def read_all_npz_files_in_folder(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npz"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Reading {file_path}")
            read_npz_file(file_path)


if __name__ == "__main__":
    read_all_npz_files_in_folder(".")
