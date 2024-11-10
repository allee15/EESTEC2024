import os
import shutil
import random
import argparse

def split_data(source_dir, train_dir, test_dir, train_ratio=0.75):
    # Creează directoarele de antrenament și test dacă nu există
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Listează toate fișierele JSON din directorul sursă
    all_files = [f for f in os.listdir(source_dir)]
    print(all_files)
    # Calculează numărul de fișiere de antrenament
    train_count = int(len(all_files) * train_ratio)

    # Amestecă fișierele și împarte-le în antrenament și test
    random.shuffle(all_files)
    train_files = all_files[:train_count]
    test_files = all_files[train_count:]

    # Copiază fișierele în directoarele corespunzătoare
    for filename in train_files:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(train_dir, filename))

    for filename in test_files:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(test_dir, filename))

    print(f"Fișierele au fost împărțite: {len(train_files)} fișiere în '{train_dir}' și {len(test_files)} fișiere în '{test_dir}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split files into training and testing sets.")
    parser.add_argument("source_dir", type=str, help="Calea către directorul sursă cu fișierele JSON.")
    parser.add_argument("train_dir", type=str, help="Calea către directorul de antrenament.")
    parser.add_argument("test_dir", type=str, help="Calea către directorul de test.")
    parser.add_argument("--train_ratio", type=float, default=0.75, help="Proporția de fișiere pentru antrenament (implicit 0.75).")
    args = parser.parse_args()

    split_data(args.source_dir, args.train_dir, args.test_dir, args.train_ratio)
