import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_and_save_images(main_folder, class_names, train_ratio=0.8, random_state=42):
    train_folder = os.path.join(main_folder, 'train')
    test_folder = os.path.join(main_folder, 'test')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    for class_name in class_names:
        class_path = os.path.join(main_folder, class_name)

        if not os.path.exists(class_path):
            continue

        image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        train_files, test_files = train_test_split(image_files, train_size=train_ratio, random_state=random_state)

        for file_name in tqdm(train_files, desc=f"Copying {class_name} to train"):
            source_path = os.path.join(class_path, file_name)
            destination_path = os.path.join(train_folder, file_name)
            shutil.copyfile(source_path, destination_path)

        for file_name in tqdm(test_files, desc=f"Copying {class_name} to test"):
            source_path = os.path.join(class_path, file_name)
            destination_path = os.path.join(test_folder, file_name)
            shutil.copyfile(source_path, destination_path)

# Exemplo de uso
main_folder = r"C:\Users\laura\OneDrive\Área de Trabalho\PLANTAS"
class_names = ['abies_concolor', 'acer_campestre', 'amelanchier_canadensis']  # Adicione outras classes se necessário

load_and_save_images(main_folder, class_names)
