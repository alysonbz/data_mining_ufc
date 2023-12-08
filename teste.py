from tkinter import filedialog

def load_soaps_image():
    file_path = filedialog.askopenfilename(title="Selecione a imagem")
    return file_path
