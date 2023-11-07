import kaggle

kaggle.api.authenticate()
# Dataset 1
kaggle.api.dataset_download_files(dataset="anubhavgoyal10/laptop-prices-dataset",path="Dataset",unzip= True)
# Dataset 2
kaggle.api.dataset_download_files(dataset="abdallahwagih/emotion-dataset",path="Dataset",unzip= True)
