import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files(dataset="anubhavgoyal10/laptop-prices-dataset",path="Dataset",unzip= True)

