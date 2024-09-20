import pandas as pd

# 4. Armazenamento dos atributos em CSV
def save_to_csv(features, labels, filename):
    """Salva os atributos extraídos e seus rótulos em um arquivo CSV."""
    data = pd.DataFrame(features)
    data['label'] = labels
    data.to_csv(filename, index=False)
