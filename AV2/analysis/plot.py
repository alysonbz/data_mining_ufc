import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\Guilherme\Desktop\Mineração\data_mining_ufc\AV2\resultados\atributos.csv')

# 1. Informações gerais do dataset
print("Informações gerais do dataset:")
df.info()

# 2. Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe())

# 3. Distribuição das classes (label)
print("\nDistribuição das classes (label):")
print(df['label'].value_counts())

# 4. Correlação entre as variáveis numéricas
print("\nCorrelação entre as variáveis numéricas:")
print(df[['R_mean', 'G_mean', 'B_mean']].corr())

# 1. Histogramas para a distribuição dos valores de R_mean, G_mean, B_mean
plt.figure(figsize=(12, 6))

# Histograma para R_mean
plt.subplot(1, 3, 1)
plt.hist(df['R_mean'], bins=10, color='red', alpha=0.7)
plt.title('Distribuição de R_mean')
plt.xlabel('R_mean')
plt.ylabel('Frequência')

# Histograma para G_mean
plt.subplot(1, 3, 2)
plt.hist(df['G_mean'], bins=10, color='green', alpha=0.7)
plt.title('Distribuição de G_mean')
plt.xlabel('G_mean')

# Histograma para B_mean
plt.subplot(1, 3, 3)
plt.hist(df['B_mean'], bins=10, color='blue', alpha=0.7)
plt.title('Distribuição de B_mean')
plt.xlabel('B_mean')

plt.tight_layout()
plt.show()


# 3. Mapa de calor da correlação entre R_mean, G_mean e B_mean
plt.figure(figsize=(8, 6))
sns.heatmap(df[['R_mean', 'G_mean', 'B_mean']].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Mapa de Calor da Correlação')
plt.show()


# 2. Gráficos de dispersão entre os valores de R_mean, G_mean e B_mean
plt.figure(figsize=(10, 5))

# Scatter plot entre R_mean e G_mean
plt.subplot(1, 2, 1)
plt.scatter(df['R_mean'], df['G_mean'], color='purple', alpha=0.7)
plt.title('R_mean vs G_mean')
plt.xlabel('R_mean')
plt.ylabel('G_mean')

# Scatter plot entre G_mean e B_mean
plt.subplot(1, 2, 2)
plt.scatter(df['G_mean'], df['B_mean'], color='orange', alpha=0.7)
plt.title('G_mean vs B_mean')
plt.xlabel('G_mean')
plt.ylabel('B_mean')

plt.tight_layout()
plt.show()

# Listar as classes presentes no dataset
classes = df['label'].unique()

# Definir a figura para os histogramas
plt.figure(figsize=(15, 10 * len(classes)))  # Ajustar o tamanho conforme o número de classes

# Loop para criar subplots para cada classe
for idx, class_name in enumerate(classes):
    # Subplot para R_mean
    plt.subplot(len(classes), 3, idx * 3 + 1)
    plt.hist(df[df['label'] == class_name]['R_mean'], bins=10, color='red', alpha=0.5)
    plt.title(f'{class_name} - R_mean')
    plt.xlabel('R_mean')
    plt.ylabel('Frequência')

    # Subplot para G_mean
    plt.subplot(len(classes), 3, idx * 3 + 2)
    plt.hist(df[df['label'] == class_name]['G_mean'], bins=10, color='green', alpha=0.5)
    plt.title(f'{class_name} - G_mean')
    plt.xlabel('G_mean')
    plt.ylabel('Frequência')

    # Subplot para B_mean
    plt.subplot(len(classes), 3, idx * 3 + 3)
    plt.hist(df[df['label'] == class_name]['B_mean'], bins=10, color='blue', alpha=0.5)
    plt.title(f'{class_name} - B_mean')
    plt.xlabel('B_mean')
    plt.ylabel('Frequência')

plt.tight_layout()
plt.show()