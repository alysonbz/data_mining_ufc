import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from questao3 import generete_numerics_atributes


# Carregar os dados
data = "restaurant_reviews_updated.csv"
df = pd.read_csv(data)

# Gerar os atributos numéricos
df_term_freq, tfidf_term_freq, df_matrix, tfidf_matrix = generete_numerics_atributes(df, column='Review')

# Separar os dados em treino e teste
X_train_df, X_test_df, y_train, y_test = train_test_split(df_matrix, df['Rating'], test_size=0.2, random_state=42)
X_train_tfidf, X_test_tfidf, _, _ = train_test_split(tfidf_matrix, df['Rating'], test_size=0.2, random_state=42)

# Treinar o modelo utilizando Document Frequency
model_df = MultinomialNB()
model_df.fit(X_train_df, y_train)
y_pred_df = model_df.predict(X_test_df)

# Treinar o modelo utilizando TF-IDF
model_tfidf = MultinomialNB()
model_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)

# Avaliar o desempenho
print("Desempenho com Document Frequency:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_df)}")
print(classification_report(y_test, y_pred_df))

print("\nDesempenho com TF-IDF:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_tfidf)}")
print(classification_report(y_test, y_pred_tfidf))

conf_matrix_df = confusion_matrix(y_test, y_pred_df)
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Document Frequency')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Matriz de Confusão para TF-IDF
conf_matrix_tfidf = confusion_matrix(y_test, y_pred_tfidf)
sns.heatmap(conf_matrix_tfidf, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - TF-IDF')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()