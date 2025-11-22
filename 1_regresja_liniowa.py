import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('weight-height.csv', sep=';')
print(df)
# print(df.Gender.value_counts())
print(df['Gender'].value_counts())

df.Height *= 2.54
df.Weight /= 2.2
print(df)
print(df.describe().T.round(2).to_string())

df['nowa'] = (df.Height + 6) / df.Weight  # przykłąd tworzenia nowej kolumny
del (df['nowa'])

# plt.hist(df.query("Gender=='Male'").Weight, bins=30)
# # plt.show()
# plt.hist(df.query("Gender=='Female'").Weight, bins=30)
# plt.show()
#
# sns.histplot(df.query("Gender=='Male'").Weight, bins=30)
# # plt.show()
# sns.histplot(df.query("Gender=='Female'").Weight, bins=30)
# plt.show()

# sns.scatterplot(df, x='Height', y='Weight')
# plt.show()

df = pd.get_dummies(df)
print(df)
del (df['Gender_Male'])
df = df.rename(columns={'Gender_Female': 'Gender'})
print(df)

df.to_csv('waga_wzrost_po_edycji.csv')

model = LinearRegression()
model.fit(df[['Height', 'Gender']], df['Weight'])
print(f'Wspolczynnik kierunkowy: {model.coef_}')
print(f'wyraz wolny: {model.intercept_}')

print(f'waga = wzrost * 1.06960294 + plec * -8.80805024 -102.52081454490131')
print(model.predict([[192, 0], [192, 1], [160, 1], [100, 1]]))
