import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

date = datetime.now().strftime("%Y-%m-%d")
print(date)


url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-03-31/beers.csv"

try:
    df = pd.read_csv(url)
    print("dane pobrane")
except Exception as e:
    print(f"Błąd {e}")
    print('Używam danych zapasowych')
    data = {
        'nazwa': ['IPA', 'IPA', 'Lager', 'Stout', 'Pilsner', 'Wheat', 'Porter', 'Ale', 'Bock'],
        'alkohol': [6.5, 6.5, 5.0, 7.2, 4.8, 5.2, 5.8, 5.5, 6.8],
        'goryczka': [65, 65, np.nan, 45, 30, 15, 40, 35, 25],
        'ocena': [4.2, 4.2, 3.8, 4.5, 3.9, 3.7, 4.1, 4.0, 4.3],
        'styl': ['IPA', 'IPA', 'Lager', 'Ciemne', 'Lager', np.nan, 'Ciemne', 'Jasne', 'Ciemne']
    }
    df = pd.DataFrame(data)

#print(df)
print('podstawowe informacje')
print(f'Wymiary danych: {df.shape}')
print(f"Liczba wierszy: {df.shape[0]}")
print(f"Liczba kolumn: {df.shape[1]}")

print('Podgląd danych')
print('Pierwsze 5 piw:')
print(df.head())
print("Ostatnie 5 piw")
print(df.tail())

print('Typy danych')
print(f'\n{df.info()}')


print('Statystyki numeryczne')

kolumny_numeryczne = df.select_dtypes(include='number').columns
if len(kolumny_numeryczne) > 0:
    print('Statystyki dla cech numerycznych:')
    print(df[kolumny_numeryczne].describe())
else:
    print("Brak kolumn numerycznych")

print("Statystyki kategoryczne")

kolumny_tekstowe = df.select_dtypes(include='object').columns
if len (kolumny_tekstowe) > 0:
    for kolumna in kolumny_tekstowe:
        print(f'\nKolumna: {kolumna}')
        print(f'Unikalnych wartości: {df[kolumna].unique()}')
        print(f'Liczba unikalnych wartości: {len(df[kolumna].unique())}')
        print('3 najczęstsze wartości: ')
        print(df[kolumna].value_counts().head(3))
else:
    print("Brak kolumn kategorycznych")

print("Brakujące wartości")

brakujace = df.isna().sum()

if brakujace.sum() > 0:
    print('Kolumny z brakujacymi wartościami:')
    for kolumna in df.columns:
        if df[kolumna].isna().sum() > 0:
            braki_liczbowo = df[kolumna].isnull().sum()
            braki_procentowo = (braki_liczbowo/len(df))*100
            print(f'{kolumna}: {braki_liczbowo} ({braki_procentowo:.2f}%)')

print("Tworzenie wykresów")

if 'alkohol' in df.columns and False:
    plt.figure(figsize = (10,6))

    plt.subplot(1, 3, 1)
    plt.title("Rozkład zawartości alkoholu")
    plt.xlabel("Zawartość alkoholu w (%)")
    plt.ylabel("Liczba piw")
    plt.tight_layout()
    plt.hist(df.alkohol)

    plt.subplot(1, 3, 2)
    plt.title('Rozkład zawartości alkoholu')
    plt.xlabel("Zawartość alkoholu w (%)")
    plt.ylabel("Liczba piw")
    plt.tight_layout()
    df['alkohol'].hist(bins=10, color="lightblue", edgecolor="black")

    plt.subplot(1, 3, 3)
    df.boxplot(column='alkohol', grid=False)
    plt.title("Wykres pudełkowy: zawartość alkoholu")
    #plt.savefig('wykres_1.png') #zapisywanie wykresu w folderze
    plt.show()

#rozkład ocen
if 'ocena' in df.columns and False:
    plt.figure(figsize = (8, 5))
    df['ocena'].hist(bins=8, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.title('Rozkład ocen piw')
    plt.xlabel('Ocena w skali 1-5')
    plt.ylabel('Liczba piw')
    plt.grid(axis='y', alpha=0.3)
    plt.show()

if 'alkohol' in df.columns and 'ocena' in df.columns and False:
    plt.figure(figsize = (8, 6))
    plt.scatter(df['alkohol'], df['ocena'], alpha = 0.6, s=60, color='purple')
    plt.title("zależność między zawartościa alkoholu a oceną")
    plt.xlabel('zawartość alkoholu (%)')
    plt.ylabel("ocena")
    plt.grid(True, alpha = 0.3)
    df = df.sort_values(by='alkohol')
    z = np.polyfit(df.alkohol, df.ocena, 4)
    p = np.poly1d(z)
    plt.plot(df['alkohol'], p(df['alkohol']), "r--", alpha = 0.8)
    plt.show()

if 'styl' in df.columns and False:
    plt.figure(figsize = (10, 6))
    df['styl'].value_counts().plot(kind='bar', color='orange', edgecolor='black')
    plt.title('Popularność stylów piw')
    plt.xlabel('Styl piwa')
    plt.ylabel('Liczba piw')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

#macierz korelacji
if len(kolumny_numeryczne) >= 2 and False:
    plt.figure(figsize = (8, 6))
    macierz_korelacji = df[kolumny_numeryczne].corr()
    sns.heatmap(macierz_korelacji, annot = True, center = 0)
    plt.title('Korelacje między cechami numerycznymi')
    plt.tight_layout()
    plt.show()

#analiza duplikatów

print("Analiza duplikatów")

duplikaty = df.duplicated()
if duplikaty.sum() > 0:
    print(f'Znaleziono {duplikaty.sum()} zduplikowanych wierszy')
    print('Zduplikowane wiersze: ')
    print(df[duplikaty])
else:
    print('Brak duplikatów')

print("Podsumowanie analizy")
print(f"Przeanalizowano {len(df)} piw")
print(f'Liczba cech: {len(df.columns)}')

if len(kolumny_numeryczne) > 0:
    print("Znalezione cechy numeryczne:", list(kolumny_numeryczne))

if len(kolumny_tekstowe) > 0:
    print("Znalezione cechy kategoryczne:", list(kolumny_tekstowe))

#najlepiej ocenione
if 'ocena' in df.columns and 'nazwa' in df.columns:
    print('\nTop 3 najwyżej ocenianych piw')
    najlepsze = df.nlargest(3, 'ocena')[['nazwa', 'ocena']]
    print(najlepsze)

#najwyzsza zawartosc alkoholu
if 'alkohol' in df.columns and 'nazwa' in df.columns:
    print("\n3 piwa z największą zawartościa alkoholu:")
    mocne = df.nlargest(3, 'alkohol')[['nazwa', 'alkohol']]
    print(mocne)