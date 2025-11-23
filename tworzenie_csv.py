import pandas as pd
import random

random.seed(42)

imiona = ["Anna", "Ewa", "Jola", "Paweł", "Wojciech", "Kamil", "Agata", "Magda"]
miasta = ["Łódź", "Wrocław", "Warszawa", "Legnica", " Gdańsk", "Pcim", "Kraków"]

n = 1000

dane = {
    "imie": [random.choice(imiona) for _ in range(n)],
    "miasto": [random.choice(miasta) for _ in range(n)],
    "dochód": [random.randint(5800, 25800) for _ in range(n)],
    "ma dzieci": [],
    "liczba dzieci": [],
    "ma zwierze": [],
    "jakie zwierze": [],
}

for _ in range(n):
    if random.choice([True, False]):
        dane["ma dzieci"].append("Tak")
        dane["liczba dzieci"].append(random.randint(1, 4))
    else:
        dane["ma dzieci"].append("Nie")
        dane["liczba dzieci"].append(0)

    if random.choice([True, False]):
        dane["ma zwierze"].append("Tak")
        dane["jakie zwierze"].append(random.choice(["pies", "kot", "chomik", "papuga"]))
    else:
        dane["ma zwierze"].append("Nie")
        dane["jakie zwierze"].append("brak")

df = pd.DataFrame(dane)

df.to_csv("dane_ludzie.csv", index=False, encoding="utf-8-sig")
print("zapisany plik csv")