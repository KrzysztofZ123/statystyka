import scipy.stats as scs
import numpy as np
import matplotlib.pyplot as plt

p = 0.7 # prawdopodobieństwo sukcesu
ilosc_prob = 100
k =  4 # liczba sukcesów w rozkładzie dwumianowym

# bernoulli dla 100 prób (podajemy prawdopodobieństwo np 0.7 i size- ilość prób, dostajemy zera i jedynki w liście)
zmienne_bernoulli = scs.bernoulli.rvs(p, size=ilosc_prob)
print('Średnia Bernoulli: ', np.mean(zmienne_bernoulli))
print('Wariancja Bernoulli: ', np.var(zmienne_bernoulli))
print('Kurtoza Bernoulli: ', scs.kurtosis(zmienne_bernoulli))
print('Skośność Bernoulli: ', scs.skew(zmienne_bernoulli))

# dwumianowy dla 100 prób (k- ile sukcesów, n- ile obserwacji, p- prawdopodobieństwo)
# jakie jest prawdopodobieństwo uzyskania k sukcesów przy n niezal. obserwacjach z prawdopodobieństwem p
zmienne_dwumianowe = []
for i in range(0, ilosc_prob + 1):
    wartosci_dwumianowe = scs.binom.pmf(i, ilosc_prob, p)
    zmienne_dwumianowe.append(wartosci_dwumianowe)
print()
print('zmienne dwumianowe', zmienne_dwumianowe)
print('Średnia dwumianowy: ', np.mean(zmienne_dwumianowe))
print('Wariancja dwumianowy: ', np.var(zmienne_dwumianowe))
print('Kurtoza dwumianowy: ', scs.kurtosis(zmienne_dwumianowe))
print('Skośność dwumianowy: ', scs.skew(zmienne_dwumianowe))
print('Suma prawdopodobieństw', np.sum(zmienne_dwumianowe))

# Poissona dla 100 prób (prawdopodobieństwo odchyleń od średniej w przedziale czasu)
rozklad_poissona = scs.poisson.rvs(ilosc_prob)
x = np.arange(scs.poisson.ppf(0.0001, ilosc_prob), scs.poisson.ppf(0.9999, ilosc_prob)) # definiujemy dolny i górny przedział
# wartości, ilosc_prob to średnia
print()
print('zmienne poissona', x)
print('Średnia poissona: ', np.mean(x))
print('Wariancja poissona: ', np.var(x))
print('Kurtoza poissona: ', scs.kurtosis(x))
print('Skośność poissona: ', scs.skew(x))


# wykres rozkładów
fig, ax = plt.subplots(1, 1)
x_dwumian = np.arange(1, 102, 1)
x_bernoulli = np.arange(1, 101, 1)
x_poisson = np.arange(1, 75, 1)
ax.plot(x_dwumian, zmienne_dwumianowe, 'b-', lw=5)
ax.plot(x_bernoulli, zmienne_bernoulli, 'g-', lw=5)
ax.plot(x_poisson, x, 'g-', lw=5)

plt.show()