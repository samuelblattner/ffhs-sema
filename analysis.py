from loader import Loader
from matplotlib import pyplot as plt

loader = Loader(5000, 0)

d = 0
years = []
years_last = []
min_year = 9999
max_year = 0
lengths = []
for dataframe in loader:

    chars = set()

    d += 1
    print('Data frame ', d)

    for name, first_appeared, last_appeared in zip(dataframe['name'], dataframe['first_appeared'], dataframe['last_appeared']):
        # years.setdefault(int(first_appeared), 0)
        # years[int(first_appeared)] += 1
        y = int(first_appeared)
        yl = int(last_appeared)
        lengths.append(len(name))

        if 'xing stew' in name.lower():
            print(name)

        if y > 1:

            if y > 2025:
                y -= 1000
            years.append(y)
            max_year = max(y, max_year)
            min_year = min(y, min_year)

        if yl > 1:

            if yl > 2025:
                yl -= 1000
            years_last.append(yl)

print('avg len ', sum(lengths)/len(lengths))
print(min_year)
print(max_year)
print(len(lengths))
plt.style.use('ggplot')
g = plt.hist([years, years_last], bins=10, range=(min_year, max_year), label=['first_appeared', 'last_appeared'])
plt.legend(loc='upper right')
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('test.png', dpi=300)   # no need for DPI setting, assuming the fonts and figures are all vector based
