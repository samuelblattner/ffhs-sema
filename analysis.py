import string

from loader import Loader
from matplotlib import pyplot as plt

from loader3 import MEAT_LIST

loader = Loader(5000, 0)

d = 0
years = []
years_last = []
min_year = 9999
max_year = 0
lengths = []
spans = []
more = 0
less = 0

DICTIONARIES = {
    'German': {
        'path': 'data/dictionaries/deutsch.txt',
        'count': 0,
        'dict': {}
    },
    'English': {
        'path': 'data/dictionaries/english3.txt',
        'count': 0,
        'dict': {}
    },
    'French': {
        'path': 'data/dictionaries/francais.txt',
        'count': 0,
        'dict': {}
    },
    'Italian': {
        'path': 'data/dictionaries/italiano.txt',
        'count': 0,
        'dict': {}
    },
    'Spanish': {
        'path': 'data/dictionaries/espanol.txt',
        'count': 0,
        'dict': {}
    },
    'Dutch': {
        'path': 'data/dictionaries/nederlands3.txt',
        'count': 0,
        'dict': {}
    },
    'All': {
        'count': 0,
    }
}

menus_appeared_hist = []
times_appeared_hist = []
word_lengths = []
name_lengths = []
veg_violations = 0

for name, dictionary in DICTIONARIES.items():
    if 'path' in dictionary:
        with open (dictionary['path'], 'r', encoding='ISO-8859-15') as f:
            DICTIONARIES[name]['dict'] = {l.strip().lower(): 0 for l in f.readlines()}

def get_dicts(word):
    multi = False
    hits = 0
    hit_dict = None
    for name, dictionary in DICTIONARIES.items():
        if 'dict' in dictionary and word in dictionary['dict']:
            hits += 1
            hit_dict = name
        if hits > 1:
            return 'All'

    if hits == 1:
        return hit_dict
    return None


check_names = {}

repetition = 0
in_line_repetitions = []
legible_words = 0
unlegible_words = 0

with open('data/output/2000-bNONVEG-VEG_1.5_100_2194_75_512_2_0.1_15_y.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()

        name_lengths.append(len(line))
        if line in check_names:
            repetition += 1
        else:
            inline_rep = 0
            checked_words = []
            words = line.strip(string.punctuation).split(' ')
            for w, word in enumerate(words):
                check_word = word.strip(string.punctuation).strip().lower()
                for meat_word in MEAT_LIST:
                    if meat_word in check_word:
                        veg_violations += 1

                in_dict = get_dicts(word.strip(string.punctuation).lower())
                if in_dict is None:
                    unlegible_words += 1
                else:
                    legible_words += 1
                    DICTIONARIES[in_dict]['count'] += 1

                word_lengths.append(len(word))
                if word not in checked_words:
                    checked_words.append(word)
                    for w2, word2 in enumerate(words):
                        if w != w2 and word.lower() == word2.lower():
                            inline_rep += 1

            in_line_repetitions.append(inline_rep / len(words))

            check_names[line] = 0

num_meat_lines = 0
lines = 0
for dataframe in loader:

    chars = set()

    d += 1
    print('Data frame ', d)

    for name, menus_appeared, times_appeared, first_appeared, last_appeared in zip(dataframe['name'], dataframe['menus_appeared'], dataframe['times_appeared'],
                                                                                   dataframe['first_appeared'], dataframe['last_appeared']):
        # years.setdefault(int(first_appeared), 0)
        # years[int(first_appeared)] += 1
        y = int(first_appeared)
        yl = int(last_appeared)
        lengths.append(len(name))

        spans.append(yl - y)
        menus_appeared_hist.append(int(menus_appeared))
        times_appeared_hist.append(int(times_appeared))

        if menus_appeared > 2:
            more += 1
        else:
            less += 1

        for word in name.split(' '):
            if word.strip(string.punctuation).strip().lower() in MEAT_LIST:
                num_meat_lines += 1

        lines += 1

        # for check_name, _ in check_names.items():
        #     if check_name.lower() in name.lower():
        #         check_names[check_name] += 1

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

print('MEAT: ', num_meat_lines / lines)
print('Total Words: ', legible_words + unlegible_words)
print('Legible: ', legible_words / (legible_words + unlegible_words) * 100)
print('Illegible: ', unlegible_words / (legible_words + unlegible_words) * 100)
print('Avg word length: ', sum(word_lengths) / len(word_lengths))
print('Avg name length: ', sum(name_lengths) / len(name_lengths))
print('New: ', (1 - len(list(filter(lambda o: o[1] > 0, check_names.items()))) / len(check_names.items())) * 100)
print('Repetition: ', repetition/len(check_names.items()))
print('Avg inline repetition: ', sum(in_line_repetitions)/len(in_line_repetitions))
print('Vegetarian violations: ', veg_violations/legible_words)

print('Dictionaries:\n')
for name, dictionary in DICTIONARIES.items():
    print('{}: {}, {}'.format(name, dictionary['count'], dictionary['count']/legible_words * 100))

print(times_appeared_hist[:10])
print(times_appeared_hist[200000])
print('MORE than 2: ', more)
print('LESS than 2: ', less)

print(min_year)
print(max_year)
print(len(lengths))
print(spans[0:20])
print(min(spans), max(spans))
sorted_lengths = list(sorted(lengths))
print(sorted_lengths[int(len(sorted_lengths) / 2)])
plt.style.use('ggplot')
# g = plt.hist([years, years_last], bins=20, range=(min_year, max_year), label=['first_appeared', 'last_appeared'])
# g = plt.hist([menus_appeared_hist, times_appeared_hist], bins=20, range=(1, 100), label=['menus_appeared', 'times_appeared'])
# g = plt.hist(spans, bins=100, range=(0, 20), label=['span',])
# g = plt.hist([lengths], bins=100, range=(0, 200))
g = plt.boxplot([lengths])
plt.ylim(-10, 100)
plt.legend(loc='upper right')
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('test.pdf', dpi=300, format='pdf')  # no need for DPI setting, assuming the fonts and figures are all vector based
