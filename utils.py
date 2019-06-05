import csv


def sort_csv():

    with open('data/full/whats-on-the-menu/Dish.csv') as f:

        content = csv.reader(f, delimiter=',')
        lines = [l for l in content]

        content_lines = lines[1:]

        new_lines = lines[0:1] + list(sorted(filter(lambda l: len(l[1]) >= 6, content_lines), key=lambda l: (len(l[1]), l[1])))

        with open('data/full/whats-on-the-menu/Dish_sorted.csv', 'w') as f2:
            writer = csv.writer(f2, delimiter=',')
            writer.writerows(new_lines)


sort_csv()