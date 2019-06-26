import csv


def sort_csv():

    with open('data/full/whats-on-the-menu/Dish.csv') as f:

        content = csv.reader(f, delimiter=',')
        lines = [l for l in content]

        content_lines = lines[1:]

        new_lines = lines[0:1] + list(sorted(filter(lambda l: len(l[1]) >= 6, content_lines), key=lambda l: (len(l[1]), l[1]), reverse=True))

        with open('data/full/whats-on-the-menu/Dish_sorted.csv', 'w') as f2:
            writer = csv.writer(f2, delimiter=',')
            writer.writerows(new_lines)

sort_csv()


# def year_only(year):
#
#     with open('data/full/whats-on-the-menu/Dish.csv') as f:
#
#         content = csv.reader(f, delimiter=',')
#         lines = [l for l in content]
#
#         content_lines = lines[1:]
#
#         new_lines = lines[0:1] + list(filter(lambda l: int(l[5]) == year, content_lines))
#
#         with open('data/full/whats-on-the-menu/Dish_only_year_{}.csv'.format(year), 'w') as f2:
#             writer = csv.writer(f2, delimiter=',')
#             writer.writerows(new_lines)
#
#
# year_only(2000)