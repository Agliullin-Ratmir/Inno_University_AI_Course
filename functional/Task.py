from itertools import groupby, filterfalse, takewhile, dropwhile, repeat, chain
from functools import reduce
import re
def run_length_encode(s):

    groups = groupby(s)

    encoded_groups = map(
        lambda group: group[0] + str(len(list(group[1]))),
        groups
    )

    print(reduce(lambda x, y: x + y, encoded_groups, ''))



# based on that: https://habr.com/ru/companies/otus/articles/529356/
def run_length_decode_with_one_digit(s):
    non_digits_filtered = filterfalse(lambda x: x.isdigit(), s)
    digits_filtered = filterfalse(lambda x: not x.isdigit(), s)
    dec1 = zip(non_digits_filtered, digits_filtered)
    list_dec = list(dec1)
    print(list_dec)
    decoded_groups = map(
        lambda group: repeat(group[0], times=int(group[1])),
        list_dec
    )
    print(''.join(chain.from_iterable(decoded_groups)))

def run_length_decode_with_several_digits(s):
    # Match each letter followed by one or more digits
    pattern = r'([a-zA-Z])(\d+)'
    matches = re.findall(pattern, s)

    # Reconstruct by repeating each character by its count
    decoded_groups = map(
        lambda match: repeat(match[0], int(match[1])),
        matches
    )

    print(''.join(chain.from_iterable(decoded_groups)))

run_length_encode("aaabbc") # a3b2c1
run_length_decode_with_one_digit("a3b2c1") # aaabbc
run_length_decode_with_one_digit("a10b11") # a
run_length_decode_with_several_digits("a10b11c3") # aaaaaaaaaabbbbbbbbbbbccc