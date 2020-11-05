from collections import defaultdict

f = open('simResults.txt', 'r')
lines = f.readlines()

s_lines = sorted(lines)

status_count = defaultdict(int)
for line in s_lines:
    status_count[line[:-2]] += 1

for k, v in status_count.items():
    print(k, v)