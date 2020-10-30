import json
file1 = open('yelp_review.json', 'r')
lines = file1.readlines()

file1.close()
ret = []
file2 = open('reviews.json', 'w')
for index in range(1000):
    x = json.loads(lines[index])
    x['isFake'] = 'N'
    ret.append(x)

with open('reviews.json', 'w') as fout:
    fout.write('\n'.join(map(json.dumps, ret)))
