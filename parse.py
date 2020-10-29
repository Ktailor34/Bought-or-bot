#import json
import pandas as pd
#JSON Data
review = pd.read_json('yelp_review.json',lines=True, chunksize=1)

for r in review:
  subset = r
  break

print(subset.shape)
print(subset.head)
