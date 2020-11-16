import pandas as pd

data = pd.read_json('reviews.json', lines=True)
data2 = pd.read_json('yelp_user.json', lines=True)

merged_inner = pd.merge(left=data, right=data2, left_on='user_id', right_on='user_id')
waffles = ['review_id','business_id','useful_x','funny_x','cool_x','compliment_hot','compliment_more','compliment_profile','compliment_list','compliment_cute','compliment_note','compliment_plain','compliment_cool','compliment_funny','compliment_writer','compliment_photos','average_stars','cool_y','elite','fans','yelping_since','name','date','useful_y','funny_y']
merged_inner = merged_inner.drop(waffles, axis=1)

print(merged_inner)
