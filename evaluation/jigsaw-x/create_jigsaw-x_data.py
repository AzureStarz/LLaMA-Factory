import pandas as pd

# # Step 1: Read the csv file
# df = pd.read_csv('/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/jigsaw-x/data/val/validation.csv')

# # Step 2: Get unique 'lang' values
# langs = df['lang'].unique()

# # Step 3: For each 'lang', create a new DataFrame and save it as a csv file
# for lang in langs:
#     new_df = df[df['lang'] == lang][['comment_text', 'toxic']]
#     new_df.to_csv(f'/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/evaluation/jigsaw-x/data/val/{lang}.csv', index=False)

# Step 1: Read the csv files
df1 = pd.read_csv('data/test/test.csv')
df2 = pd.read_csv('data/test/test_labels.csv')

# Step 2: Merge the two DataFrames on 'id'
df = pd.merge(df1, df2, on='id')

# Step 3: Get unique 'lang' values
langs = df['lang'].unique()

# Step 4: For each 'lang', create a new DataFrame and save it as a csv file
for lang in langs:
    new_df = df[df['lang'] == lang][['content', 'toxic']]
    new_df.to_csv(f'{lang}.csv', index=False)