from categorical_data import df

# ## Mapping ordinal features



size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['size'] = df['size'].map(size_mapping)
df
# 2019.09.08 add
if __name__ == '__main__':
    print(df)

inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)
# 2019.09.08 add
if __name__ == '__main__':
    print(df['size'].map(inv_size_mapping))

