import numpy as np

from categorical_data import df

# create a mapping dict
# to convert class labels from strings to integers
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
class_mapping
# 2019.09.08 add
if __name__ == '__main__':
    print(class_mapping)



# to convert class labels from strings to integers
df['classlabel'] = df['classlabel'].map(class_mapping)
df
# 2019.09.08 add
if __name__ == '__main__':
    print(df)



# reverse the class label mapping
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df
# 2019.09.08 add
if __name__ == '__main__':
    print(df)