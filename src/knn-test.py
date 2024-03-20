from ylearn.goml.knn import KNN 
import seaborn as sns
import pandas as pd
df = sns.load_dataset("iris")
print(df.head())
sns.scatterplot(x=df['sepal_length'],y=df['sepal_width'], hue=df['species'])

