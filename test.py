import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'Dataset/churn-bigm   l-80.csv')
print(df.head())
sns.heatmap(df.corr(),annot=True)
# plt.show()
print(df.isnull().sum())
df = df.drop(['State', 'Area code', 'Total day charge', 'Total eve charge',
               'Total night charge', 'Total intl charge'],axis=1)
sns.heatmap(df.corr(),annot = True)
# plt.show()
cat = df.select_dtypes(include= 'object')
print(df.shape)
print(cat)
vmp = df['Voice mail plan'].replace(to_replace={'Yes': 1, 'No': 0})
ip = df['International plan'].replace(to_replace={'Yes': 1, 'No': 0})
print(df.head())
print(df.columns)
df['Voice mail plan']=vmp
df['International plan']=vmp
print(df.head())
print(df.columns)