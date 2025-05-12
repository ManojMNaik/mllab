#import matplotlib
#matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
data=fetch_california_housing(as_frame=True)
df=data.frame
n_columns=3
n_rows=(len(df.select_dtypes(include=['float64','int64']).columns)*2+n_columns-1)//n_columns
fig,axes=plt.subplots(n_rows,n_columns,figsize=(15,10))
axes=axes.flatten()
columns=df.select_dtypes(include=['float64','int64']).columns
for i, column in enumerate(columns):
    ax=axes[i]
    df[column].hist(bins=30,edgecolor='black',ax=ax)
    ax.set(title=f"histogram of {column}",xlabel=column,ylabel="Frequency")
    ax=axes[len(columns)+i]
    df.boxplot(column=column, grid=False, ax=ax)
    ax.set(title=f"Box plot of {column}")
plt.tight_layout()
plt.savefig("combined_plots.png")
plt.show()
