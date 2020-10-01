####### one way ANOVA on meta data
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
import pandas as pd
import numpy as np
import csv

df = pd.read_csv('meta-data_Anova.csv')
df['Coeff'] = df['Coeff'].multiply(1e-3)
#print(df["Category"])

#degrees of freedom
#df1 = 4 # num of groups - 1
#df2 = 156 # num of observations - num of groups
#df.rename(columns={ Category: "Category1", "Weight(*1e-6)": "weights"})


lm = ols('Coeff ~ Category', data=df).fit()
table = sm.stats.anova_lm(lm)
print(table)

#MultiComp = MultiComparison(df['Category'], df['Coeff'])
# Show all pair-wise comparisons:

# Print the comparisons

#print(MultiComp.tukeyhsd().summary())


# perform multiple pairwise comparison (Tukey HSD)
m_comp = pairwise_tukeyhsd(endog=df['Coeff'], groups=df['Category'], alpha=0.05)
print(m_comp)