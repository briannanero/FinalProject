# FinalProject
Female Python Data wrangling code
[11]:

import seaborn as sns
import pandas as pd
import sklearn as sk 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
from pathlib import Path
[12]:

Marketing = pd.read_csv('marketing_campaign.csv',sep='\t')
[13]:

Marketing.head()
[13]:
ID	Year_Birth	Education	Marital_Status	Income	Kidhome	Teenhome	Dt_Customer	Recency	MntWines	...	NumWebVisitsMonth	AcceptedCmp3	AcceptedCmp4	AcceptedCmp5	AcceptedCmp1	AcceptedCmp2	Complain	Z_CostContact	Z_Revenue	Response
0	5524	1957	Graduation	Single	58138.0	0	0	04-09-2012	58	635	...	7	0	0	0	0	0	0	3	11	1
1	2174	1954	Graduation	Single	46344.0	1	1	08-03-2014	38	11	...	5	0	0	0	0	0	0	3	11	0
2	4141	1965	Graduation	Together	71613.0	0	0	21-08-2013	26	426	...	4	0	0	0	0	0	0	3	11	0
3	6182	1984	Graduation	Together	26646.0	1	0	10-02-2014	26	11	...	6	0	0	0	0	0	0	3	11	0
4	5324	1981	PhD	Married	58293.0	1	0	19-01-2014	94	173	...	5	0	0	0	0	0	0	3	11	0
5 rows × 29 columns

[14]:

#Originally ran without seperation , needed to remove to see columns
[15]:

Marketing.tail()
[15]:
ID	Year_Birth	Education	Marital_Status	Income	Kidhome	Teenhome	Dt_Customer	Recency	MntWines	...	NumWebVisitsMonth	AcceptedCmp3	AcceptedCmp4	AcceptedCmp5	AcceptedCmp1	AcceptedCmp2	Complain	Z_CostContact	Z_Revenue	Response
2235	10870	1967	Graduation	Married	61223.0	0	1	13-06-2013	46	709	...	5	0	0	0	0	0	0	3	11	0
2236	4001	1946	PhD	Together	64014.0	2	1	10-06-2014	56	406	...	7	0	0	0	1	0	0	3	11	0
2237	7270	1981	Graduation	Divorced	56981.0	0	0	25-01-2014	91	908	...	6	0	1	0	0	0	0	3	11	0
2238	8235	1956	Master	Together	69245.0	0	1	24-01-2014	8	428	...	3	0	0	0	0	0	0	3	11	0
2239	9405	1954	PhD	Married	52869.0	1	1	15-10-2012	40	84	...	7	0	0	0	0	0	0	3	11	1
5 rows × 29 columns

[16]:

# More than enough data to run analysis.
[17]:

# Recode Married and together columns - Recode single, divorced , and widowed colums
[18]:

Marketing.Marital_Status.value_counts()
[18]:
Married     864
Together    580
Single      480
Divorced    232
Widow        77
Alone         3
Absurd        2
YOLO          2
Name: Marital_Status, dtype: int64
[19]:

# Determined there were also other answers within Marital status column - will need to include alone and with single group // unable to use absurd or yolo . still using__% of data
[20]:

Marketing.sum()a
[20]:
ID                                                              12526438
Year_Birth                                                       4410125
Education              GraduationGraduationGraduationGraduationPhDMas...
Marital_Status         SingleSingleTogetherTogetherMarriedTogetherDiv...
Income                                                       115779909.0
Kidhome                                                              995
Teenhome                                                            1134
Dt_Customer            04-09-201208-03-201421-08-201310-02-201419-01-...
Recency                                                           110005
MntWines                                                          680816
MntFruits                                                          58917
MntMeatProducts                                                   373968
MntFishProducts                                                    84057
MntSweetProducts                                                   60621
MntGoldProds                                                       98609
NumDealsPurchases                                                   5208
NumWebPurchases                                                     9150
NumCatalogPurchases                                                 5963
NumStorePurchases                                                  12970
NumWebVisitsMonth                                                  11909
AcceptedCmp3                                                         163
AcceptedCmp4                                                         167
AcceptedCmp5                                                         163
AcceptedCmp1                                                         144
AcceptedCmp2                                                          30
Complain                                                              21
Z_CostContact                                                       6720
Z_Revenue                                                          24640
Response                                                             334
dtype: object
[21]:

Marketing1 = Marketing[['Marital_Status', 'MntGoldProds']]
[22]:

Marketing1.head()
[22]:
Marital_Status	MntGoldProds
0	Single	88
1	Single	6
2	Together	42
3	Together	5
4	Married	15
[23]:

def Marital_Status_recode (series): 
    if series == "Married":
        return 0
    if series == "Together": 
        return 0
    if series == "Single": 
        return 1
    if series == "Divorced": 
        return 1
    if series == "Widow": 
        return 1
    if series == "Alone":
        return 1
​
Marketing1['Marital_StatusR'] = Marketing1['Marital_Status'].apply(Marital_Status_recode)
/var/folders/p1/kvfdhsqn6pd3cv4cthctbcd00000gn/T/ipykernel_1781/775897020.py:15: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  Marketing1['Marital_StatusR'] = Marketing1['Marital_Status'].apply(Marital_Status_recode)
[24]:

Marketing1.head()
[24]:
Marital_Status	MntGoldProds	Marital_StatusR
0	Single	88	1.0
1	Single	6	1.0
2	Together	42	0.0
3	Together	5	0.0
4	Married	15	0.0
[25]:

Marketing1.head()
[25]:
Marital_Status	MntGoldProds	Marital_StatusR
0	Single	88	1.0
1	Single	6	1.0
2	Together	42	0.0
3	Together	5	0.0
4	Married	15	0.0
[26]:

# still need to drop misc answers from Marital Status column
[27]:

sns.distplot(Marketing1['MntGoldProds'])
/opt/homebrew/Cellar/jupyterlab/3.4.5/libexec/lib/python3.10/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
  warnings.warn(msg, FutureWarning)
[27]:
<AxesSubplot:xlabel='MntGoldProds', ylabel='Density'>

[28]:

# The amount of gold products is very high around zero .. but goes past a little over 200- positively askew ?
[29]:

Marketing1['Marital_StatusR'].value_counts().plot(kind='bar')
[29]:
<AxesSubplot:>

[30]:

# There are about double the amount of people that are Married or together in this data . 
[31]:

Marketing1.MntGoldProds.value_counts()
[31]:
1      73
4      70
3      69
5      63
12     63
       ..
178     1
169     1
132     1
262     1
247     1
Name: MntGoldProds, Length: 213, dtype: int64
[32]:

# Drop original Category of Marital_Status
[33]:

Marketing2 = Marketing1[['Marital_StatusR','MntGoldProds']]
[34]:

Marketing2.head()
[34]:
Marital_StatusR	MntGoldProds
0	1.0	88
1	1.0	6
2	0.0	42
3	0.0	5
4	0.0	15
[35]:

# independent variable is categorical marital status - 2 categories 
[36]:

# the amount of gold being bought is dependent on marital status - continuous
[37]:

# change what test to run ?  manova or independent t test?
[38]:

Marketing2.Marital_StatusR.value_counts()
[38]:
0.0    1444
1.0     792
Name: Marital_StatusR, dtype: int64
[39]:

# Test for assumptions of normality - Compare dependent variable with each Independent variable
[40]:

Marketing2['MntGoldProds'][Marketing2['Marital_StatusR'] == 1 ].hist()
[40]:
<AxesSubplot:>

[41]:

Marketing2['MntGoldProds'][Marketing2['Marital_StatusR'] == 0 ].hist()
[41]:
<AxesSubplot:>

[42]:

#Pretty Normally distributed - Runt independent ttest to determine significance
[43]:

ttest_ind(Marketing2.MntGoldProds[Marketing2.Marital_StatusR == 1],Marketing2.MntGoldProds[Marketing2.Marital_StatusR == 0])
[43]:
Ttest_indResult(statistic=1.2145091726601946, pvalue=0.22468170848488891)
[44]:

# No signigicant difference based on p value - Take away would not be what was expected . Could Market to both single and couples for any gold product marketing
[48]:

Marketing2.to_csv('Marketing2.csv',sep='\t', index=False)
[49]:

Marketing2.head()
[49]:
Marital_StatusR	MntGoldProds
0	1.0	88
1	1.0	6
2	0.0	42
3	0.0	5
4	0.0	15
[ ]:

​
IndependentChiSquare.ipynb
MachineLearning.ipynb
FinalCodeProject.ipynb
AnalyzingDataHandsOn.ipynb
StatisticsFinalPython.ipynb

Simple
0
7
Python 3 (ipykernel) | Idle
FinalCodeProject.ipynb
Ln 1, Col 1
