## EXNO-3-DS

### NAME: SATHYAA R
### REG NO: 212223100052

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```

![image](https://github.com/user-attachments/assets/0b8d6f44-06f3-4e19-9216-ecd625b9d935)

```
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/user-attachments/assets/24cd04f8-084b-4850-855f-84811d06adc0)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/user-attachments/assets/007dec11-b9d2-4912-b64d-b4488cc33376)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![image](https://github.com/user-attachments/assets/33194da4-9b2f-446e-80bd-088219cd0d68)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))

df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/user-attachments/assets/80805fbf-2399-4eb4-9ee3-fa9f2412daaa)

```
pd.get_dummies(df2,columns=["nom_0"])
```

![image](https://github.com/user-attachments/assets/c7f5dd5a-3221-4bf1-bfdd-d2377e6086d6)

```
pip install --upgrade category_encoders
```

```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```

![image](https://github.com/user-attachments/assets/33d44473-15d8-48a8-b537-2a8135659bdb)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
df
```

![image](https://github.com/user-attachments/assets/2565a1a5-0050-4a53-9c17-4137ea071358)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![image](https://github.com/user-attachments/assets/81a05a96-3810-40af-bea4-de577ead5cfb)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```

![image](https://github.com/user-attachments/assets/3ac35af5-4730-49d4-9926-9610ef5d40c0)

```
df.skew()
```

![image](https://github.com/user-attachments/assets/4e9c9f29-f732-43a0-859a-1f3c5f8245aa)

```
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/4080aaff-7523-4125-8b91-ede9d0626d14)

```
np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/user-attachments/assets/11ecbdd7-2ca9-473d-a2a9-ffa7851bd708)

```
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/ca10d26e-6cf9-435a-b5a3-cf7a2d4b328b)

```
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/1361c9ad-32b0-4bb8-94e4-867cdbcb8603)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/user-attachments/assets/78437f55-1de6-4819-a02d-562d2e3d3617)

```
df.skew()
```

![image](https://github.com/user-attachments/assets/8283ce2c-eccc-4d32-8615-bc7ae8a51a48)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

![image](https://github.com/user-attachments/assets/a5de0e42-7c10-43dd-a91d-54f146f3b1c3)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

![image](https://github.com/user-attachments/assets/f05ad55b-fb6c-4396-96c5-f7ba7c002a50)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/741bc12f-a50e-4a0d-9f59-6b8b023175ce)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/4d6c4264-2465-45ae-ad00-0852dec1de7e)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/0ed6c5e5-74e8-487f-ba5b-14833c2cd851)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/cfa96f97-c1b3-420f-a6f4-314fc0839247)


```
dt=pd.read_csv("titanic_dataset.csv")
dt
```

![image](https://github.com/user-attachments/assets/4895c94d-d270-4661-8dab-a45c9410d3b7)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

![image](https://github.com/user-attachments/assets/9410b5c9-038f-4033-8251-4c05a8d4e2d6)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/7196edaa-2cee-41de-b206-a78d4bde556c)


# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
