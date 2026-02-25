import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

"""
DATASET TO DOWNLOAD: https://www.kaggle.com/datasets/nphantawee/pump-sensor-data
"""

df = pd.read_csv("data/sensor.csv")

del df['Unnamed: 0']

df = df.drop_duplicates()

del df['sensor_15']

df['sensor_50'].fillna((df['sensor_50'].mean()), inplace=True)
df['sensor_51'].fillna((df['sensor_51'].mean()), inplace=True)
df['sensor_00'].fillna((df['sensor_00'].mean()), inplace=True)
df['sensor_08'].fillna((df['sensor_08'].mean()), inplace=True)
df['sensor_07'].fillna((df['sensor_07'].mean()), inplace=True)
df['sensor_06'].fillna((df['sensor_06'].mean()), inplace=True)
df['sensor_09'].fillna((df['sensor_09'].mean()), inplace=True)

df = df.dropna()

warnings.filterwarnings("ignore")
df['date'] = pd.to_datetime(df['timestamp'])
del df['timestamp']

df = df.set_index('date')

# Extract the names of the numerical columns
df2 = df.drop(['machine_status'], axis=1)
names=df2.columns
x = df[names]
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler, pca)
pipeline.fit(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

df['pc1']=pd.Series(principalDf['pc1'].values, index=df.index)
df['pc2']=pd.Series(principalDf['pc2'].values, index=df.index)

print(principalDf.head())
print(df.head())

cleaned_df = df[['pc1', 'pc2', 'machine_status']]

cleaned_df['status'] = cleaned_df['machine_status'].map({'NORMAL': 'HEALTHY', 'RECOVERING': 'UNHEALTHY', 'BROKEN': 'UNHEALTHY'})
cleaned_df.to_csv("data/sensor_pca_cleaned.csv", index=False)