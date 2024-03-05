# read ICLR text data from 'ICLR_Mistral_Embeddings.csv' and split to train/test/validation

import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('ICLR_Mistral_Embeddings.csv')
df = df.iloc[:,:5]

df['text'] = df['title'] + '. ' + df['abstract']
df = df.loc[:,['title','abstract','text']]

train, test = train_test_split(df, test_size=0.2)
test, validation = train_test_split(test, test_size=0.5)

train.to_csv('ICLR_train.csv',index=False)
test.to_csv('ICLR_test.csv',index=False)
validation.to_csv('ICLR_validation.csv',index=False)
