import pandas as pd
from FileLoader import smart_load
from Core import infer_and_convert_series

df = smart_load('test_data.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^0-9a-zA-Z_]', '', regex=True)

print('After normalization:')
print(df.columns.tolist())
print('\ndate_joined before inference:')
print(df['date_joined'])
print(f'\ndate_joined dtype before: {df["date_joined"].dtype}')

df['date_joined'] = infer_and_convert_series(df['date_joined'])

print('\ndate_joined after inference:')
print(df['date_joined'])
print(f'\ndate_joined dtype after: {df["date_joined"].dtype}')
