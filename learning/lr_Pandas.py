#https://www.datacamp.com/community/blog/python-pandas-cheat-sheet
import pandas as pd
s=pd.Series([3,-5,'mostafa',4], index=['a',12,'c','d'])
print(s)
#--------------------------
data = {'Country': ['Belgium',  'India',  'Brazil'],

'Capital': ['Brussels',  'New Delhi',  'Brasilia'],

'Population': [11190846, 1303171035, 207847528]}

df = pd.DataFrame(data,columns=['Country',  'Capital',  'Population'])

print(df)

#Asking For Help--------------------------
help(pd.Series.loc)

