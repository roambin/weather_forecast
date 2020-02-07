import pandas as pd
def getData():
    df=pd.DataFrame({"Month":[], "value":[]})
    data = pd.read_csv("data.csv")
    for i in range(len(data)-1):
        for j in range(12):
            if data.iloc[i][j+1]==-99.99:
                df.loc[12*i+j]=[str(int(data.iloc[i][0]))+"-"+str(j+1),int(100)]
            else:
                df.loc[12*i+j]=[str(int(data.iloc[i][0]))+"-"+str(j+1),100+int(10*data.iloc[i][j+1])]
    df.to_csv ("dataCsv.csv" , encoding = "utf-8",index=0)
    return
