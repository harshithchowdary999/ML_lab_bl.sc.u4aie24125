import pandas as pd
import numpy.linalg as la
def featurematrixanditstuff():
    pd.set_option('display.max_columns', None)
    file_path = r"C:\Users\harsh\Downloads\Lab_Data.xlsx"
    df = pd.read_excel(file_path, sheet_name='Purchase data')
    x = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df[["Payment (Rs)"]].values

    xinv = la.pinv(x)
    #@ = linear algebra multiplication
    c = xinv @ y
    return c,la.matrix_rank(x),la.matrix_rank(y)

def featurematrixandclassifier():
    pd.set_option('display.max_columns', None)
    file_path = r"C:\Users\harsh\Downloads\Lab_Data.xlsx"
    df = pd.read_excel(file_path, sheet_name='Purchase data')
    df['category']=""
    row,col = df.shape
    for i in range(row):
            if df.loc[i,'Payment (Rs)']>200:
                df.loc[i,'category']="Rich"
            else:
                df.loc[i,'category']="Poor"
    cols_needed = [
        "Customer",
        "Candies (#)",
        "Mangoes (Kg)",
        "Milk Packets (#)",
        "Payment (Rs)",
        "category"
    ]

    df[cols_needed].to_excel(
        r"C:\Users\harsh\Downloads\Lab_Data.xlsx",
        sheet_name="Purchase data",
        index=False
    )
import statistics
def irctcdata():
    file_path = r"C:\Users\harsh\Downloads\Lab_Data.xlsx"
    df = pd.read_excel(file_path,sheet_name='IRCTC Stock Price')
    cols_needed = ["Date","Month","Day","Price","Open","High","Low","Close","Volume","Chg%"]
    k=df["Price"]
    mean1=statistics.mean(k)
    var1=statistics.variance(k)
    selfmean=sum(k)/len(k)
    ksqr=[i**2 for i in k]
    selfvar=(sum(ksqr)/len(k))-(selfmean**2)


    row,col = df.shape
    wed_pricesum=0
    count=0
    for i in range(row):
        if df.loc[i,'Day']=="Wed":
            wed_pricesum+=df.loc[i,'Price']
            count+=1
    samplemeanforwed=wed_pricesum/count
    april_pricesum = 0
    count = 0
    for i in range(row):
        if df.loc[i, 'Month'] == "Apr":
            april_pricesum += df.loc[i, 'Price']
            count += 1
    samplemeanforapril = april_pricesum / count

    return mean1, var1, selfmean, selfvar,samplemeanforwed,samplemeanforapril


