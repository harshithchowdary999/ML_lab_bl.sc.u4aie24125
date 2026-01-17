import pandas as pd
import numpy as np
import numpy.linalg as la
import statistics

import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


def featurematrixanditstuff():
    pd.set_option('display.max_columns', None)
    file_path = r"C:\Users\harsh\Downloads\Lab_Data.xlsx"

    df = pd.read_excel(file_path, sheet_name='Purchase data')
    x = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df[["Payment (Rs)"]].values

    xinv = la.pinv(x)
    c = xinv @ y

    return c, la.matrix_rank(x), la.matrix_rank(y)



def featurematrixandclassifier():
    file_path = r"C:\Users\harsh\Downloads\Lab_Data.xlsx"

    df = pd.read_excel(file_path, sheet_name='Purchase data')
    df['category'] = ""

    row, col = df.shape
    for i in range(row):
        if df.loc[i, 'Payment (Rs)'] > 200:
            df.loc[i, 'category'] = "Rich"
        else:
            df.loc[i, 'category'] = "Poor"

    cols_needed = [
        "Customer",
        "Candies (#)",
        "Mangoes (Kg)",
        "Milk Packets (#)",
        "Payment (Rs)",
        "category"
    ]

    df[cols_needed].to_excel(
        r"C:\Users\harsh\Downloads\Lab_Data_updated.xlsx",
        sheet_name="Purchase data",
        index=False
    )


def irctcdata():
    file_path = r"C:\Users\harsh\Downloads\Lab_Data.xlsx"
    df = pd.read_excel(file_path, sheet_name='IRCTC Stock Price')

    k = df["Price"]

    mean1 = statistics.mean(k)
    var1 = statistics.variance(k)

    selfmean = sum(k) / len(k)
    selfvar = (sum([i**2 for i in k]) / len(k)) - selfmean**2

    row = len(df)

    wed_sum = 0
    wed_cnt = 0
    for i in range(row):
        if df.loc[i, "Day"] == "Wed":
            wed_sum += df.loc[i, "Price"]
            wed_cnt += 1

    apr_sum = 0
    apr_cnt = 0
    for i in range(row):
        if df.loc[i, "Month"] == "Apr":
            apr_sum += df.loc[i, "Price"]
            apr_cnt += 1

    return mean1, var1, selfmean, selfvar, wed_sum / wed_cnt, apr_sum / apr_cnt

def thyroid_question():
    file_path = r"C:\Users\harsh\Downloads\Lab_Data.xlsx"
    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

    nominal_columns = df.select_dtypes(include='object').columns
    mean_val = df.mean(numeric_only=True)
    var_val = df.var(numeric_only=True)

    return nominal_columns, mean_val, var_val



def jc_smc_thyroid():
    file_path = r"C:\Users\harsh\Downloads\Lab_Data.xlsx"
    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

    d1, d2 = df.iloc[0], df.iloc[1]
    binary_cols = [c for c in df.columns if df[c].dropna().nunique() == 2]

    v1 = d1[binary_cols]
    v2 = d2[binary_cols]

    f11 = ((v1 == 1) & (v2 == 1)).sum()
    f00 = ((v1 == 0) & (v2 == 0)).sum()
    f10 = ((v1 == 1) & (v2 == 0)).sum()
    f01 = ((v1 == 0) & (v2 == 1)).sum()

    den_jc = f11 + f10 + f01
    den_smc = f11 + f10 + f01 + f00

    JC = f11 / den_jc if den_jc != 0 else 0
    SMC = (f11 + f00) / den_smc if den_smc != 0 else 0

    return JC, SMC


def cosine_similarity_thyroid():
    file_path = r"C:\Users\harsh\Downloads\Lab_Data.xlsx"
    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

    df_encoded = pd.get_dummies(df)
    v1 = df_encoded.iloc[0].values
    v2 = df_encoded.iloc[1].values

    cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cosine_sim



def heatmap_similarity():
    file_path = r"C:\Users\harsh\Downloads\Lab_Data.xlsx"
    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI").iloc[:20]

    binary_cols = [c for c in df.columns if df[c].dropna().nunique() == 2]

    def jc_smc(v1, v2):
        f11 = ((v1 == 1) & (v2 == 1)).sum()
        f00 = ((v1 == 0) & (v2 == 0)).sum()
        f10 = ((v1 == 1) & (v2 == 0)).sum()
        f01 = ((v1 == 0) & (v2 == 1)).sum()

        jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) != 0 else 0
        smc = (f11 + f00) / (f11 + f10 + f01 + f00) if (f11 + f10 + f01 + f00) != 0 else 0

        return jc, smc

    JC = np.zeros((20, 20))
    SMC = np.zeros((20, 20))

    for i in range(20):
        for j in range(20):
            JC[i][j], SMC[i][j] = jc_smc(
                df.iloc[i][binary_cols],
                df.iloc[j][binary_cols]
            )

    encoded = pd.get_dummies(df).to_numpy(dtype=float)
    norm = np.linalg.norm(encoded, axis=1, keepdims=True)
    COS = (encoded @ encoded.T) / (norm @ norm.T)

    sns.heatmap(JC); plt.title("JC Heatmap")
    plt.savefig("JC_heatmap.png"); plt.clf()

    sns.heatmap(SMC); plt.title("SMC Heatmap")
    plt.savefig("SMC_heatmap.png"); plt.clf()

    sns.heatmap(COS); plt.title("Cosine Heatmap")
    plt.savefig("COS_heatmap.png"); plt.clf()

def impute_data():
    file_path = r"C:\Users\harsh\Downloads\Lab_Data.xlsx"
    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

    num_cols = df.select_dtypes(exclude='object').columns
    cat_cols = df.select_dtypes(include='object').columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def normalize_data():
    file_path = r"C:\Users\harsh\Downloads\Lab_Data.xlsx"
    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

    num_cols = df.select_dtypes(exclude='object').columns
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    c, rx, ry = featurematrixanditstuff()
    print("C:\n", c)
    print("rank(X):", rx, " rank(Y):", ry)

    featurematrixandclassifier()
    print("Classifier done")

    m, v, sm, sv, wed, apr = irctcdata()
    print("Mean:", m, "Var:", v)
    print("Self Mean:", sm, "Self Var:", sv)
    print("Wed Mean:", wed, "Apr Mean:", apr)

    nom, mean_t, var_t = thyroid_question()
    print("Nominal:", list(nom))
    print("Mean:\n", mean_t)
    print("Var:\n", var_t)

    jc, smc = jc_smc_thyroid()
    print("JC:", jc, "SMC:", smc)

    cos = cosine_similarity_thyroid()
    print("COS:", cos)

    heatmap_similarity()
    print("Heatmaps saved")

    impute_data()
    print("Imputation done")

    normalize_data()
    print("Normalization done")


if __name__ == "__main__":
    main()
