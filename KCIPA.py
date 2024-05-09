import random
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
class KCIA:
    def fit_resample(self, X_train,y_train,n_nbor,Mutation_rate,Crossover_rate):
        df1 = pd.DataFrame(X_train)
        df2 = pd.DataFrame(y_train)
        dataset = pd.concat([df1, df2], axis=1)
        dataset = pd.DataFrame(dataset)
        dataset1=dataset[dataset.iloc[:,-1]==1]
        dataset0=dataset[dataset.iloc[:,-1]==0]
        n0=dataset0.shape[0]
        n1=dataset1.shape[0]
        ns=n0-n1
        print(f'需要生成{ns}个样本')
        df=dataset1.iloc[:,:-1]
        def calculate_x(x):
            return sum(x)
        def calculate_xy(x, y):
            result = [1 if xi or yi else 0 for xi, yi in zip(x, y)]
            return sum(result)
        def dis(x,y):
            dis=(calculate_xy(x,y)-min(calculate_x(x),calculate_x(y)))/max(calculate_x(x),calculate_x(y))
            return dis
        def binary_to_decimal(binary_list):
            binary_str = ''.join(map(str, binary_list))
            return int(binary_str, 2)
        binary_df = pd.DataFrame()
        l_onehot=[]
        l1=[0]
        lenss=0
        for col in df.columns:
            df[col] = df[col].astype(float)
            integer_part = df[col].apply(lambda x: int(x))
            decimal_part = df[col] - integer_part
            max_decimal_length = decimal_part.apply(lambda x: len(str(x).split('.')[1])).max()
            decimal_part = decimal_part.apply(lambda x: round(x, max_decimal_length))
            sign_column = df[col].apply(lambda x: int(x < 0))
            integer_part_binary = integer_part.apply(lambda x: bin(abs(int(x)))[2:])
            decimal_part_binary = decimal_part.apply(
                lambda x: bin(int(abs(x) * 10 ** len(str(x).split('.')[1])))[2:])
            max_len1 = max(integer_part_binary.apply(len)) + 1
            max_len2 = max(decimal_part_binary.apply(len))
            l_onehot.append(max_len1)
            l_onehot.append(max_len2)
            lenss = lenss + max_len1 + max_len2 + 1
            l1.append(lenss)
            integer_part_binary = integer_part_binary.apply(lambda x: '0' * (max_len1 - len(x)) + x)
            decimal_part_binary = decimal_part_binary.apply(lambda x: '0' * (max_len2 - len(x)) + x)
            for i in range(max_len1):
                binary_df[f'{col}_integer_binary_{i}'] = [int(x[i]) for x in integer_part_binary]
            binary_df.insert(binary_df.columns.get_loc(f'{col}_integer_binary_0'), f'{col}_sign_binary', sign_column.values)
            for i in range(max_len2):
                binary_df[f'{col}_decimal_binary_{i}'] = [int(x[i]) for x in decimal_part_binary]
        binary_df=np.array(binary_df)
        def calculate_n_neighbors(df, n):
            neighbors_model = NearestNeighbors(n_neighbors=n, metric=dis)
            neighbors_model.fit(df)
            distances, indices = neighbors_model.kneighbors(df)
            result_df = pd.DataFrame(index=df.index)
            for i in range(n):
                result_df[f'Neighbor_{i + 1}_Index'] = indices[:, i]
            return result_df
        def mutate(vector, mutation_rate):
            mutated_vector = vector.copy()
            for i in range(len(mutated_vector)):
                if i not in l1:
                    if random.random() < mutation_rate:
                        mutated_vector[i] = 1 - mutated_vector[i]
            return mutated_vector
        def crossover(vector1, vector2, crossover_rate):
            crossover_point = random.randint(1, len(vector1) - 1)
            if random.random() < crossover_rate:
                crossed_vector1 = np.concatenate((vector1[:crossover_point], vector2[crossover_point:]))
            else:
                crossed_vector1 = vector1
            return crossed_vector1
        df = np.array(df)
        df_1=df
        nb=0
        for z in range(1000):
            nb = 0+nb
            print(f'第{z}轮')
            print(f'满足条件的有{nb}个')
            if nb == ns:
                break
            newdf1=[]
            arr = calculate_n_neighbors(pd.DataFrame(binary_df), n=n_nbor)
            for j in range(ns-nb):
                arr1 = np.array(arr.iloc[:, 1:])
                la=arr1.shape[0]
                if j >=la-1:
                 j = random.randint(0, la-1)
                x=arr1[j]
                a=random.randint(0, n_nbor-2)
                i=x[a]
                vector1=mutate(binary_df[i],mutation_rate=Mutation_rate)
                vector2=crossover(vector1,binary_df[a],crossover_rate=Crossover_rate)
                newdf1.append(vector2)
            for vector2 in newdf1:
                vector2_df = pd.DataFrame(vector2)
                vector2_df = vector2_df.transpose()
                vector2_dfnew = pd.DataFrame()
                s = 0
                for i, x in enumerate(l1):
                    if i == len(l1) - 1:
                        break
                    onehotpartv = vector2_df.iloc[:, l1[i]:l1[i + 1]]
                    # 分成整数列和小数列
                    integer_part_index = l_onehot[2 * i]
                    integer_part = onehotpartv.iloc[:, 1:integer_part_index]
                    decimal_part = onehotpartv.iloc[:, integer_part_index:]
                    integer_part_decimal = integer_part.apply(binary_to_decimal, axis=1)
                    decimal_part_decimal = decimal_part.apply(binary_to_decimal, axis=1)
                    sign = onehotpartv.iloc[:, 0]
                    result = -float((integer_part_decimal.astype(str) + '.' + decimal_part_decimal.astype(str))) * (
                                2 * sign - 1)
                    vector2_dfnew.loc[:, s] = result
                    s = s + 1
                vector2_dfnew = np.array(vector2_dfnew).astype(float)
                vector2_dfnew = np.array(vector2_dfnew).astype(float)
                sample_center = np.mean(df_1, axis=0)
                covariance_matrix = np.cov(df_1.T)
                mahalanobis_distances = [mahalanobis(x, sample_center, covariance_matrix) for x in df_1]
                threshold =max(mahalanobis_distances)
                vector2_distance = mahalanobis(vector2_dfnew.reshape(-1), sample_center, covariance_matrix)
                if vector2_distance < threshold:
                    nb=nb+1
                    df=np.vstack([df, vector2_dfnew])
        re_df = pd.DataFrame(df)
        re_df.to_csv('re.csv')
        re_df.insert(len(re_df.columns), column='label', value=1)
        re_df.columns=dataset0.columns
        OVS_df=pd.concat([re_df,dataset0], ignore_index=True)
        OVS_df.to_csv('ovs.csv')
        X_r=OVS_df.iloc[:,:-1]
        y_r = OVS_df.iloc[:, -1]
        frequency = y_r.value_counts()
        print(frequency)
        return X_r,y_r













