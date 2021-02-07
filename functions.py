from sklearn import preprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.neighbors import LocalOutlierFactor

from scipy.spatial import distance

# Receives the DB and normalizes the data by column
def normalize(original_data):
  data = original_data.drop(columns=['Time(s)', 'Class', 'PathOrder'])
  scaler = preprocessing.MinMaxScaler()
  std_values = scaler.fit_transform(data)
  data_std = pd.DataFrame(data=std_values, columns=data.columns)
  df2 = original_data[['Time(s)','Class', 'PathOrder']].copy()
  data_normalized = data_std.join(df2)
  return data_normalized

# Receives DataFrame normalized, the drivers selected, the window size and the selected features
def build_df_final(data_normalized, drivers, block_sizes, selected_features):

  def create_column_names(block_size,selected_features):
    c_names = []
    for t in np.arange(block_size):
      for a in selected_features:
        c_names.append(a + '_s' + str(t))

    return c_names
  block_sizes = [block_sizes]
  for driver in drivers:
    for b in block_sizes:
      # Generate column names
      c_names = create_column_names(b,selected_features)

      # Create a data frame for driver and block_size = b
      data = pd.DataFrame(columns=c_names)

      # Select driver
      driver_df = data_normalized[data_normalized["Class"] == driver] 
      
      # Sweep driver records (1 record per second)
      for i in tqdm(np.arange(len(driver_df)-b)):
        row = []
        time_stamps = []

        for j in np.arange(b):
          # Get register i+j
          df_temp = driver_df.iloc[i+j]
          
          # Build row with selected features
          row = row + [df_temp[a] for a in selected_features]
          time_stamps.append(df_temp['Time(s)'])

        # Check time consistency
        df_temp = driver_df.iloc[i]
        if len(set(np.arange(df_temp['Time(s)'],df_temp['Time(s)']+b)).intersection(time_stamps) ) == len(time_stamps):
          # Add rows to dataframe if times are consistent
          #row = row +
          row_df = pd.DataFrame(np.array([row]),columns=c_names)
          data = pd.concat([data,row_df])

      #data.to_csv('driver_' + driver +'_block_'+ str(b) + 's')
  return data

def build_impostors_df(data_normalized, drivers, block_sizes, selected_features, main_driver):
  def create_column_names(block_size,selected_features):
    c_names = []
    for t in np.arange(block_size):
      for a in selected_features:
        c_names.append(a + '_s' + str(t))

    return c_names
  block_sizes = [block_sizes]
  for b in block_sizes:
    
    
    # Generate column names
    c_names = create_column_names(b,selected_features)

    # Create a data frame for driver and block_size = b
    data = pd.DataFrame(columns=c_names)
    impostores = []
    for driver in drivers:
      if (driver==main_driver):
        print('Não compara driver', driver, 'com ', main_driver)
      else:
        impostores.append(driver)
        # Select driver
        driver_df = data_normalized[data_normalized["Class"] == driver] 
        
        # Sweep driver records (1 record per second)
        for i in tqdm(np.arange(len(driver_df)-b)):
          row = []
          time_stamps = []

          for j in np.arange(b):
            # Get register i+j
            df_temp = driver_df.iloc[i+j]
            
            # Build row with selected features
            row = row + [df_temp[a] for a in selected_features]
            time_stamps.append(df_temp['Time(s)'])

          # Check time consistency
          df_temp = driver_df.iloc[i]
          if len(set(np.arange(df_temp['Time(s)'],df_temp['Time(s)']+b)).intersection(time_stamps) ) == len(time_stamps):
            # Add rows to dataframe if times are consistent
            row_df = pd.DataFrame(np.array([row]),columns=c_names)
            
            data = data.append(row_df)
    data.to_csv('data.csv')        
  return data, impostores

def split_data(data_std):
  labels = np.ones((data_std.shape[0],1))
  x_train, x_val, y_train, y_val = train_test_split(data_std,  labels, test_size=0.2, random_state=42)
  return x_train, x_val

def clusters_of_maneuvers(x_train, n_c):  
  x_train_function = pd.DataFrame(x_train, copy=True)
  kmeans = KMeans(n_clusters=n_c, random_state=42)
  kmeans.fit(x_train_function)
  
  x_train_function['K-Class'] = kmeans.labels_
  centroid_train = kmeans.cluster_centers_
  labels_train = kmeans.labels_
  return labels_train, centroid_train, x_train_function

def general_parameters():
  driver_main = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] 
  driver_main = ['D']
  impostor = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
  impostor = ['A', 'B', 'C', 'D', 'E']
  n_clusters = [1,3,5,10,30]
  n_clusters = [1]
  selected_features = ['Intake_air_pressure','Engine_soacking_time', 'Long_Term_Fuel_Trim_Bank1', 'Torque_of_friction', 'Engine_coolant_temperature', 'Steering_wheel_speed']
  window_size = [3, 5, 10, 30]
  window_size = [3,5]
  method = ['LOF']
  return driver_main, impostor, n_clusters, selected_features, window_size, method

################################ ISOLATION FOREST ###################################
def if_parameters():
  n_estimators = [5, 15, 25, 50]
  return n_estimators
def train_model_if( labels_train, centroid_train, x_train_class, x_val, ne):
  if_list = []
  for n in range(0, max(labels_train)+1):
    bola = range(0, max(labels_train)+1)
    df_cluster = x_train_class[x_train_class['K-Class'] == n]
    model = IsolationForest(n_estimators=ne, random_state=42).fit(df_cluster.drop(columns=['K-Class']))
    if_list.append({'cluster': n, 'driver': 'a', 'model': model, 'centroid': centroid_train[n], 'x_val': x_val })
  return if_list
def test_model_if(if_list, data_final, data_impostor, centroides):
  result = []
  control = []
  for i in tqdm(range(len(data_impostor))):   # Percorrer o DF de um motorista impostor
    menor = 1000
    for m in range(len(centroides)): # Percorrer entre as 10 manobras 
      dist = distance.euclidean(data_impostor.iloc[i], centroides[m]) # Calcular se uma manabora do motorista impostor se aproxima desse cluster N do motorista principal
      if (dist<menor):
        menor = dist
        manobra_correspondente = m
        # ERRO APARENTA ESTAR AQUI
    result.append(if_list[manobra_correspondente]['model'].predict([data_impostor.iloc[i].values])[0])
    control.append(manobra_correspondente)
    # ESTRANHO O VETOR DE PREDICT
  return result
#####################################################################################

################################## ONE CLASS SVM ####################################
def ocsvm_parameters():
  kernel = ['rbf', 'sigmoid']
  #kernel = ['rbf']
  nu = [ 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
  #nu = [0.00001, 0.000001]
  return kernel, nu
def train_model_ocsvm(labels_train, centroid_train, x_train_class, x_val, k, nu):
  ocsvm_list = []
  for n in range(0, max(labels_train)+1):
    df_cluster = x_train_class[x_train_class['K-Class'] == n]  
    model = OneClassSVM(kernel=k, nu=nu).fit(df_cluster.drop(columns=['K-Class']))
    ocsvm_list.append({'cluster': n, 'driver': 'a', 'model': model, 'centroid': centroid_train[n], 'x_val': x_val })
  return ocsvm_list
def test_model_ocsvm(oscvm_list, data_final, data_impostor, centroides):
  result = []
  for i in range(len(data_impostor)):   # Percorrer o DF de um motorista impostor
    menor = 1000
    for m in range(len(centroides)): # Percorrer entre as 10 manobras 
      dist = distance.euclidean(data_impostor.iloc[i], centroides[m]) # Calcular se uma manabora do motorista impostor se aproxima desse cluster N do motorista principal
      if (dist<menor):
        menor = dist
        manobra_correspondente = m
    result.append(oscvm_list[manobra_correspondente]['model'].predict([data_impostor.iloc[i].values])[0])
  return result
#####################################################################################

#########################3####### ELLIPTIC ENVELOPE #################################
def train_model_elliptic(labels_train, centroid_train, x_train_class, x_val):
  elliptic_list = []
  for n in range(0, max(labels_train)+1):
    df_cluster = x_train_class[x_train_class['K-Class'] == n]  
    model = EllipticEnvelope(contamination=0.001, random_state=42).fit(df_cluster.drop(columns=['K-Class']))
    elliptic_list.append({'cluster': n, 'driver': 'a', 'model': model, 'centroid': centroid_train[n], 'x_val': x_val })
  return elliptic_list
def test_model_elliptic(elliptic_list, data_final, data_impostor, centroides):
  result = []
  for i in range(len(data_impostor)):
    menor = 1000
    for m in range(len(centroides)):
      dist = distance.euclidean(data_impostor.iloc[i], centroides[m])
      if (dist<menor):
        menor = dist
        manobra_correspondente = m
    result.append(elliptic_list[manobra_correspondente]['model'].predict([data_impostor.iloc[i].values])[0])
  return result
#####################################################################################

###################################### DBSCAN #######################################
def train_model_dbscan(labels_train, centroid_train, x_train_class, x_val):
  dbscan_list = []
  eps_list = []
  for n in range(0, max(labels_train)+1):
    eps = 0
    anomalias = 1
    df_cluster = x_train_class[x_train_class['K-Class'] == n] 
    while(anomalias > 0):
      eps = eps + 0.05
      clustering = DBSCAN(eps=eps).fit(df_cluster.drop(columns=['K-Class']))
      anomalias = Counter(clustering.labels_ == -1)[True]
    dbscan_list.append({'cluster': n, 'driver': 'a', 'model': clustering, 'centroid': centroid_train[n], 'x_val': x_val, 'eps': eps })
    eps_list.append(eps)
  return dbscan_list, eps_list
def test_model_dbscan(dbscan_list, data_final, data_impostor, centroides, x_train):
  result = []
  for i in tqdm(range(len(data_impostor))):
    menor = 1000
    for m in range(len(centroides)):
      dist = distance.euclidean(data_impostor.iloc[i], centroides[m])
      if (dist<menor):
        menor = dist
        manobra_correspondente = m
    x_train_with_impostor = x_train[x_train['K-Class']==manobra_correspondente].append(data_impostor.iloc[i]).drop(columns=['K-Class'])

    clustering = DBSCAN(eps=dbscan_list[manobra_correspondente]['eps']).fit(x_train_with_impostor)
    result.append( -1 if clustering.labels_[-1] == -1 else 1) 

  return result
#####################################################################################

############################# LOCAL OUTLIER FACTOR ##################################
def new_train_model_lof(labels_train, centroid_train, x_train_class, x_val):
  lof_list = []
  nn_list = []
  for n in range(0, max(labels_train)+1):
    nn = 0
    anomalias = 110
    df_cluster = x_train_class[x_train_class['K-Class'] == n]
    while(anomalias > 100):
      nn = nn + 100
      clustering = LocalOutlierFactor(n_neighbors=nn).fit_predict(df_cluster.drop(columns=['K-Class']))
      anomalias = Counter(clustering == -1)[True]
    lof_list.append({'cluster': n, 'driver':'a', 'model': clustering, 'centroid': centroid_train[n], 'x_val': x_val, 'nn': nn})
    nn_list.append(nn)
  return lof_list, nn_list
def new_test_model_lof(lof_list, data_final, data_impostor, centroides, x_train):
  result = []
  for i in tqdm(range(len(data_impostor))):
    menor = 1000
    for m in range(len(centroides)):
      dist = distance.euclidean(data_impostor.iloc[i], centroides[m])
      if (dist<menor):
        menor = dist
        manobra_correspondente = m
    x_train_with_impostor = x_train[x_train['K-Class']==manobra_correspondente].append(data_impostor.iloc[i]).drop(columns=['K-Class'])

    clustering = LocalOutlierFactor(n_neighbors=lof_list[manobra_correspondente]['nn']).fit_predict(x_train_with_impostor)
    result.append( -1 if clustering[-1] == -1 else 1)
  return result

def train_model_lof(x_train):
  anomalias = 150
  n_n = 0
  while (anomalias > 100):
    n_n = n_n + 500
    model = LocalOutlierFactor(n_neighbors=n_n).fit_predict(x_train)
    anomalias = Counter(model == -1)[True]
  print(anomalias)
  print(model)
  return n_n
def test_model_lof(n_n, data_test):
  result = LocalOutlierFactor(contamination=0.1).fit_predict(data_test)
  print(result)
  return result
#####################################################################################

def evaluating_result(result, ws):
  contSeqNormais = [] #ocorrencias de 1 consecutivas
  cont = 0 #conta as occorencias de 1 consecutivas
  for i in result:
    if (i ==-1): # quando o elemento for -1 adiciona a contagem de 1s no vetor e zera a contagem
      contSeqNormais.append(cont)
      cont = 0
    else:
      cont += 1 # so cai aqui se não entra no if

  if (cont!=0): # essa linha adiciona a contagem de zeros caso o ultimo elemento não seja -1
    contSeqNormais.append(cont)

  contagemLimpa = list(filter(lambda num: num != 0, contSeqNormais))
  
  acc = (sum(contagemLimpa)/len(result))*100
  print(contagemLimpa)
  if not contagemLimpa:
    return acc, 0, 0, 0, 0
  return acc, min(contagemLimpa)-1 + ws, np.mean(contagemLimpa)-1 + ws, max(contagemLimpa)-1 + ws, np.std(contagemLimpa, ddof = 1)

def evaluating_result_old(result):
  print(Counter(result))
  score = 0
  quantidade_manobras = 0
  contador_positivos = 0
  soma = 0 
  for r in range(len(result)):
    if (result[r]!=-1):
      score = score + 1
      contador_positivos = contador_positivos + 1 
    
    if (result[r]==-1):
      if (contador_positivos>quantidade_manobras):
        quantidade_manobras = contador_positivos
        soma = soma + contador_positivos
        contador_positivos = 0  
    
  print(score)
  print(len(result))
  acc = score / len(result)*100 # Acurácia - Porcentagem de manobras que foram classificadas como normais

  return acc, 50

if __name__ == '__main__':
  print("")