import pandas as pd
import numpy as np
from tqdm import tqdm
import functions as fc

# Utilizar somente quando quiser normalizar os dados novamente
#print('Normalizing data')
#original_data = pd.read_csv("https://raw.githubusercontent.com/cggcaio/Anomaly-Detection-for-Driver-Identification/master/Data_Bases/KIA_DB/Driving%20Data(KIA%20SOUL)_(150728-160714)_(10%20Drivers_A-J).csv")
#data_normalized = (normalize(original_data))

#data_normalized = pd.read_csv("https://raw.githubusercontent.com/cggcaio/Anomaly-Detection-for-Driver-Identification/master/Data_Bases/KIA_DB/data_normalized.csv")
data_normalized = pd.read_csv('data_normalized.csv')


columns = ['Method', 'Time_Window', 'Driver','Impostors', 'Method_Parameters', '%_FAR', 'Min_Time_Detection(s)', 'Average_Time_Detection(s)', 'Max_Time_Detection(s)', 'Standard_deviation(s)','%_FRR', 'Min_Time_Detection(s)', 'Average_Time_Detection(s)', 'Max_Time_Detection(s)', 'Standard_deviation(s)', 'Number_clusters']
results = pd.DataFrame([], columns=columns)

driver_main, impostor, n_clusters, selected_features, window_size, method = fc.general_parameters()
n_estimators = fc.if_parameters()
kernel, nu = fc.ocsvm_parameters()

for driver in driver_main:
  for ws in window_size:
    print("Building DF for Driver", driver, "with Window_Size", ws)
    data_final = fc.build_df_final(data_normalized, driver, ws, selected_features)
    
    print('Bulding DF with all impostors')
    data_impostor, impostores = fc.build_impostors_df(data_normalized, impostor, ws, selected_features, driver)

    print('Doing data split')
    x_train, x_val = fc.split_data(data_final)

    for c in n_clusters:
      print('Create clusters')
      labels_train, centroid_train, x_train_class = fc.clusters_of_maneuvers(x_train, c)
      
      for m in method:
        if (m=='DBSCAN'):
          print('Training DBSCAN')
          dbscan_list, eps_list = fc.train_model_dbscan(labels_train, centroid_train, x_train_class, x_val)
          
          print('Doing predictions DBSCAN')
          result = fc.test_model_dbscan(dbscan_list, data_final, data_impostor, centroid_train, x_train_class)

          print('Evaluanting the results')
          acc, min_man, media_man, max_man, deviation = fc.evaluating_result(result, ws)
          print(acc)
          
          print('Doing Validation Model')
          validation = fc.test_model_dbscan(dbscan_list, data_final, x_val, centroid_train, x_train_class)
          
          print('Evaluation the validation')
          acc_val, min_man_val, media_man_val, max_man_val, deviation_val = fc.evaluating_result(validation, ws)

          row = [m, ws, driver, impostores, str({'eps':eps_list}), round(acc,2), min_man, media_man, max_man, deviation, round(100-acc_val,2), min_man_val, media_man_val, max_man_val, deviation_val, c]
          df = pd.DataFrame([row], columns=columns)
          results = results.append(df)
          results.to_csv('ResultsDBSCAN.csv')

        if (m=='LOF'):
          print('Training LOF')
          lof_list, nn_list = fc.new_train_model_lof(labels_train, centroid_train, x_train_class, x_val)
          
          print('Doing predictions LOF')
          result = fc.new_test_model_lof(lof_list, data_final, data_impostor, centroid_train, x_train_class)
          
          print('Evaluanting the results')
          acc, min_man, media_man, max_man, deviation = fc.evaluating_result(result, ws)
          print(acc)
          
          print('Doing Validation Model')
          validation = fc.new_test_model_lof(lof_list, data_final, x_val, centroid_train, x_train_class)
          
          print('Evaluation the validation')
          acc_val, min_man_val, media_man_val, max_man_val, deviation_val = fc.evaluating_result(validation, ws)

          row = [m, ws, driver, impostores, str({'n_neighbors':nn_list}), round(acc,2), min_man, media_man, max_man, deviation, round(100-acc_val,2), min_man_val, media_man_val, max_man_val, deviation_val, c ]
          df = pd.DataFrame([row], columns=columns)    
          results = results.append(df)
          results.to_csv('ResultsLOF4.csv')

        if (m=='EE'):
          print('Training Elliptical Envelope')
          elliptic_list = fc.train_model_elliptic(labels_train, centroid_train, x_train_class, x_val)
          
          print('Doing predictions Elliptic')
          result = fc.test_model_elliptic(elliptic_list, data_final, data_impostor, centroid_train)

          print('Evaluating the results')
          acc, min_man, media_man, max_man, deviation = fc.evaluating_result(result, ws)
          print(acc)

          print('Doing validation model')
          validation = fc.test_model_elliptic(elliptic_list, data_final, x_val, centroid_train)

          print('Evaluating the validation')
          acc_val, min_man_val, media_man_val, max_man_val, deviation_val = fc.evaluating_result(validation, ws)
          print(acc_val)

          row = [m, ws, driver, impostores, str({'contamination: 0'}), round(acc,2), min_man, media_man, max_man, deviation, round(100-acc_val,2), min_man_val, media_man_val, max_man_val, deviation_val, c ]
              
          df = pd.DataFrame([row], columns=columns)
          results = results.append(df)
          results.to_csv('ResultsEE.csv')

        if (m=='IF'):
          for ne in n_estimators:
            print('Training IF')
            if_list = fc.train_model_if(labels_train, centroid_train, x_train_class, x_val, ne)

            print('Doing predictions IF')
            result = fc.test_model_if(if_list, data_final, data_impostor, centroid_train)

            print('Evaluanting the results')
            acc, min_man, media_man, max_man, deviation = fc.evaluating_result(result, ws)
            print(acc)

            print('Doing Validation Model') 
            validation = fc.test_model_if(if_list, data_final, x_val, centroid_train)

            print('Evaluating the validation')
            acc_val, min_man_val, media_man_val, max_man_val, deviation_val = fc.evaluating_result(validation, ws)
            print(acc_val)

            row = [m, ws, driver, impostores, str({'n_estimators':ne}), round(acc,2),  min_man, media_man, max_man, deviation, round(100-acc_val,2), min_man_val, media_man_val, max_man_val, deviation_val, c  ]
            df = pd.DataFrame([row], columns=columns)    
            results = results.append(df)
            results.to_csv('ResultsIFJ.csv')
            
        if (m=='OCSVM'):
          for k in kernel:
            for n in nu:
              print('Training OCSVM')
              ocsvm_list = fc.train_model_ocsvm(labels_train, centroid_train, x_train_class, x_val, k, n)

              print('Doing predictions OCSVM')
              result = fc.test_model_ocsvm(ocsvm_list, data_final, data_impostor, centroid_train)

              print('Evaluanting the results')
              acc, min_man, media_man, max_man, deviation = fc.evaluating_result(result, ws)
              print(acc)

              print('Doing Validation Model') 
              validation = fc.test_model_ocsvm(ocsvm_list, data_final, x_val, centroid_train)

              print('Evaluating the validation')
              acc_val, min_man_val, media_man_val, max_man_val, deviation_val = fc.evaluating_result(validation, ws)
              print(acc_val)

              row = [m, ws, driver, impostores, str({'kernel':k, 'nu': n}), round(acc,2), min_man, media_man, max_man, deviation, round(100-acc_val,2), min_man_val, media_man_val, max_man_val, deviation_val, c ]
              
              df = pd.DataFrame([row], columns=columns)
              
              results = results.append(df)
              results.to_csv('ResultsOCSVM.csv')
