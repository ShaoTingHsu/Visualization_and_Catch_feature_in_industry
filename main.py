# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:47:54 2020

@author: TAO_DT_WIN10
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from datetime import datetime
import os
import math
from os.path import join
from scipy import stats
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix

# =============================================================================
# input function variable
# =============================================================================
machine_list = ['ADSP-C09']
#target_variable = ['SLURRY_F_PRESSURE_DIFFERENCE','CYCLE_PUMP_P201_RPM','SLURRY_TEMP_IN']
target_variable = ['UPPER_CHILLER_FLOW','CYCLE_PUMP_P201_RPM','SLURRY_HALF_LIFE','PLATE_COOLING_TEMP_IN','POLISHING_TIME_COUNTDOWN']
machine = machine_list[0]
dst_dir = r'C:/Users/TimHsu/Documents/Cloud/TAO_2/要交出去的Code/Variable control and valiate'
time_inf = '2020-08-05 00:00:00'
time_sup = '2020-08-05 23:59:00'#time_sup = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
# =============================================================================
# start function
# =============================================================================
def obs_variable(target_variable, machine_list,time_inf, time_sup, dst_dir):
    SLURRY_list = ['SLURRY_GROUP_ADSP_EQP_QTY',	'SLURRY_HALF_LIFE',	'SLURRY_LIFE',
               'SLURRY_IN_TEMP',	'SLURRY_F_IN_PRESSURE',	'SLURRY_F_OUT_PRESSURE',
               'SLURRY_F_PRESSURE_DIFFERENCE',	'SLURRY_TANK_A_LIQUID_LEVEL',
               'SLURRY_TANK_B_LIQUID_LEVEL',	'CYCLE_PUMP_P201_RPM',
               'OPERATION_SUPPLY_MODE',	'OPERATION_CLEAN_MODE',	'SLURRY_TYPE_AJ10',
               'SLURRY_TYPE_SK5','TANK_TYPE_A','TANK_TYPE_B',	'CLEAN_MODE_KOH',
               'CLEAN_MODE_DIW','CLEAN_MODE_SLURRY_REPLACE','SYSID']
    PLC_list = ['STEP_SETTIME',	'STEP_TIMMING',	'LOADING',	'UPPER_ROTATING_SPEED',
                'LOWER_ROTATING_SPEED',	'CARRIER_ROTATING_SPEED',	'CENTER_FLOW',	
                'GAP_VALUE',	'FLUTTERING',	'PLATE_COOLING_TEMP_IN',	
                'PLATE_COOLING_TEMP_OUT',	'SLURRY_TEMP_IN',	'SLURRY_TEMP_OUT',	
                'UPPER_TORQUE',	'LOWER_TORQUE',	'CARRIER_TORQUE',	'UPPER_PAD_LIFE',	
                'LOWER_PAD_LIFE',	'SLURRY_LIFE',	'POLISH_TIMES',	'DRESSER_CYCLE',	'DRESSER_SET',
                'BRUSH_CYCLE',	'BRUSH_SET',	'TO_SLURRY_SYSID',	'POLISH_RECIPE_NO',	'POLISH1_1',
                'POLISH1_2',	'POLISH1_3',	'POLISH1_4',	'DRESSER_RECIPE_NO',	'DRESSER1_1',
                'DRESSER1_2',	'DRESSER1_3',	'JET_RECIPE_NO',	'JET_1',	'JET_2',	'JET_3',
                'UPPER_CHILLER_FLOW',	'LOWER_CHILLER_FLOW',	'RE_POLISHING_SET_TIME',	
                'STEP_5_SET_TIME',	'POLISHING_TIME_COUNTDOWN',	'SUNGEAR_LIFE',	'OUTERGEAR_LIFE',
                'UPPER_PAD_LIFE_SETTING',	'LOWER_PAD_LIFE_SETTING',	'SUNGEAR_LIFE_SETTING',	
                'OUTERGEAR_LIFE_SETTING',	'GAP',	'UPPER_PLATE_POSITION',	'LOADER1_LOT_ID',
                'LOADER1_CASSETTE_ID',	'LOADER1_CASSETTE_RECIPEID',	'UNLOADER1_CASSETTE_ID',
                'UNLOADER1_CASSETTE_RECIPEID',	'LOADER2_LOT_ID',	'LOADER2_CASSETTE_ID',
                'LOADER2_CASSETTE_RECIPEID',	'UNLOADER2_CASSETTE_ID',	'UNLOADER2_CASSETTE_RECIPEID']
    for machine in machine_list:
        input_slurry = []
        input_plc = []
        for i in target_variable:
            if(i in SLURRY_list):
                input_slurry.append(i)
            elif(i in PLC_list):
                input_plc.append(i)
            else:
                print(i,'is not in candidate variable')
        n_split = 0
        allFileList = os.listdir(join(dst_dir,machine))
        for j in range(len(allFileList)):
            if('ADSP_X_' in allFileList[j]):
                n_split+=1
        Main_data = pd.read_csv(join(dst_dir,machine+'/ADSP_Main.csv'))
        Main_data['COLLECT_TIME'] = pd.to_datetime(Main_data['COLLECT_TIME'])
        Main_data = Main_data[['COLLECT_TIME', 'EQUIP_ID', 'LOT_ID', 'BATCH_ID']].iloc[np.where((Main_data['COLLECT_TIME']<=time_sup)&(Main_data['COLLECT_TIME']>=time_inf))]
        #PLC variable
        PLC_variable = pd.DataFrame(columns = ['COLLECTTIME','EQPID','CURRENT_LOTID','BATCH_COUNT','TO_SLURRY_SYSID']+input_plc)
        file_names = [join(dst_dir,machine+'/'+ 'ADSP_X_'+str(k)+'.csv') for k in range(1, n_split+1)]
        for names in file_names:
            d1 = pd.read_csv(names)
            s1 = np.asarray(np.where(~d1['CURRENT_LOTID'].isnull().values))[0]
            d1 = d1.iloc[s1]
            if(d1.shape[0]!=0):
                d1['COLLECTTIME'] = pd.to_datetime(d1['COLLECTTIME'])
                for LOT_BATCH_ID_ind in range(Main_data.shape[0]):
                    LOT_BATCH_ID = Main_data[['LOT_ID','BATCH_ID']].iloc[LOT_BATCH_ID_ind]
                    sub_PLC_variable = d1[['COLLECTTIME','EQPID','CURRENT_LOTID','BATCH_COUNT','TO_SLURRY_SYSID']+input_plc].iloc[np.where((d1['CURRENT_LOTID']==LOT_BATCH_ID[0])&(d1['BATCH_COUNT']==LOT_BATCH_ID[1]))]
                    if(sub_PLC_variable.shape[0]!=0):
                        PLC_variable = pd.merge(PLC_variable,sub_PLC_variable,how = 'outer')
        #SLURRY variable
        SLURRY_variable = pd.DataFrame(columns = ['SYSID','COLLECTTIME','EQPID']+input_slurry)
        file_names = [join(dst_dir,machine+'/'+ 'ADSP_SLURRY_'+str(k)+'.csv') for k in range(1, n_split+1)]
        for names in file_names:
            d1 = pd.read_csv(names)
            d1['COLLECTTIME'] = pd.to_datetime(d1['COLLECTTIME'])
            sub_SLURRY_variable = d1[['SYSID','COLLECTTIME','EQPID']+input_slurry].iloc[np.where(d1['SYSID'].isin(PLC_variable['TO_SLURRY_SYSID']))]
            if(sub_SLURRY_variable.shape[0]!=0):
                SLURRY_variable = pd.merge(SLURRY_variable,sub_SLURRY_variable,how = 'outer')
        # if((PLC_variable.shape[0]-SLURRY_variable.shape[0])/PLC_variable.shape[0] > 0.1):
        #     print('SLURRY跟PLC資料值差超過一成了!!!')
        #merge plc and slurry
        slurry_mix_x = pd.merge(PLC_variable,SLURRY_variable,left_on='TO_SLURRY_SYSID', right_on='SYSID', how='inner')
        slurry_mix_x.drop(slurry_mix_x.filter(regex='_y$').columns.tolist()+['SYSID'],axis=1, inplace=True)
        for variable_name in target_variable:
            # =============================================================================
            # 畫moving average的
            # =============================================================================
            #rolling window
            df = slurry_mix_x[['COLLECTTIME_x','CURRENT_LOTID','BATCH_COUNT']+[variable_name]]
            df['SMA'] = df.iloc[:,3].rolling(window=20).mean()
            
            index = [i for i in range(df.shape[0])]
            time = pd.to_datetime(df['COLLECTTIME_x'])
            dates = matplotlib.dates.date2num(time.tolist())
            # label
            n = 6
            sort_dict={}
            for i in range(len(dates)):
                sort_dict[sorted(dates)[i]]=i
            for i in range(len(dates)):
                dates[i]=sort_dict[dates[i]]
            label_num = [(i*(len(dates)-1)//5) for i in range(n)]
            label_time = [sorted(time.tolist())[i] for i in label_num ]
            label_time1 = [datetime.strftime(t1, '%m/%d %H') for t1 in label_time]
            
            og_batch_info = []
            og_batch_junction = []
            batch_info_slice = []
            for LOT_BATCH_ID_ind in range(Main_data.shape[0]):
                LOT_BATCH_ID = Main_data[['LOT_ID','BATCH_ID']].iloc[LOT_BATCH_ID_ind]
                batch_raw = df.iloc[np.where((df['CURRENT_LOTID']==LOT_BATCH_ID[0])&(df['BATCH_COUNT']==LOT_BATCH_ID[1]))[0],3]
                batch_len =  math.floor(batch_raw.shape[0]/100)
                sub_batch_info_slice = []
                for batch_ind in range(100):
                    sub_batch_info_slice = sub_batch_info_slice + [np.average(batch_raw[(batch_len*batch_ind):((batch_len*(batch_ind+1))-1)])]
                batch_info_slice.append(sub_batch_info_slice)
                og_batch_junction.append(np.max(np.where((df['CURRENT_LOTID']==LOT_BATCH_ID[0])&(df['BATCH_COUNT']==LOT_BATCH_ID[1]))[0]))
                og_batch_info.append(np.average(batch_raw))
            batch_info_slice = pd.DataFrame(batch_info_slice)
            plot_batch_junction = [0]+og_batch_junction
            plot_batch_junction = np.repeat(plot_batch_junction,[2])
            plot_batch_info = np.repeat(og_batch_info,[2]).tolist()
            plot_batch_info = [0]+plot_batch_info+[0]
            plot_batch_junction = np.array(plot_batch_junction)
            plot_batch_info = np.array(plot_batch_info)
            # =============================================================================
            #使用DBSCAN抓outliers
            # =============================================================================
            #定義距離矩陣
            distance_M = distance_matrix(batch_info_slice, batch_info_slice)
            distance_M = np.tril(distance_M)
            clustering_eps = np.quantile(distance_M[distance_M>0],.3)
            model_clustering = DBSCAN(eps=clustering_eps, min_samples=5)
            outliers_ind_batch_slice = np.where(model_clustering.fit(batch_info_slice).labels_==-1)[0]
            # =============================================================================
            # 觀察極端值比例
            # =============================================================================
            ind = og_batch_junction[-(1+round(Main_data.shape[0]/5))]
            data_old = df[variable_name].iloc[:ind].dropna().sort_values()
            data_latest = df[variable_name].iloc[(ind+1):].dropna()
            num_data_latest_outlier = sum(data_latest<data_old.iloc[round(data_old.shape[0]*.05)])+sum(data_latest>data_old.iloc[round(data_old.shape[0]*.95)])
            print(num_data_latest_outlier/data_latest.shape[0])
            if(num_data_latest_outlier/data_latest.shape[0]>.15):
                print('Too many outliers')
            #把outliers的點標出來
            outliers_ind = np.where((data_latest<=data_old.iloc[round(data_old.shape[0]*.05)])|(data_latest>=data_old.iloc[round(data_old.shape[0]*.95)]))[0]+ind+1
            
            # =============================================================================
            # 畫圖囉
            # =============================================================================
            plt.rcParams['figure.figsize'] = (10.0, 4.0)
            fig, ax = plt.subplots()
            plt.plot(index, df[variable_name], label='Original',alpha = .2,linewidth = 1)
            plt.plot(index, df['SMA'], label='SMA',alpha = 1)
            plt.scatter(outliers_ind,df[variable_name].iloc[outliers_ind],c='r',edgecolors='none',s=3,label = 'Outliers')
            for plot_ind in range(len(og_batch_junction)):
                if(plot_ind in outliers_ind_batch_slice):
                    color_name = 'red'
                else:
                    color_name = 'yellow'
                if(plot_ind == 0):
                    plt.fill_between([0,og_batch_junction[plot_ind]], [og_batch_info[plot_ind]]*2, 0,alpha = 0.3,color='yellow')
                else:
                    plt.fill_between([og_batch_junction[(plot_ind-1)]+1,og_batch_junction[plot_ind]], [og_batch_info[plot_ind]]*2, 0,alpha = 0.3,color=color_name)
            plt.axhline(data_old.iloc[round(data_old.shape[0]*.05)],alpha = .5, xmin=index[0], xmax=index[-1],c='black',linestyle='--')
            plt.axhline(data_old.iloc[round(data_old.shape[0]*.95)],alpha = .5, xmin=index[0], xmax=index[-1],c='black',linestyle='--')
            plt.xticks(label_num, label_time1,rotation=60)
            plt.xlim((0,np.max(index)))
            plt.ylim((np.min(df[variable_name]),np.max(df[variable_name])))
            for k in range(len(og_batch_junction)):
                xc = og_batch_junction[k]
                if(xc==ind):
                    plt.axvline(x=xc,ymin=0, ymax=1,c='blue',linestyle='-',alpha = 0.8)
                else:
                    y_max = (np.max(og_batch_info[k:(k+2)])-np.min(df[variable_name]))/(np.max(df.iloc[:,3])-np.min(df[variable_name]))
                    plt.axvline(x=xc,ymin=0, ymax=y_max,c='coral',linestyle='--',alpha = 0.8)
            plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0,frameon=False,scatterpoints = 1)
            plt.title(variable_name)
            plt.tight_layout()
            plt.show()
            fig.savefig(join(dst_dir,variable_name+'_'+machine+'.png'), bbox_inches='tight')
            # =============================================================================
            # 畫MovingRange
            # =============================================================================
            Cons_diff = df[variable_name].diff()  
            Cons_diff = abs(Cons_diff)   
            
            #X_men = df[variable_name].tail(df[variable_name].shape[0]-ind).mean(axis=0)
            R_men = Cons_diff.tail(df[variable_name].shape[0]-ind).mean(axis=0)
            DDT_UCLr = 3.27 * R_men
            
            fig, ax = plt.subplots()
            plt.plot(index, [DDT_UCLr]*len(index),label = 'Upper Confidence Line',color='brown', alpha=0.8)
            plt.plot(index, Cons_diff,label='Orginal line', color='blue', alpha=0.2)
            plt.xticks(label_num, label_time1,rotation=60)
            plt.axvline(x=ind,ymin=0, ymax=1,c='blue',linestyle='-',alpha = 0.8)
            plt.xlim((0,np.max(index)))
            plt.title('Moving Range '+variable_name)
            plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0,frameon=False,scatterpoints = 1)
            plt.tight_layout()
            plt.show()
            fig.savefig(join(dst_dir,variable_name+'_'+machine+'Moving Range'+'.png'), bbox_inches='tight')
            # =============================================================================
            # 觀察極端值比例
            # =============================================================================
            ind = og_batch_junction[-(1+round(Main_data.shape[0]/5))]
            mr_old = Cons_diff.tolist()[:ind]
            mr_latest = Cons_diff.tolist()[(ind+1):]
            num_mr_old_outlier = sum(np.array(mr_old)>=DDT_UCLr)/len(mr_old)
            num_mr_latest_outlier = sum(np.array(mr_latest)>=DDT_UCLr)/len(mr_latest)
            print(num_mr_old_outlier,num_mr_latest_outlier)
#        if(abs((num_mr_latest_outlier-num_mr_old_outlier)/num_mr_old_outlier)>.5):
#            print('MR分配變化過大')
obs_variable(target_variable, machine_list,time_inf, time_sup, dst_dir)
