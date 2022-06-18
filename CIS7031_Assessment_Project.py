#import relevant libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
%matplotlib inline

print("Initialization Completed")

#TASK 1 - Data processing

#Alter filepath as necessary
#This assumes the files are named in the following format: 'WalesStatsExport2009.csv'
path = #Set path as needed

#Create initial dataframe with column for data to merge into
main_df = pd.DataFrame(columns = ["Industry"])

#Read csv files in loop, and ensure the upper bound of the range includes the last file to be read
for v in range(2009, 2019, 1):
    temp_df = pd.read_csv(path + str(v) + '.csv', header=None)
    temp_df.columns = ["Industry",str(v)]
    main_df = main_df.merge(temp_df, on='Industry', how='right')
     
#instantiate values to be used in column iteration
current_mean = 0
current_min = 0
current_max = 0

#alter null values to mean value per column, taking care of outliers as well
for i in range(2009, 2019, 1):
    current_mean = main_df[(str(i))].mean()
    current_min = main_df[(str(i))].min()
    current_max = main_df[(str(i))].max()
    for j in range(0, 10, 1):
        current_value = main_df.loc[j, (str(i))]
        if current_value is None:
            current_value = current_mean
        if current_value < current_min:
            current_value = current_mean
        if current_value > current_max:
            current_value = current_mean
        main_df.loc[j, (str(i))]=current_value

#Rename industries as per assessment specification requirement
main_df.loc[0,'Industry'] = "Agriculture"
main_df.loc[3,'Industry'] = "Retail"
main_df.loc[4,'Industry'] = "ICT"
main_df.loc[5,'Industry'] = "Finance"
main_df.loc[6,'Industry'] = "Real_Estate"
main_df.loc[7,'Industry'] = "Professional_Service"
main_df.loc[8,'Industry'] = "Public_Administration"
main_df.loc[9,'Industry'] = "Other_Service"

#keep main_df as it is and copy the data to a new frame for calculation
calc_df = main_df.copy(deep = True)

#create new dataframe for industries data manipulation
ind_names_df = pd.DataFrame(main_df['Industry'])

industry_names = main_df['Industry']

main_df = main_df.set_index("Industry")

#TASK 2 - Data Analysis

#Initialise arrays for total values
total_values_per_ind = np.array([])

#Calculate totals per row and add as additional column
for r in range(0, 10, 1):
    #reset total to 0 for each industry
    current_ind_total = 0
    for c in range(1, 11, 1):  
        current_value = calc_df.iloc[r, c]
        current_ind_total = current_ind_total + current_value
    #add current industry total to array of totals
    total_values_per_ind = np.append(total_values_per_ind, current_ind_total)

calc_df['Industry_Totals'] = ""
    
#Add calculated array into dataframe as additional column
for q in range(0, 10, 1):
    calc_df.iloc[q, 11] = total_values_per_ind[q]

total_values_per_year = pd.DataFrame(columns = ["Industry"])

#Calculate each year's totals and add it to array
for t in range(2009, 2019, 1):
    temp_df = pd.DataFrame(columns=["Industry", str(t)])
    current_year_total = calc_df[(str(t))].sum()
    temp_df.loc[0, str(t)] = current_year_total
    total_values_per_year = total_values_per_year.merge(temp_df, on='Industry', how='right')

total_values_per_year.loc[0, "Industry"] = "Year_Totals"    

#append calculated row to frame
calc_df = calc_df.append(total_values_per_year, ignore_index = True, sort = False)

figure1 = go.Figure()

years = ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']

for x in range(0 ,10, 1):
    values_array = np.array([])
    for y in range(0, 10, 1):
        values_array = np.append(values_array, calc_df.iloc[x, (y + 1)])    
    figure1.add_trace(go.Bar(x=years, 
                    y=values_array,
                    name=str(calc_df.iloc[x,0])
                    ))
figure1.update_xaxes(tickangle=45, tickfont=dict(color='black', size=14),tickmode='linear')
figure1.update_layout(barmode='group', title_text='Wales Employment (2009 to 2018)')
figure1.show()

#TASK 2 - Data comparison

#calculate growth of industries per year
new_calc_df = calc_df.drop(['Industry'], axis = 1)

pct_chg_df = pd.DataFrame(new_calc_df.pct_change(axis = 1))

pct_chg_df.loc[: ,'Industry_Totals']= pct_chg_df.sum(axis=1)

pct_chg_df = pd.concat([pct_chg_df, ind_names_df], axis=1)

pct_chg_df.drop([10], inplace = True)

pct_chg_plot_df = pd.DataFrame(pct_chg_df['Industry_Totals'])

pct_chg_plot_df = pd.concat([pct_chg_plot_df, pct_chg_df['Industry']], axis=1)

pct_chg_values_array = np.array([])
pct_chg_names_array = np.array([])

for r in range(0, 10, 1):
    pct_chg_values_array = np.append(pct_chg_values_array, (pct_chg_plot_df.iloc[r,0] * 100))
    pct_chg_names_array = np.append(pct_chg_names_array, pct_chg_plot_df.iloc[r,1])

#Set color values for bars
colors = ['#0000ff', '#ff0000', '#00ff00', '#dd00ff', '#ffaa00', '#00bbff', '#ff6666', '#ccee88', '#ffaadd', '#ffdd00']
    
figure2 = go.Figure([go.Bar(x=pct_chg_names_array, 
                            y=pct_chg_values_array, 
                            marker_color=colors)
                    ])
figure2.update_xaxes(tickangle=45, tickfont=dict(color='black', size=14),tickmode='linear')
figure2.update_layout(barmode='group', title_text='Wales Employment Total Change Percentage(%) (2009 to 2018)')
figure2.show()

#Show annual totals per sector
year_totals_array = np.array([])

for r in range(1, 11, 1):
    year_totals_array = np.append(year_totals_array, (calc_df.iloc[10, r]))    
    
figure3 = go.Figure([go.Bar(x=years, 
                            y=year_totals_array, 
                            marker_color='#9999dd')
                    ])
figure3.update_xaxes(tickangle=45, tickfont=dict(color='black', size=14),tickmode='linear')
figure3.update_layout(barmode='group', title_text='Wales Employment Totals Per Year  (2009 to 2018)')
figure3.show()

