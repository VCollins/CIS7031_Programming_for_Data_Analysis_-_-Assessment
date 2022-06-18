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
#%matplotlib inline

print("Initialization Completed")

# TASK 1 - Data preprocessing

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

# TASK 2 - Data Analysis

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

# TASK 3 - Data manipulation in lieu of scatter plot creation

#Show worst and best performing sectors
temp_calc_df = calc_df.drop(['Industry_Totals'], axis = 1)

temp_calc_df.drop([10], inplace = True)

melt_df = pd.melt(temp_calc_df, id_vars=['Industry'])

if len(colors) <= 10:
    colors = colors * 10

# Create interactive, dynamic scatter plot
figure4 = px.scatter(melt_df, 
                     x = melt_df.variable, 
                     y = melt_df.value, 
                     animation_frame = melt_df.variable, 
                     animation_group = melt_df.Industry,
                     color = melt_df.Industry,
                     range_x = [2008,2019], 
                     range_y = [0,450000]
                )

figure4.update_xaxes(tickangle=90, tickfont=dict(color='black', size=14), tickmode='linear')
figure4.show()

# TASK 4 - Data manipulation in lieu of principal component analysis

#Data preprocessing
temp_calc_df = calc_df.drop(['Industry_Totals'], axis = 1)

x_range = temp_calc_df.loc[:, years].values

y_range = temp_calc_df.loc[:,['Industry']].values

x_range = StandardScaler().fit_transform(x_range)

pd.DataFrame(data = x_range, columns = years)

#Principal component analysis
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x_range)

principal_df = pd.DataFrame(data = principalComponents,
               columns = ['principal component 1', 'principal component 2'])

plot_df = pd.concat([principal_df, calc_df[['Industry']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

colors = ['#0000ff', '#ff0000', '#00ff00', '#dd00ff', '#ffaa00', '#00bbff', '#ff6666', '#ccee88', '#ffaadd', '#ffdd00']

for industry, color in zip(industry_names,colors):
    indicesToKeep = plot_df['Industry'] == industry
    ax.scatter(plot_df.loc[indicesToKeep, 'principal component 1'],
               plot_df.loc[indicesToKeep, 'principal component 2'],
               c = color,
               s = 50
              )
ax.legend(industry_names)
ax.grid()

#Heatmap generation
plt.figure(figsize=(20,15))

sns.heatmap(main_df.corr(), annot=True, fmt=".5f", linewidths=.5)

# TASK 5 - K-Means cluster code
#prepare data frame for cluster purposes
cluster_df = temp_calc_df.drop(['2009'], axis = 1)
cluster_df = cluster_df.drop([10], axis = 0)
new_cluster_df = cluster_df.drop(['Industry'], axis = 1)
cluster_df = cluster_df.drop(['2011', '2012', '2013', '2014', '2015', '2016', '2017'], axis = 1)
cluster_df = cluster_df.set_index("Industry")

#fig, ax = plt.subplots()
#cluster_df.plot('2010', '2018', kind='scatter', ax=ax)

#for k, v in cluster_df.iterrows():
#    ax.annotate(k, v)
    
#fig.canvas.draw()
arr = new_cluster_df.to_numpy()

plt.figure(figsize=(20,20))

kmeans2 = KMeans(n_clusters=2, max_iter=50, random_state=1)  
kmeans2.fit(arr)
plt.scatter(arr[:,0], arr[:,1], c=kmeans2.labels_, cmap='rainbow', s=200)
plt.rcParams.update({'font.size': 18})
x_value=arr[:,0]
y_value=arr[:,1]
for r in range(0, 10, 1):
    plt.annotate(industry_names[r], (x_value[r], y_value[r]))
    
plt.figure(figsize=(20,20))
plt.scatter(kmeans2.cluster_centers_[:,0], kmeans2.cluster_centers_[:,1], s=600, color='black')

plt.figure(figsize=(20,20))
kmeans3 = KMeans(n_clusters=3, max_iter=50, random_state=1)  
kmeans3.fit(arr)
plt.scatter(arr[:,0], arr[:,1], c=kmeans3.labels_, cmap='rainbow', s=200)
x_value=arr[:,0]
y_value=arr[:,1]
for r in range(0, 10, 1):
    plt.annotate(industry_names[r], (x_value[r], y_value[r]))

plt.figure(figsize=(20,20))
plt.scatter(kmeans3.cluster_centers_[:,0], kmeans3.cluster_centers_[:,1], s=600, color='black')

# TASK 6 - Hierarchy Map

hierarchy_data_df = temp_calc_df.drop(['Industry'], axis = 1)
hierarchy_data_df = hierarchy_data_df.drop([10], axis = 0)
hierarchy_df = hierarchy.linkage(hierarchy_data_df.values, 'single')
dn = hierarchy.dendrogram(hierarchy_df, labels=industry_names.to_list(), orientation='right')