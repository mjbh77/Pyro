# Tuto used : https://docs.streamlit.io/library/get-started/create-an-app

# Modules
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff




st.set_page_config(layout="wide")
st.title('Wildfires in USA - Analysis from 1992 to 2018')
data_filename = 'wildfires_final_v2.csv'



# ------------   Labels used in the streamlit :
fires_number = 'Number of wildfires'
fires_causes = 'Causes of wildfires'
fires_surf = 'Surface of wildfires'
fires_dur = 'Duration of wildfires'
fires_temp = 'Temporal Data'
fires_state = 'States'

months_labels = ['Jan', 'Feb','Mar', 'Apr','May','Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
days_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
seasons_labels = ['Winter', 'Spring', 'Summer', 'Fall']
causes_labels = ['Individuals\' mistake', 'Criminal', 'Infrastructure accident', 'Natural (lightning)', 'Other/Unknown']
causes_labels_split = ['Individuals\' \nmistake', 'Criminal', 'Infrastructure \naccident', 'Natural \n(lightning)', 'Other/\nUnknown']
regions_list = ['East', 'West', 'North', 'South', 'Center',
     'North-East', 'North-West', 'South-East', 'South-West', 'Tropical']
regions_list_split = ['East', 'West', 'North', 'South', 'Center',
     'North-\nEast', 'North-\nWest', 'South\n-East', 'South-\nWest', 'Tropical']

dico_regions = {
'AL': 'South-East', 'AK': 'North', 'AZ': 'South-West', 'AR': 'Center', 'CA': 'South-West',
'CO': 'Center','CT': 'North-East','DE': 'North-East','DC': 'North-East','FL': 'South-East',
'GA': 'South-East','HI': 'Tropical','ID': 'North-West','IL': 'Center','IN': 'North-East',
'IA': 'Center','KS': 'Center','KY': 'East','LA': 'South-East','ME': 'North-East',
'MD': 'North-East','MA': 'North-East','MI': 'North-East','MN': 'North','MS': 'South-East',
'MO': 'Center','MT': 'North-West','NE': 'Center','NV': 'West','NH': 'North-East',
'NJ': 'North-East','NM': 'South','NY': 'North-East','NC': 'East','ND': 'North','OH': 'North-East',
'OK': 'South','OR': 'North-West','PA': 'North-East','PR': 'Tropical','RI': 'North-East',
'SC': 'East','SD': 'North','TN': 'East','TX': 'South','UT': 'West','VT': 'North-East','VA': 'East',
'WA': 'North-West','WV': 'North-East','WI': 'North','WY': 'North-West'}
df_regions = pd.DataFrame(dico_regions.items(), columns=['State', 'Region'])

# ------------ Colors
color_fire = 'firebrick'
categories_palette = ['#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026',
    '#800026', "darkorange", "gold", "goldenrod", "lemonchiffon", "cornsilk"]
month_colors = ['#80cdc1', '#80cdc1', '#018571', '#018571', '#018571', '#dfc27d',
              '#dfc27d', '#dfc27d', '#a6611a', '#a6611a', '#a6611a', '#80cdc1']
days_colors = ['firebrick', 'firebrick','firebrick','firebrick','firebrick','indianred','indianred']
causes_color = ['#332288', '#44AA99', '#AA3377', '#CCBB44', 'grey']
seasons_colors = ['#80cdc1', '#018571', '#dfc27d', '#a6611a']
dico_causes = dict([(key, str(value) ) for key, value in zip(causes_labels, range(6))])
dico_causes_colors = dict([(key, value) for key, value in zip(causes_labels, causes_color)])
regions_colors = ['#ffdf87', '#7a3600', '#d39833', '#fbc591', '#efe89f',
'#ffd951', '#ffa44d', '#ffbb5c', '#bf6108', '#fd6628']
dico_regions_order = dict([(key, str(value) ) for key, value in zip(regions_list, range(10))])
dico_regions_colors = dict([(key, value) for key, value in zip(regions_list, regions_colors)])

# -------------- Global plot parameters
sns.set(rc={'axes.facecolor':(235/255, 230/255, 188/255, 1),
            'figure.facecolor':(235/255, 230/255, 188/255, 1)})
plt.style.use('default')
plt.rcParams.update({'font.size': 10})


# ------------------------------------------
# ---------------------------- Fonctions
# ------------------------------------------
@st.cache
def load_data():
    data = pd.read_csv(data_filename)
    data['DISCOVERY_DATE'] = pd.to_datetime(data['DISCOVERY_DATE'])
    data['DISC_DOY'] = data['DISCOVERY_DATE'].dt.dayofyear
    data['Region'] = [dico_regions[x] for x in data.STATE]
    data.rename(columns = {'LATITUDE':'lat', 'LONGITUDE':'lon'}, inplace = True)
    data.drop(['COUNTY', 'OWNER_DESCR', 'NWCG_CAUSE_AGE_CATEGORY',
        'NWCG_REPORTING_AGENCY'], axis = 1, inplace = True)
    return data
# ------------------------------------ Countplots
def make_countplot(data, x, title = '',
    xtitle ='', ytitle = 'Number of \nevents',
    x_rot = 0, rm_legend = False,
    color_plot = None, palette = None,
    order = None, xlabels = None,
    hue = None, hue_order = None,
    edgecolor = 'black', linewidth = 0.8,
    width = 8, height = 2.5):
    fig, ax = plt.subplots(figsize = (width, height))
    sns.countplot(x = x, data = data,
            hue = hue, hue_order = hue_order, order = order,
            color = color_plot, palette = palette,
            edgecolor = edgecolor, linewidth = linewidth,
            ax = ax)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title, y = 1.1)
    ax.tick_params(axis='x', labelrotation= x_rot)
    if xlabels :
        ax.set_xticklabels(labels = xlabels, rotation = x_rot, ha = 'center')
    if rm_legend :
        ax.get_legend().remove()
    else :
        plt.legend(ncol=2, title = '', fontsize = 9, edgecolor = 'white')
    return(fig)


def make_countplot_with_annot(data, x, title = '',
    xtitle ='', ytitle = 'Number of \nevents',
    x_rot = 0, xlabels = None,
    color_plot = None, palette = None,
    order = None,
    edgecolor = 'black', linewidth = 0.8,
    width = 8, height = 2.5, rm_legend = False):
# Work only for dataframe with CAUSE
    fig, ax = plt.subplots(figsize = (width, height))
    sns.countplot(x = x, data = data,
            order = order,
            color = color_plot, palette = palette,
            edgecolor = edgecolor, linewidth = linewidth,
            ax = ax)
    if xlabels :
        ax.set_xticklabels(labels = xlabels, rotation = x_rot, ha = 'center')
    ax.set_xlabel('')
    ax.set_ylim(0, max(data.CAUSE.value_counts()) + max(data.CAUSE.value_counts())*0.3)
    ax.set_ylabel(ytitle)
    ax.set_title(title, y = 1.1);
    for i in range(len(order)) :
        ax.annotate(str(round((data[x] == order[i]).sum() *100/data.shape[0], 1) ) + '%',
            xy = (i, (data[x] == order[i]).sum() + max(data.CAUSE.value_counts())*0.1),
            ha = 'center' )
    if rm_legend :
        ax.get_legend().remove()
    return(fig)

# ------------------------------------ Boxplot
def make_boxplot(data, x, y, title = '',
    xtitle ='', ytitle = 'Number of \nevents',
    x_rot = 0, xlabels = None,
    color_plot = None, palette = None,
    hue = None, hue_order = None,
    width = 8, height = 2.5):
    fig, ax = plt.subplots(figsize = (width, height))
    sns.boxplot(x = x, y = y,
            data = data,
            hue = hue, hue_order = hue_order,
            color = color_plot, palette = palette,
            ax = ax)
    if xlabels :
        ax.set_xticklabels(xlabels)
    ax.tick_params(axis='x', labelrotation= x_rot)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    plt.title(title, y = 1.1)
    return(fig)

# ------------------------------------ Barplot
def make_barplot(data, x, y, title = '',
    xtitle = '', x_rot = 0, order = None,
    hue = None, hue_order = None,
    xlabels = None, ytitle = '',
    palette = None, color_plot = None,
    edgecolor = 'black', linewidth = 0.8,
    errcolor='.26', errwidth=None,
    width = 8, height = 2.5, rm_legend = False, ncol = 3):
    fig, ax = plt.subplots(figsize = (8, 2.5))
    sns.barplot( x = x, y = y,
        data = data, order = order,
        hue = hue, hue_order = hue_order,
        color = color_plot, palette = palette,
        edgecolor = edgecolor, linewidth = linewidth,
        errcolor = errcolor, errwidth = errwidth,
        ax = ax)
    if xlabels :
        ax.set_xticklabels(xlabels)
    if rm_legend :
        ax.get_legend().remove()
    else :
        plt.legend(ncol=ncol, bbox_to_anchor=(1, -0.2))
    ax.tick_params(axis='x', labelrotation= x_rot)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    plt.title(title, y = 1.1)
    return(fig)

def ridgeplot(data, title = 'Distribution of wildfires \nalong a year') :
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(data.sort_values(by = 'DISC_YEAR'),
                      row = 'DISC_YEAR', aspect=10, height=0.4)
    g.map(sns.kdeplot, 'DISC_DOY',
          bw_adjust=1, clip_on=True, color = color_fire,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, 'DISC_DOY',
          bw_adjust=1, clip_on=False,
          color="w", lw=3) # White contour
    g.map(plt.axhline, y=0, color = color_fire,
          lw=2, clip_on=False)
    g.fig.subplots_adjust(hspace=-0.5)
    for i, ax in enumerate(g.axes.flat):
        ax.text(-60, 0.0005,
                data.sort_values(by = 'DISC_YEAR').DISC_YEAR.unique()[i],
                fontweight='bold', fontsize=15,
                color= 'grey')
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    axes = g.axes.flatten()
    for ax in axes:
        ax.set_ylabel("")
    plt.xlim(-5, 365)
    plt.xticks(ticks = [0, 181, 360], labels=['Jan', 'Jun','Dec'])
    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.xlabel('Month', fontsize=15)
    plt.figure(figsize=(10, 3))
    g.fig.suptitle(title,
                   ha='center',
                   y=1.05,
                   fontsize=16)
    return(g)

# ------------------------------------ Lineplot
def make_lineplot(data, x, y, title = '',
    xtitle ='', ytitle = 'Number of \nevents',
    x_rot = 0, xlabels = None,
    color_plot = None, palette = None,
    marker = 'o', hue = None,
    width = 8, height = 2.5):
    fig, ax = plt.subplots(figsize = (width, height))
    sns.lineplot(x = x, y = y,
            data = data, hue = hue, marker = marker,
            color = color_plot, palette = palette,
            ax = ax)
    if xlabels :
        ax.set_xticklabels(xlabels)
    ax.tick_params(axis='x', labelrotation= x_rot)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    plt.legend(ncol=3, bbox_to_anchor=(1, -0.2))
    plt.title(title, y = 1.1)
    return(fig)


# ------------------------------------------
#---------------------------- Dataframe Import
# ------------------------------------------
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
df_fires = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")

# ------------------------------------------
# -------------------------- Create Sub-datframe used for plots later
# ------------------------------------------
# --- For global analysis
fires_months_tmp_df = df_fires.groupby( [ 'DISC_YEAR', 'DISC_MONTH' ],
    as_index=False).agg({'STATE' : 'count'})
fires_days_tmp_df = df_fires.groupby( ['DISC_DOW', 'DISC_YEAR'],
    as_index=False).agg({ 'STATE' : 'count' })

surface_fires_tmp = pd.DataFrame(df_fires.groupby('DISC_YEAR', as_index=False).agg({'FIRE_SIZE': 'mean'}))
surface = df_fires.groupby('DISC_YEAR', as_index=False)['FIRE_SIZE'].sum()
surface_total_state = df_fires.groupby(['DISC_YEAR', 'STATE', 'STATE_FULL'], as_index=False)['FIRE_SIZE'].sum()
surface_total_state.columns = ['year', 'St', 'State', 'Total burned area (ha)']
surface_avg_state = df_fires.groupby(['DISC_YEAR', 'STATE', 'STATE_FULL'],
                        as_index=False)['FIRE_SIZE'].mean().groupby(['STATE', 'STATE_FULL'],
                            as_index = False)['FIRE_SIZE'].mean()
surface_avg_state.columns = ['St', 'State', 'Avg burned area (ha)']


surface_avg = df_fires.groupby(['DISC_YEAR','CAUSE'], as_index=False)['FIRE_SIZE'].mean()
surface_avg_year = df_fires.groupby(['DISC_YEAR','CAUSE'],as_index=False).agg({'FIRE_SIZE': 'mean'})

months_cause = df_fires.groupby(['DISC_MONTH','CAUSE'],as_index=False).agg({'FIRE_SIZE':'mean'})
months_cause_total = df_fires.groupby(['DISC_MONTH', 'CAUSE'],
                                 as_index=False).agg({'FIRE_SIZE':'sum'})
months_year_total = df_fires.groupby(['DISC_MONTH', 'DISC_YEAR'],
                                 as_index=False).agg({'FIRE_SIZE':'sum'})
day_size = df_fires.groupby('DISC_DOW', as_index=False).agg({'FIRE_SIZE':'mean'})
day_size_cause = df_fires.groupby(['DISC_DOW','CAUSE'], as_index=False).agg({'FIRE_SIZE':'mean'}).set_index('DISC_DOW')
weekday_size_cause = pd.pivot_table(data=day_size_cause,
                                  index=day_size_cause.index, columns='CAUSE',
                                  values='FIRE_SIZE', aggfunc='mean')

cause_month_year = df_fires.groupby(['DISC_YEAR', 'DISC_MONTH', 'CAUSE', 'NWCG_GENERAL_CAUSE'], as_index = False).agg({'STATE' : 'count'})

ct_classe_cause = pd.crosstab(df_fires.FIRE_SIZE_CLASS, df_fires.CAUSE)
ct_classe_cause_perc = ct_classe_cause.apply(lambda x : (x/x.sum()) *100, axis  = 1)
ct_classe_cause_perc = ct_classe_cause_perc[causes_labels]


# --- For state analysis
state_year_tmp_df = df_fires.groupby(['STATE', 'STATE_FULL', 'DISC_YEAR'],
    as_index=False).agg({ 'FPA_ID' : 'count', 'lat' : 'mean', 'lon' : 'mean' })
state_year_tmp_df.columns = ['St', 'State', 'Year', 'Number of fires', 'lat', 'lon']
state_year_avg_df = state_year_tmp_df.groupby(['State', 'St'],
    as_index=False).agg({'Number of fires' : 'mean'})
# --- For region anlysis
region_cause_df = pd.crosstab(df_fires['Region'], df_fires['CAUSE'])
region_cause_df['total']=region_cause_df.sum(axis=1)
for col in region_cause_df.columns:
    region_cause_df[col]=region_cause_df[col]/region_cause_df['total']*100
region_cause_df.drop('total', axis=1, inplace=True)
region_fire_number = pd.crosstab(df_fires['DISC_YEAR'], df_fires['Region'])
saison_region_df = df_fires.groupby(['Region', 'DISC_YEAR', 'Season'], as_index=False).agg({'FPA_ID' : 'count'})

# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
#---------------------------- Intro text

st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, \
  sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
  Ut enim ad minim veniam, quis nostrud exercitation ullamco \
  laboris nisi ut aliquip ex ea commodo consequat. \
  Duis aute irure dolor in reprehenderit in voluptate \
  velit esse cillum dolore eu fugiat nulla pariatur. \
  Excepteur sint occaecat cupidatat non proident, \
  sunt in culpa qui officia deserunt mollit \
  anim id est laborum.")
st.write("Github : [https://github.com/DataScientest-Studio/Pyromaniacs](https://github.com/DataScientest-Studio/Pyromaniacs)")

#---------------------------- Check table is correctly loaded
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df_fires.head())

genre = st.radio(
     "What kind of analysis to you want to perform ?",
     ('Global', 'Regional', 'By State'))

# ---------------------------------------------
# ---------------------------- Plots on full US
#----------------------------------------------
if genre == 'Global':
    # # -------------- What will be the variables to plot ?
    variables_columns = st.columns((5, 1, 1, 1 ,1 , 1))
    option_main_variable = variables_columns[0].selectbox(
             'Which variable to plot ?',
             (fires_number, fires_causes, fires_surf, fires_dur))
    # # Dummy blank text to align checkboxes to selectbox
    variables_columns[1].markdown(f'<p style="color:#ffffff;font-size:14px;">{"text”"}</p>',
                unsafe_allow_html=True)


    if (option_main_variable == fires_surf) | (option_main_variable == fires_dur) :
        check_cause = variables_columns[1].checkbox('Causes')


    # -------------- Organize layout where plot will be
    # Space out the maps so the first one is 1.25x the size of the other one
    #----------------------------
    #----------------------------  Analysis of fire numbers
    if option_main_variable == fires_number :
        c1, c2 = st.columns((1.25, 1))
        with c1 :
            st.header('Overview of the number of wildfires from 1992 to 2018 -')
            map_type_fires = st.radio(
     "Map type :", ('Year by year', 'Average by year'))
            if map_type_fires == 'Year by year' :
                fig = px.choropleth(
                    state_year_tmp_df,
                    locations='St',
                    color='Number of fires',
                    locationmode='USA-states',
                    color_continuous_scale='Reds',
                    range_color = [1, 15000],
                    animation_frame = 'Year',
                    hover_name = 'State',
                    hover_data = {'St' : False, 'Year' : False}
                )
                fig.add_trace(go.Scattergeo(
                    locationmode = 'USA-states',
                    locations=state_year_avg_df['St'],    ###codes for states,
                    text=state_year_avg_df['St'],
                    hoverinfo = 'skip',
                    mode = 'text' )  )
                fig.update_layout(
                    title_text='Number of fires per state per year',
                    geo = dict(
                        scope='usa',
                        projection=go.layout.geo.Projection(type = 'albers usa'),
                        showlakes=True, # lakes
                        lakecolor='rgb(255, 255, 255)'),
                        margin=dict(
                            l=0, r=0, b=0, t=30, pad=2  )
                )
                st.plotly_chart(fig, use_container_width=True)
            else :
                fig2 = px.choropleth(
                    state_year_avg_df,
                    locations='St',
                    color='Number of fires',
                    locationmode='USA-states',
                    color_continuous_scale='Reds',
                    range_color = [1, 10000],
                    hover_name = 'State',
                    hover_data = {'St' : False}
                )
                fig2.add_trace(go.Scattergeo(
                    locationmode = 'USA-states',
                    locations=state_year_avg_df['St'],    ###codes for states,
                    text=state_year_avg_df['St'],
                    hoverinfo = 'skip',
                    mode = 'text' )  )
                fig2.update_layout(
                    title_text='Average number of fires per state per year',
                    geo = dict(
                        scope='usa',
                        projection=go.layout.geo.Projection(type = 'albers usa'),
                        showlakes=True, # lakes
                        lakecolor='rgb(255, 255, 255)'),
                    margin=dict(l=0, r=0, b=0, t=30, pad=2)

                )
                st.plotly_chart(fig2, use_container_width=True)
        with c2 :
            fig = make_countplot(df_fires, 'DISC_YEAR',
                ytitle = 'Number of fires', title = 'Number of fires per year',
                xtitle ='', x_rot = 90, color_plot = color_fire)
            st.pyplot(fig, use_container_width=True) # SHOW THE FIGURE
            st.write("On the global US territory, \
                the number of fires didn't significately increase since 1992.\
                However, some years have been more affected than other \
                (2006 and 2011 for example).")
            fig = make_boxplot(fires_months_tmp_df, 'DISC_MONTH', 'STATE',
                title = "Average number of fires per month",
                xtitle = '', x_rot = 0, xlabels = months_labels,
                ytitle = 'Number of fires \nper year', palette = month_colors)
            st.pyplot(fig, use_container_width=True) # SHOW THE FIGURE
            st.write("Wildfires are particularly abundant in the beginning of \
                spring and summer.")
            fig = make_boxplot(fires_days_tmp_df, 'DISC_DOW', 'STATE',
                title = "Number of fires per year along the week",
                xtitle = '', x_rot = 0, xlabels = days_labels,
                ytitle = 'Number of fires \nper year', palette = days_colors)
            st.pyplot(fig, use_container_width=True) # SHOW THE FIGURE
            st.write("Saturday is the day where wildfires occurs the most.")
    if option_main_variable == fires_surf :
        c1, c2 = st.columns((1.25, 1))
        with c1 :
            map_type_fires = st.radio(
     "Map type :", ('Year by year', 'Average by year'))
            if map_type_fires == 'Year by year' :
                # df_subset = state_year_tmp_df[state_year_tmp_df.DISC_YEAR == year_plot]
                fig = px.choropleth(
                    surface_total_state,
                    locations='St',
                    color='Total burned area (ha)',
                    locationmode='USA-states',
                    color_continuous_scale='YlOrBr',
                    range_color = [1, 400000],
                    animation_frame = 'year',
                    hover_name = 'State',
                    hover_data = {'St' : False, 'year' : False}
                )
                fig.add_trace(go.Scattergeo(
                    locationmode = 'USA-states',
                    locations=surface_total_state['St'],    ###codes for states,
                    text=surface_total_state['St'],
                    hoverinfo = 'skip',
                    mode = 'text' )  )
                fig.update_layout(
                    title_text='Number of fires per state per year',
                    geo = dict(
                        scope='usa',
                        projection=go.layout.geo.Projection(type = 'albers usa'),
                        showlakes=True, # lakes
                        lakecolor='rgb(255, 255, 255)'),
                        margin=dict(
                            l=0, r=0, b=0, t=30, pad=2  )
                )
                st.plotly_chart(fig, use_container_width=True)
            else :
                fig2 = px.choropleth(
                    surface_avg_state,
                    locations='St',
                    color='Avg burned area (ha)',
                    locationmode='USA-states',
                    color_continuous_scale='YlOrBr',
                    range_color = [1, 250],
                    hover_name = 'State',
                    hover_data = {'St' : False}
                )
                fig2.add_trace(go.Scattergeo(
                    locationmode = 'USA-states',
                    locations=surface_avg_state['St'],    ###codes for states,
                    text=surface_avg_state['St'],
                    hoverinfo = 'skip',
                    mode = 'text' )  )
                fig2.update_layout(
                    title_text='Average surface burned per state per year',
                    geo = dict(
                        scope='usa',
                        projection=go.layout.geo.Projection(type = 'albers usa'),
                        showlakes=True, # lakes
                        lakecolor='rgb(255, 255, 255)'),
                    margin=dict(l=0, r=0, b=0, t=30, pad=2)

                )
                st.plotly_chart(fig2, use_container_width=True)
            fig = make_countplot(x='FIRE_SIZE_CLASS', data=df_fires,
                ytitle = 'Number of fires', xtitle = 'Fire class',
                title = 'Count of the fires based on their size category, 1992-2018',
                order=['A','B','C','D','E','F','G'],
                palette = categories_palette)
            st.pyplot(fig, use_container_width=True)
            st.write("L'administration fédérale américaine classe \
                les feux selon leur taille dans une nomenclature en 7 lettres : \
                \nA - environ moins de 1000 m2 ; \nB - entre 1000 m2 et 4 ha environ ;\
                 \nC - de 4 à 40 ha environ ; \nD - de 40 à 120 ha environ ; \nE - \
                 de 120 à 400 ha environ ; \nF - de 400 à 2000 ha environ ;\
                  \nG - plus de 2000 ha.")
            st.write("On voit nettement que la grande majorité des feux sont de \
                relative petite taille (moins de 4 ha). Les très grands feux sont dans \
                l'ensemble très peu nombreux sur la période d'observation.")
            fig, ax = plt.subplots(figsize = (8, 2.5))
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_fires.groupby('DISC_YEAR',
                as_index=False).agg({'FIRE_SIZE': 'mean'}).DISC_YEAR,
                df_fires.groupby('DISC_YEAR', as_index=False).agg({'FIRE_SIZE': 'mean'}).FIRE_SIZE)
            line = slope*surface_fires_tmp.DISC_YEAR+intercept
            ax.plot(surface_fires_tmp['DISC_YEAR'] , surface_fires_tmp['FIRE_SIZE'],
                     c = color_fire, marker = 'o')
            plt.plot(surface_fires_tmp.DISC_YEAR, line, color = 'grey', linestyle = 'dotted', lw = 3,
                label='y = {:.2f}x{:.2f}'.format(slope,intercept))
            plt.ylabel('Average surface (ha)')
            plt.title('Average surface burned \nper fire (hectares)', y = 1.1);
            st.pyplot(fig, use_container_width=True)
            st.write('The average area of a fire increases progressively throughout \
                the period despite significant annual variations; the regression line (grey) \
                confirms this trend.')

        with c2:
            if check_cause :
                fig, ax = plt.subplots(figsize = (8, 3))
                ct_classe_cause_perc.plot.bar(stacked = False, ax = ax, color = causes_color,
                                              edgecolor  = 'black', alpha = 0.7)
                ax.legend(title = '', loc = 'upper left', ncol = 3, fontsize=9)
                ax.set_ylim(0, 100)
                ax.set_xlabel('')
                ax.set_ylabel('Percentage of fires per category', fontsize = 15)
                ax.tick_params(axis='both', color = 'black')
                ax.tick_params(axis='x', labelrotation= 0, length = 6, color = 'black')
                plt.title('Causes of fire for each fire size category', y = 1.05, fontsize = 16);
                st.pyplot(fig)
                st.write('Lightning : major part of category F G')
                fig = make_lineplot(x = 'DISC_YEAR', y = 'FIRE_SIZE',
                    data = surface_avg, hue = 'CAUSE',
                    palette = causes_color,
                    ytitle = 'Avg damaged surface \nper fire (ha)',
                    title = 'Change in the average damage surface per year, 1992-2018')
                st.pyplot(fig, use_container_width=True)
                st.write('Lightning cause the most extensive fires throughout the study \
                    period, and this has been increasing. Next come technical accidents \
                    on infrastructures (which concern sparks from braking or mechanical \
                    failure, or from work on roads or railways).')
                fig = make_barplot(df_fires,'DISC_MONTH','FIRE_SIZE',
                    hue = 'CAUSE', hue_order = causes_labels, xlabels = months_labels,
                    ytitle = 'Average surface damaged',
                    title = 'Avg damaged surface of fires depending on \nthe cause and the month',
                    palette = causes_color)
                st.pyplot(fig, use_container_width=True)
                st.write('Different causes cause fires of unequal size depending on the \
                    month of the year considered. For example, spring and summer storms \
                    cause much larger fires. We also observe that technical accidents on \
                    infrastructures cause larger fires in the fall.')
                fig = weekday_size_cause.plot(kind='barh', stacked=True, color = dico_causes_colors)
                plt.title('Average surface burnt by day \naccording to the cause of the fire',
                    fontsize=16, fontweight='bold')
                plt.ylabel('')
                plt.yticks(ticks=range(0,7), labels=days_labels, fontsize=13)
                plt.xlabel('Damaged surface (ha)', fontsize=13)
                plt.legend(ncol=2, bbox_to_anchor=(0.99, -0.2))
                st.pyplot(fig.figure)
                st.write('It can be seen that fires caused by lightning strikes \
                    and technical accidents cause more damage on Sundays, probably \
                    because the means of fire control, especially personnel, \
                    are more reduced than on weekdays.')
            else:
                fig = make_barplot(df_fires,'DISC_MONTH','FIRE_SIZE',
                    xlabels = months_labels,
                    ytitle = 'Average surface damaged per year (ha)',
                    title = 'Avg damaged surface of fires depending on \nthe month',
                    palette = month_colors)
                st.pyplot(fig, use_container_width=True)
                st.write('It is during the summer period, especially in June, \
                    that fires are the most devastating in terms of area burned')
                fig = make_barplot(months_year_total, 'DISC_MONTH', 'FIRE_SIZE',
                    title = 'Total burnt surfaces for each month, 1992-2018',
                    xtitle = '', x_rot = 0, ytitle = 'Total surface damaged per year (ha)',
                    palette = month_colors, xlabels = months_labels, edgecolor = 'black')
                st.pyplot(fig, use_container_width=True)
                st.write('Nonetheless, when the total area burnt over the entire period is considered, \
                    July is the most damaging month, as fires are more numerous, even though \
                    their average area is smaller than in June.')
                fig = make_barplot(df_fires, 'DISC_DOW', 'FIRE_SIZE',
                    title = 'Avg size of a fire depending \non the day of the week (ha)',
                    xtitle = '', x_rot = 0, xlabels = days_labels,
                    ytitle = 'Average fire size (ha)', palette = days_colors,
                    edgecolor = 'black', linewidth = 0.8,
                    errcolor='.26', errwidth=None)
                st.pyplot(fig, use_container_width=True)
                st.write('There is no particular day when the fires are particulary wide. This is probably \
                    due to the fact that wide fires are mostly caused by natural causes that \
                    are not subject to a weekly schedule.')
    if option_main_variable == fires_causes :
        c1, c2 = st.columns((1.75, 1))
        with c1 :
            st.header('What are the major causes of wildfires ?')
            st.write('It is clear on the figure (right) that the majority of wildfires\
                are caused by human misbehaviors. BALBLALL')
        with c2 :
            fig = make_countplot_with_annot(df_fires, 'CAUSE', title = 'Causes of wildfires',
                order = causes_labels, xlabels = causes_labels_split, height = 4,
                ytitle = 'Total number of fires', palette = causes_color)
            st.pyplot(fig, use_container_width=True)

        c1, c2 = st.columns((1, 1.75))
        with c1 :
            st.write('Text')
        with c2 :
            fig = make_countplot(df_fires, 'DISC_YEAR', hue = 'CAUSE',
                title = 'Evolution of the cause of wildfire from 1992 to 2018',
                ytitle = 'Number of fires', x_rot = 90, edgecolor = 'white', linewidth = 0.4,
                hue_order = causes_labels, palette = causes_color, rm_legend = True)
            st.pyplot(fig, use_container_width=True)
            fig = make_barplot(cause_month_year, 'DISC_MONTH', 'STATE',
                hue = 'CAUSE', hue_order = causes_labels,
                title = 'Average number of wildfires per year depending on the cause',
                xlabels = months_labels,
                ytitle = 'Number of fires', x_rot = 0, edgecolor = 'black', linewidth = 0.4,
                palette = causes_color, rm_legend = True)
            st.pyplot(fig, use_container_width=True)

        st.write('We can have a closer look to the "Individuals\' mistake" category \
            (see below), ')
        fig = make_barplot(cause_month_year[cause_month_year.CAUSE == 'Individuals\' mistake'], 'DISC_MONTH', 'STATE',
            hue = 'NWCG_GENERAL_CAUSE',
            title = 'Average number of wildfires per year depending on the cause',
            xlabels = months_labels,
            ytitle = 'Number of fires', x_rot = 0, edgecolor = 'black', linewidth = 0.4,
            palette = 'Paired', rm_legend = False, ncol = 2)
        st.pyplot(fig, use_container_width=True)


# ---------------------------------------------
# ---------------------------- Plots State by State
#----------------------------------------------
elif genre == 'By State' :
    selected_state = st.selectbox(
     'Select the state you would like to analyse',
     np.sort(df_fires.STATE_FULL.unique()))
    #
    # Prepare subsets used later
    #
    df_sub = df_fires[(df_fires.STATE_FULL == selected_state)]
    state_abb = df_sub.STATE.unique()[0]
    state_nb_month = df_sub.groupby( [ 'DISC_YEAR', 'DISC_MONTH' ],
        as_index=False).agg({'FPA_ID' : 'count'})
    state_surf_year = df_sub.groupby( [ 'DISC_YEAR' ],
        as_index=False).agg({'FIRE_SIZE' : 'sum'})
    ct_cause_state_year = pd.crosstab(df_sub.DISC_YEAR, df_sub.CAUSE)
    ct_cause_state_year_perc = ct_cause_state_year.apply(lambda x : (x/x.sum()) *100, axis  = 1)
    cause_on = st.radio(
        "Do you want to separate results by the cause of the wildfire ?",
        ('Yes', 'No'))
    if cause_on == 'Yes' :
        c1, c2, c3 = st.columns((0.62, 1, 0.01)) # "Hide" the 3rd column
    else :
        c1, c2, c3 = st.columns((1.25, 1, 1))
    with c1 :
        map_type_fires = st.radio(
            "Map type :", ('Year by year', 'Average by year'))
        if map_type_fires == 'Year by year' :
            # df_subset = df_sub[df_sub.DISC_YEAR == year_plot]
            fig = px.scatter_geo(
                df_sub,
                lat = 'lat',
                lon = 'lon',
                locationmode = 'USA-states',
                color = 'CAUSE',
                category_orders = dico_causes,
                color_discrete_map = dico_causes_colors,
                animation_frame = 'DISC_YEAR',
                size = 'FIRE_SIZE',
                size_max = 50,
                opacity = 0.8,
                )
            fig.update_layout(
                title_text='Number of fires in ' + selected_state,
                legend = dict(
                    title = '',
                    yanchor="bottom", y=0.7,
                    xanchor="left", x=0.7),
                geo = dict(
                    scope='usa',
                    projection=go.layout.geo.Projection(type = 'albers usa'),
                    showlakes=True, # lakes
                    lakecolor='rgb(255, 255, 255)'),
                    margin=dict(l=0, r=0, b=0, t=30, pad=2  )
                    )
            fig.update_geos(fitbounds="locations")
            st.plotly_chart(fig, use_container_width=True)
        elif map_type_fires == 'Average by year' :
            fig = px.scatter_geo(
                df_sub.groupby(['DISC_YEAR', 'CAUSE'],
                    as_index = False).agg({'lat' : 'mean', 'lon' : 'mean', 'FIRE_SIZE' : 'sum'}),
                lat = 'lat',
                lon = 'lon',
                locationmode = 'USA-states',
                color = 'CAUSE',
                category_orders = dico_causes,
                color_discrete_map = dico_causes_colors,
                size = 'FIRE_SIZE',
                size_max = 50,
                opacity = 0.8,
                width = 800, height = 600,
                )
            fig.update_layout(
                title_text='Number of fires in ' + selected_state,
                legend = dict(
                    title = '',
                    yanchor="bottom", y=0.7,
                    xanchor="left", x=0.7),
                geo = dict(
                    scope='usa',
                    projection=go.layout.geo.Projection(type = 'albers usa'),
                    showlakes=True, # lakes
                    lakecolor='rgb(255, 255, 255)'),
                    margin=dict(l=0, r=0, b=0, t=30, pad=2  )
                    )
            fig.update_geos(fitbounds="locations")
            st.plotly_chart(fig, use_container_width=True)
            st.write('Each point represent all the fires due to one of the 5 \
                possible causes in one year. Their position has been calculated as \
                the average longitude and latitude coordinates from all the fires from a year.\
                Thus, it doesn\'t perfectly reflect the distribution of the fires geography\
                but it highlights the most affected region of a state.')
    with c2 :
        if cause_on == 'No' :
            f1 = make_countplot( df_sub,
                'DISC_YEAR', title = 'Number of fires per year in ' + selected_state,
    xtitle ='', ytitle = '\n\n', x_rot = 90, color_plot = color_fire)
            f2 = make_barplot(state_surf_year,
                'DISC_YEAR', 'FIRE_SIZE',
                title = "Total surface burned per year in " + selected_state,
                xtitle = '', x_rot = 90, xlabels = range(1992, 2019),
                ytitle = 'Total surface burned\n(hectares)', palette = [color_fire])
            f3 = make_boxplot(state_nb_month,
                'DISC_MONTH', 'FPA_ID',
                title = "Number of fires per month in " + selected_state,
                xtitle = '', x_rot = 0, xlabels = months_labels,
                ytitle = 'Number of fires \nper year', palette = month_colors)
            st.pyplot(f1, use_container_width=True)
            st.pyplot(f2, use_container_width=True)
            st.pyplot(f3, use_container_width=True)
        else :
            f1 = make_countplot_with_annot(df_sub, 'CAUSE',
                title = 'Distribution of the wildfires causes in ' + selected_state,
                xtitle ='', ytitle = '\n\n', x_rot = 0,
                order = causes_labels, xlabels = causes_labels_split, palette = causes_color)
            f2 = make_countplot(df_sub, 'DISC_YEAR',
                hue = 'CAUSE', hue_order = causes_labels,
                ytitle = '',
                title = 'Number of fires per year in ' + selected_state + "\ndepending of the cause",
                xtitle ='', x_rot = 90, palette = causes_color,
                edgecolor = 'white', linewidth = 0.1,
                width = 8, height = 2.5, rm_legend = True)
            f3 = make_barplot(df_sub.groupby(['DISC_YEAR', 'CAUSE'],
                as_index = False).agg({'FIRE_SIZE':'sum'}),
                'DISC_YEAR', 'FIRE_SIZE',
                hue = 'CAUSE', hue_order = causes_labels,
                title = 'Total surface burned per year in ' + selected_state +
                "\ndepending of the cause",
                xtitle = '', x_rot = 90, palette = causes_color,
                ytitle = 'Surface burned (hectares)',
                edgecolor = 'white', linewidth = 0.1,
                errcolor = '0.1', errwidth = 0.2,
                rm_legend = True)
            st.pyplot(f1, use_container_width=True)
            st.pyplot(f2, use_container_width=True)
            st.pyplot(f3, use_container_width=True)


    with c3:
        if cause_on == 'No' :
            f4 = ridgeplot(df_sub,
                title = 'Distribution of wildfires \nalong a year in ' + selected_state)
            st.pyplot(f4, use_container_width=True)
elif genre == 'Regional':
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, \
  sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
  Ut enim ad minim veniam, quis nostrud exercitation ullamco \
  laboris nisi ut aliquip ex ea commodo consequat. \
  Duis aute irure dolor in reprehenderit in voluptate \
  velit esse cillum dolore eu fugiat nulla pariatur. \
  Excepteur sint occaecat cupidatat non proident, \
  sunt in culpa qui officia deserunt mollit \
  anim id est laborum.")
    options_region = st.multiselect(
     'What regions do you want to highlight ?',
     ['East', 'West', 'North', 'South', 'Center',
     'North-East', 'North-West', 'South-East', 'South-West', 'Tropical'],
     ['East', 'West', 'North', 'South', 'Center',
     'North-East', 'North-West', 'South-East', 'South-West', 'Tropical'])
    c1, c2 = st.columns((1, 1.5))
    df_sub_region = df_fires[df_fires.Region.isin(options_region)]
    with c1:
        fig = px.choropleth(
            df_regions[df_regions.Region.isin(options_region)],
            locations='State',
            locationmode='USA-states',
            color = 'Region',
            hover_name = 'State',
            color_discrete_map  = dico_regions_colors,
            category_orders = dico_regions_order,
        )
        fig.add_trace(go.Scattergeo(
            locationmode = 'USA-states',
            locations=df_regions.State[df_regions.Region.isin(options_region)],
            text=df_regions.State[df_regions.Region.isin(options_region)],
            hoverinfo = 'skip',
            mode = 'text' ) )
        fig.update_layout(
            title_text='Regions',
            geo = dict(
                scope='usa',
                projection=go.layout.geo.Projection(type = 'albers usa'),
                showlakes=True, # lakes
                lakecolor='rgb(255, 255, 255)'),
                margin=dict(
                    l=0, r=0, b=0, t=30, pad=2  )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, \
  sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
  Ut enim ad minim veniam, quis nostrud exercitation ullamco \
  laboris nisi ut aliquip ex ea commodo consequat. \
  Duis aute irure dolor in reprehenderit in voluptate \
  velit esse cillum dolore eu fugiat nulla pariatur. \
  Excepteur sint occaecat cupidatat non proident, \
  sunt in culpa qui officia deserunt mollit \
  anim id est laborum.")
        st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, \
  sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
  Ut enim ad minim veniam, quis nostrud exercitation ullamco \
  laboris nisi ut aliquip ex ea commodo consequat. \
  Duis aute irure dolor in reprehenderit in voluptate \
  velit esse cillum dolore eu fugiat nulla pariatur. \
  Excepteur sint occaecat cupidatat non proident, \
  sunt in culpa qui officia deserunt mollit \
  anim id est laborum.")
    with c2:
        fig, ax = plt.subplots()
        region_cause_df[region_cause_df.index.isin(options_region)].plot(
            kind='barh',
            stacked=True,
            color={'Criminal':'#44AA99',
            "Individuals' mistake":'#332288',
            'Infrastructure accident':'#AA3377',
            'Natural (lightning)':'#CCBB44',
            'Other/Unknown':'grey'},
            alpha = 0.8,
            edgecolor = 'black', ax = ax
        )
        plt.ylabel('')
        ax.tick_params(axis='both', color = 'black', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('% of total fires reported from 1992 to 2018', fontsize = 14, labelpad=15)
        plt.legend(labels=['Criminal',"Individuals'\nmistake",
                           "Infrastructure \naccident",
                           "Natural \n(lightning)",
                           'Other/Unknown'],
                   bbox_to_anchor=(1.02, 0.8),
                   fontsize = 14, ncol = 1 )
        plt.title('Wildfire origins depending on the region', fontsize=15);
        st.pyplot(fig, use_container_width=True)

        fig, ax = plt.subplots()
        for r in regions_list :
            if r in options_region :
                col = dico_regions_colors[r]
                plt.plot(region_fire_number[r].rolling(10).mean(),
                    color = col, lw = 5, ls='-', label = r)
            else :
                plt.plot(region_fire_number[r].rolling(10).mean(),
                    color = 'lightgrey', lw = 3, ls=':', label = r)
        plt.ylim(0, 23000)
        plt.legend(fontsize = 14, ncol = 1, bbox_to_anchor=(1.02, 0.8) )
        plt.ylabel('Number of fires', fontsize=14)
        plt.title('Evolution of the number of fires in the main regions', fontsize=14)
        ax.set_xticks([2001,2006,2011,2016])
        ax.set_xticklabels(['2002', '2007', '2012', '2017'])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        f2 = make_barplot(saison_region_df[saison_region_df.Region.isin(options_region)],
            'Region', 'FPA_ID', hue = 'Season',
            title = "Average number of wildfires per year\n" +
             "depending of the season and the region",
             order = regions_list, xlabels = regions_list_split,
            xtitle = '', x_rot = 0,
            ytitle = 'Average number per year', palette = seasons_colors)
        st.pyplot(f2, use_container_width=True)
