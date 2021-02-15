############ EDA Health App #############

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from sklearn import neighbors
from collections import Counter
from itertools import takewhile

##########################################
#                                        #
# Computation and Description Functions  #
#                                        #
##########################################

def getMissingValuesPercentPer(data):
    ''' 
        Calculates the mean percentage of missing values
        in a given pandas dataframe per unique value 
        of a given column
        
        Parameters
        ----------------
        data   : pandas dataframe
                 The dataframe to be analyzed
        
        Returns
        ---------------
        A pandas dataframe containing:
            - a column "column"
            - a column "Percent Missing" containing the percentage of 
              missing value for each value of column     
    '''
    
    missing_percent_df = pd.DataFrame({'Percent Missing':data.isnull().sum()/len(data)*100})

    missing_percent_df['Percent Filled'] = 100 - missing_percent_df['Percent Missing']

    missing_percent_df['Total'] = 100

    percent_missing = data.isnull().sum() * 100 / len(data.columns)
    
    return missing_percent_df


#------------------------------------------

def descriptionJeuDeDonnees(sourceFiles):
    ''' 
        Outputs a presentation pandas dataframe for the dataset.
        
        Parameters
        ----------------
        A dictionary with:
        - keys : the names of the files
        - values : a list containing two values : 
            - the dataframe for the data
            - a brief description of the file
        
        Returns
        ---------------
        A pandas dataframe containing:
            - a column "Nom du fichier" : the name of the file
            - a column "Nb de lignes"   : the number of rows per file 
            - a column "Nb de colonnes" : the number of columns per file
            - a column "Description"    : a brief description of the file     
    '''

    print("Les données se décomposent en {} fichier(s): \n".format(len(sourceFiles)))

    filenames = []
    files_nb_lines = []
    files_nb_columns = []
    files_descriptions = []

    for filename, file_data in sourceFiles.items():
        filenames.append(filename)
        files_nb_lines.append(len(file_data[0]))
        files_nb_columns.append(len(file_data[0].columns))
        files_descriptions.append(file_data[1])

        
    # Create a dataframe for presentation purposes
    presentation_df = pd.DataFrame({'Nom du fichier':filenames, 
                                    'Nb de lignes':files_nb_lines, 
                                    'Nb de colonnes':files_nb_columns, 
                                    'Description': files_descriptions})

    presentation_df.index += 1

    
    return presentation_df

#------------------------------------------

def getBootstrapMean(X, numberb):
    '''
        Calculate the bootstrap mean for a column
        Source : Introduction to Data Science,
        
        Parameters
        ----------------
        - X       : the data series to calculate the
                    bootstrap mean on
        - numberb : the number of samplings to perform
        
        Returns
        ---------------
        The bootstrap men of the column
    '''
    
    x = [0] * numberb
    
    for i in range(numberb):
        sample = [X[j]
                  for j
                  in np.random.randint(len(X), size=len(X))
                 ]
        x[i] = np.mean(sample)
    return mean(x)
    
#------------------------------------------

#-----------------------------------------------------------------
# WARNING : only works if 2 doublons
# TODO : Systemize function below to work for 3 or more doublons
#-----------------------------------------------------------------

def fuseDoublonRows(data, criteria_doublon):
    '''
        Returns a dataframe containing one fused row
        instead of two doublons rows for each couple of doublons
        
        ----------------
        - data             : DataFrame containing doublon rows
        - criteria_doublon : the name of the column to determine if
                             two rows are doublons
        
        Returns
        ---------------
        The dataframe with the fused rows, without the doublons
    '''
    
    duplicates = data[data[criteria_doublon].duplicated()][criteria_doublon]
    data_duplicates = data[data[criteria_doublon].isin(duplicates)]

    #-----------------------------------------------------------------
    # Merging the data from the doublons :
    # the new row becomes the first one, with its NaN values replaced
    # by the non NaN values of the second row
    #-----------------------------------------------------------------

    merged_duplicates = pd.DataFrame()

    for criteria_duplicate in duplicates:
        tmp_df = data_duplicates[data_duplicates[criteria_doublon]==criteria_duplicate]

        obj_df = tmp_df.select_dtypes(include=[np.object])
        num_df = tmp_df.select_dtypes(exclude=[np.object])

        merged_row = pd.concat([obj_df.head(1).fillna(obj_df.tail(1)).reset_index(drop=True),
                                num_df.max().to_frame().T.reset_index(drop=True)], axis=1)

        merged_duplicates = pd.concat([merged_duplicates, merged_row])

    #-----------------------------------------------------------------
    # Creating a new dataframe without the doublons, but with the
    # newly merged rows
    #-----------------------------------------------------------------

    return pd.concat([data[~data[criteria_doublon].isin(duplicates)],
                            merged_duplicates])
     
#------------------------------------------

def replaceValues(data, column, values):
    '''
        Replaces in place the given values by the given replacement
        values in a column of a dataframe
        
        ----------------
        - data   : DataFrame containing the values to
                   replace
        - column : The name of the column containing the values
                    to replace
        - values : A dictionary containing the values to replace
                   as keys, and the replacement values as values
        
        Returns
        ---------------
        _
    '''
    
    for value_error, value_correct in values.items():
        data.loc[data[column] == value_error, column] = value_correct
        
#------------------------------------------

def replaceNaNValues(data, cols, criterion, valueType):
    '''
        Replaces in place the NaN values in the list of columns given
        grouped by criterion if given, by the mode or the mean
        depending on the valueType given ("QUAL" or "QUANT")
        
        ----------------
        - data      : DataFrame containing the NaN values
        - cols      : A list containing the names of the columns in which
                      to replace the NaN values
        - criterion : The name of the column to groupby on for the calculation
                      of the mean
        - valueType : The type of values contained in the columns listed in
                      "cols" : "QUAL" or "QUANT"
        
        Returns
        ---------------
        _
    '''
    
    for column in cols:
        if criterion != None:
            value_per_criterion = {}

            for val_criterion, data_df in data.groupby([criterion]):
                if valueType == "QUAL":
                    value_per_criterion[val_criterion] = data_df[column].mode()[0]
                elif valueType == "QUANT":
                    value_per_criterion[val_criterion] = data_df[column].mean()

            for criterion_value, value in value_per_criterion.items():
                data.loc[data[criterion] == criterion_value, column] \
                = \
                data.loc[data[criterion] == criterion_value, column].fillna(value)
        else:
            if valueType == "QUAL":
                value = data[column].mode()[0]
            elif valueType == "QUANT":
                value = data[column].mean()
            else:
                raise Exception("Invalid value type :" + valueType)

            data[column] = data.loc[:, column].fillna(value)
    
#------------------------------------------

def replaceNaNQualitative(data, qualitative_cols, criterion=None):
    '''
        ** ONLY FOR COLUMNS CONTAINING QUALITATIVE INFO **
        
        Replaces in place the NaN values in the list of columns given
        by the mean of the column, grouped by criterion if given
        
        ----------------
        - data             : DataFrame containing the NaN values
        - qualitative_cols : A list containing the names of the
                             columns in which to replace the NaN
                             values
        - criterion        : The name of the column to groupby on for the
                             calculation of the mean
        
        Returns
        ---------------
        _
    '''
    
    replaceNaNValues(data, qualitative_cols, criterion, "QUAL")

#------------------------------------------

def replaceNaNQuantitative(data, quantitative_cols, criterion=None):
    '''
        ** ONLY FOR COLUMNS CONTAINING QUANTITATIVE INFO **
        
        Replaces in place the NaN values in the list of columns given
        by the mode of the column, grouped by criterion if given
        
        ----------------
        - data              : DataFrame containing the NaN values
        - quantitative_cols : A list containing the names of the
                              columns in which to replace the NaN
                              values
        - criterion         : The name of the column to groupby on for the
                              calculation of the mode
        
        Returns
        ---------------
        _
    '''
    
    replaceNaNValues(data, quantitative_cols, criterion, "QUANT")
              
#------------------------------------------

def removeOutlierValues(data, quantitative_columns, criterion=None):
    '''
        Removes the outliers in the given dataframe, grouped by criterion
        if given, in the given columns containing quantitative data
        
        ----------------
        - data                : The dataframe from which outliers need to
                                be removed
        - qualitative_columns : A list containing the names of the
                                columns in which to replace the NaN
                                values
        - criterion            : The name of the column to groupby and
                                 determine the outliers on
        
        Returns
        ---------------
        _ filtered_df : the dataframe without the outliers
    '''
    
    filtered_df = data.copy()

    if criterion != None:
        ## TODO : IMPLEMENT
        #for criterion_value, data_criterion in filtered_df.groupby([criterion]):
        print("TO IMPLEMENT")
        
    else:
        for column in quantitative_columns:
            Q1 = filtered_df[column].quantile(0.25)
            Q3 = filtered_df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
            filtered_df = filtered_df.query('(@Q1 - 1.5 * @IQR) <= '+ str(column) +' <= (@Q3 + 1.5 * @IQR)')
          
    return filtered_df

#------------------------------------------

def getContingencyTable(data, x, y):
    '''
        Returns a dataframe containing the contingency
        table for the heatmap
        
        ----------------
        - data : The dataframe from which the table will be calculated
        - x    : The name of the 1st variable (columns)
        - y    : The name of the 2nd variable (rows)
        
        Returns
        ---------------
        data_df_contingency : a dataframe containing the contingency table
    '''
    
    data_df_contingency = pd.crosstab(data[y], data[x])
    data_df_contingency["Total"] = data_df_contingency.sum(axis=1)
    index_columns = data_df_contingency.index.tolist()
    data_df_contingency = data_df_contingency.append(data_df_contingency.sum(numeric_only=True),
                                                                             ignore_index=True)
    data_df_contingency[y]= index_columns + ["Total"]
    data_df_contingency = data_df_contingency.set_index(y)

    return data_df_contingency

#------------------------------------------

def getKhi2TableAndData(data_contingency):
    '''
        Carries out a Khi-2 test on a contingency dataframe.
        
        ----------------
        - data_contingency : The dataframe containing the contingency
                             table
        
        Returns
        ---------------
        A tuple containing :
            - table : the Khi-2 table
            - stat  : the result of the Khi-2 test
            - p     : the p-value
            - expected : the expected table in case of H0
    '''
    
    stat, p, dof, expected = chi2_contingency(data_contingency)

    measure = (data_contingency-expected)**2/expected
    table = measure/stat

    return (table, stat, p, dof, expected)

#------------------------------------------

def getContingencyKhi2Data(data, x, y):
    '''
        Calculates the contingency table for a given dataframe
        and couple of variables, then carries out a Khi-2 test
        on this contingency table.
        
        ----------------
        - data_contingency : The dataframe containing the contingency
                             table
        
        Returns
        ---------------
        A tuple containing :
            - contingency_table : the contingency table
            - table             : the Khi-2 table
            - stat              : the result of the Khi-2 test
            - p                 : the p-value
            - expected          : the expected table in case of H0
    '''
    
    contingency_table = getContingencyTable(data, x, y)
    
    (table_khi2, stat, p, dof, expected_table) = getKhi2TableAndData(contingency_table)
    
    return (contingency_table, table_khi2, stat, p, dof, expected_table)
    
#------------------------------------------

def areDependent_withStatTest(stat, dof, prob=0.95):
    '''
        Checks independence of variables using Khi-2
        test value
        
        ----------------
        - stat : The Khi-2
        - dof  : degrees of freedom
        - prob : probability to be right rejecting H0
        
        Returns
        ---------------
        True  : H0 can be rejected with 1-prob chances
                to be wrong
        False : H0 could not be rejected for the prob
                given
    '''
    
    return abs(stat) >= chi2.ppf(prob, dof)

#------------------------------------------

def areDependent_withPValue(p, prob=0.95):
    '''
        Checks independence of variables using p-value
        from Khi-2 test
        
        ----------------
        - p : p-value from Khi-2 test
        - prob : probability to be wrong rejecting H0
        
        Returns
        ---------------
        True  : H0 can be rejected with prob chances
                to be right
        False : H0 could not be rejected for the prob
                given
    '''
    
    return p < 1.0 - prob
    
#------------------------------------------

def areDependent(stat, dof, p_value, prob=0.95):
    '''
        Checks independence of variables examining
        Khi-2 and p-values.
        
        ----------------
        - stat : The Khi-2
        - dof  : degrees of freedom
        - p_value : p-value
        - prob : probability to be right rejecting H0
        
        Returns
        ---------------
        True  : H0 can be rejected with 1-prob chances
                to be wrong
        False : H0 could not be rejected for the prob
                given
    '''
    
    if areDependent_withStatTest(stat, dof) and areDependent_withPValue(p_value):
        print("The variables are dependent - H0 rejected")
    elif ~areDependent_withStatTest(stat, dof):
        print("The stat test has failed to reject H0 - variables may be independent")
    elif ~areDependent_withPValue(p_value):
        print("The p-value test has failed to reject H0 - variables may be independent")
    else:
        print("The variables are independent - Fail to reject H0")

#------------------------------------------

def eta_squared(data, x_qualit,y_quantit):
    '''
        Calculate the proportion of variance
        in the given quantitative variable for
        the given qualitative variable
        
        ----------------
        - data      : The dataframe containing the data
        - x_quantit : The name of the qualitative variable
        - y_quantit : The name of the quantitative variable
        
        Returns
        ---------------
        Eta_squared
    '''
    
    sous_echantillon = data.copy().dropna(how="any")

    x = sous_echantillon[x_qualit]
    y = sous_echantillon[y_quantit]

    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT
    
#------------------------------------------

def calculNutrigradeFromNutriscore(nscore, pnns_group_1, pnns_group_2):
    '''
        Determines the Nutri-Score letter from the Nutri-Score score
        and the pnns groups
        
        ----------------
        - nscore       : The Nutri-Score score of the product
        - pnns_group_1 : The PNNS group 1 of the product
        - pnns_group_2 : The PNNS group 2 of the product
        
        Returns
        ---------------
        The Nutri-Score letter for the product
    '''
    
    if pnns_group_1 != "Beverages":
        if nscore <=-1:
            return "a"
        elif nscore >= 0 and nscore <= 2:
            return "b"
        elif nscore >= 3 and nscore <= 10:
            return "c"
        elif nscore >= 11 and nscore <= 18:
            return "d"
        else:
            return "e"
    else:
        if pnns_group_2 == "Waters and flavored waters":
            return "a"
        elif pnns_group_2 == "Alcoholic beverages":
            return "f"
        elif nscore <=-1:
            return "b"
        elif nscore >= 2 and nscore <= 5:
            return "c"
        elif nscore >= 6 and nscore <= 9:
            return "d"
        else:
            return "e"
 
#------------------------------------------

def getLorenzGini(data):
    '''
        Calculate the lorenz curve and Gini coeff
        for a given variable
        
        ----------------
        - data       : data series
        
        Returns
        ---------------
        A tuple containing :
        - lorenz_df : a list containing the values for the
                   Lorenz curve
        - gini_coeff : the associated Gini coeff
        
        Source : www.openclassrooms.com
    '''
    
    dep = data.dropna().values
    n = len(dep)
    lorenz = np.cumsum(np.sort(dep)) / dep.sum()
    lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0

    #---------------------------------------------------
    # Gini :
    # Surface sous la courbe de Lorenz. Le 1er segment
    # (lorenz[0]) est à moitié en dessous de 0, on le
    # coupe donc en 2, on fait de même pour le dernier
    # segment lorenz[-1] qui est à 1/2 au dessus de 1.
    #---------------------------------------------------

    AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n
    # surface entre la première bissectrice et le courbe de Lorenz
    S = 0.5 - AUC
    gini_coeff = [2*S]
         
    return (lorenz, gini_coeff)
    
#------------------------------------------

def getLorenzsGinis(data):
    '''
        Calculate the lorenz curve and Gini coeffs
        for all columns in the given dataframe
        
        ----------------
        - data       : dataframe
        
        Returns
        ---------------
        A tuple containing :
        - lorenz_df : a dataframne containing the values for the
                      Lorenz curve for each column of the given
                      dataframe
        - gini_coeff : a dataframe containing the associated Gini
                       coeff for each column of the given dataframe
    '''
    
    ginis_df = pd.DataFrame()
    lorenzs_df = pd.DataFrame()

    for ind_quant in data.columns.unique().tolist():
        lorenz, gini = getLorenzGini(data[ind_quant])
        ginis_df[ind_quant] = gini
        lorenzs_df[ind_quant] = lorenz

    n = len(lorenzs_df)
    xaxis = np.linspace(0-1/n,1+1/n,n+1)
    lorenzs_df["index"]=xaxis[:-1]
    lorenzs_df.set_index("index", inplace=True)
    
    ginis_df = ginis_df.T.rename(columns={0:'Indice Gini'})
    
    return (lorenzs_df, ginis_df)

#------------------------------------------

def getNutriScore(product_df, lrs, knn=None):
    '''
        Determines the Nutri-Score (grade and score) from
        a linear regression model trained on key nutritional
        values as well as the key nutritional and pnns groups
        values for each given product
        
        ----------------
        - product_df   : dataframe containing the key nutritional
                         values and the pnns groups for the products
        - lrs          : trained linear regression model
        - knn          : trained knn classifeir. If given, the Nutri-Score
                         grade is predicted via this knn from the Nutri-Score
                         score
        
        Returns
        ---------------
        The Nutri-Score (grade and score) for the products
    '''
    
    scores_predicted = []
    letters_predicted = []
        
    for idx,val in product_df.iterrows():
            
        if val["pnns_groups_2"] == "Alcoholic Beverages":
            return ([60], ["F"])
        else:
            # Predict Nutri-Score score
            score_predicted = lrs[val["pnns_groups_2"]].predict([val[["proteins_100g",
                                                                        "salt_100g",
                                                                        "energy_100g",
                                                                        "saturated-fat_100g",
                                                                        "sugars_100g"]]])[0][0]
            scores_predicted.append(score_predicted)
                                                                        
            letters_predicted.append(calculNutrigradeFromNutriscore(score_predicted, val["pnns_groups_1"],
                                                                                    val["pnns_groups_2"]))
            
    if knn != None :
        # Predict Nutri-Score letter
        letters_predicted = knn.predict([score_predicted])

    return scores_predicted, letters_predicted
    
##########################################
#                                        #
# Graphical Functions                    #
#                                        #
##########################################

def plotPercentageMissingValuesFor(data, long, larg):
    ''' 
        Plots the proportions of filled / missing values for each unique value
        in column as a horizontal bar chart.
        
        Parameters
        ----------------
        data : pandas dataframe with:
                - a column column
                - a column "Percent Filled"
                - a column "Percent Missing"
                - a column "Total"
                                 
       long : int
            The length of the figure for the plot
        
        larg : int
               The width of the figure for the plot
                                  
        
        Returns
        ---------------
        -
    '''
    
    data_to_plot = getMissingValuesPercentPer(data).sort_values("Percent Filled").reset_index()

    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 50


    sns.set(style="whitegrid")

    #sns.set_palette(sns.dark_palette("purple", reverse=True))

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(long, larg))

    plt.title("PROPORTIONS DE VALEURS RENSEIGNÉES / NON-RENSEIGNÉES PAR COLONNE",
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    b = sns.barplot(x="Total", y="index", data=data_to_plot,label="non renseignées", color="thistle", alpha=0.3)
    b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
    _, ylabels = plt.yticks()
    _, xlabels = plt.xticks()
    b.set_yticklabels(ylabels, size=TICK_SIZE)


    # Plot the Percent Filled values
    c = sns.barplot(x="Percent Filled", y="index", data=data_to_plot,label="renseignées", color="darkviolet")
    c.set_xticklabels(c.get_xticks(), size = TICK_SIZE)
    c.set_yticklabels(ylabels, size=TICK_SIZE)


    # Add a legend and informative axis label
    ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, ncol=1, frameon=True,
             fontsize=LEGEND_SIZE)

    ax.set(ylabel="Colonnes",xlabel="Pourcentage de valeurs (%)")

    lx = ax.get_xlabel()
    ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ly = ax.get_ylabel()
    ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:2d}'.format(int(x)) + '%'))
    ax.tick_params(axis='both', which='major', pad=TICK_PAD)

    sns.despine(left=True, bottom=True)

    plt.savefig('missingPercentagePerColumn.png')

    # Display the figure
    plt.show()

#------------------------------------------

def plotPieChart(data, groupby_col, long, larg, title, title_fig_save):
    ''' 
        Plots a pie chart of the proportion of each modality for groupby_col
        with the dimension (long, larg), with the given title and saved figure
        title.
        
        Parameters
        ----------------
        data           : pandas dataframe containing the data, with a "groupby_col"
                         column
        
        groupby_col    : the name of the quantitative column of which the modality
                         frequency should be plotted.
                                  
        long           : int
                         The length of the figure for the plot
        
        larg           : int
                         The width of the figure for the plot
        
        title          : title for the plot
        
        title_fig_save : title under which to save the figure
                 
        Returns
        ---------------
        -
    '''
    
    TITLE_SIZE = 25
    TITLE_PAD = 60

    # Initialize the figure
    f, ax = plt.subplots(figsize=(long, larg))


    # Set figure title
    # Set figure title
    plt.title(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD,)
       
    # Put everything in bold
    #plt.rcParams["font.weight"] = "bold"


    # Create pie chart for topics
    a = data[groupby_col].value_counts(normalize=True).plot(kind='pie',
                                                        autopct=lambda x:'{:2d}'.format(int(x)) + '%', 
                                                        fontsize =20)
    # Remove y axis label
    ax.set_ylabel('')
    
    # Make pie chart round, not elliptic
    plt.axis('equal') 
    
    # Save the figure 
    plt.savefig(title_fig_save)
    
    # Display the figure
    plt.show()

#------------------------------------------

def plotQualitativeDist(data, long, larg):
    '''
        Displays a bar chart showing the frequency of the modalities
        for each column of data.
        
        Parameters
        ----------------
        data : dataframe containing exclusively qualitative data
                                 
        long : int
               The length of the figure for the plot
        
        larg : int
               The width of the figure for the plot
                                  
        
        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 45
    TITLE_PAD = 1.05
    TICK_SIZE = 25
    TICK_PAD = 20
    LABEL_SIZE = 30
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    nb_rows = 2
    nb_cols = 2

    f, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    f.suptitle("DISTRIBUTION DES VALEURS QUALITATIVES", fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)


    row = 0
    column = 0

    for ind_qual in data.columns.tolist():
        
        data_to_plot = data.sort_values(by=ind_qual).copy()
        
        ax = axes[row, column]
        
        if ind_qual == "nutriscore_grade":
            nutri_colors = ["#287F46", "#78BB42", "#F9C623", "#E6792B", "#DC3D2A"]
            b = sns.countplot(y=ind_qual, data=data_to_plot, palette=sns.color_palette(nutri_colors),ax=ax)
        elif ind_qual == "nova_group":
            b = sns.countplot(y=ind_qual, data=data_to_plot, palette="Purples", ax=ax)
        else:
            b = sns.countplot(y=ind_qual, data=data_to_plot,
                              color="darkviolet",
                              ax=ax,
                              order = data_to_plot[ind_qual].value_counts().index)


        plt.tight_layout()
        
        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=1.4, hspace=0.2)

        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
        
        if ind_qual == "nova_group":
            ylabels = [item.get_text()[0] for item in ax.get_yticklabels()]
        else:
            ylabels = [item.get_text().upper() for item in ax.get_yticklabels()]
        b.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold")

        lx = ax.get_xlabel()
        ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
        ly = ax.get_ylabel()
        ax.set_ylabel(ly.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))
        
        ax.xaxis.grid(True)

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0
        
#------------------------------------------

def plotDistplotsWithRug(data, long, larg, nb_rows, nb_cols):
    '''
        Plots the distribution of all columns in the given
        dataframe (must be quantitative columns only) coupled
        with a rug plot of the distribution
        
        Parameters
        ----------------
        data : dataframe containing exclusively quantitative data
                                 
        long : int
               The length of the figure for the plot
        
        larg : int
               The width of the figure for the plot
               
        nb_rows : int
                  The number of rows in the subplot
        
        nb_cols : int
                  The number of cols in the subplot
                                 
        Returns
        ---------------
        -
    '''
        
    TITLE_SIZE = 30
    TITLE_PAD = 1.05
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    sns.set_palette(sns.dark_palette("purple", reverse="True"))

    f, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    f.suptitle("DISTRIBUTION DES VALEURS QUANTITATIVES", fontweight="bold",
               fontsize=TITLE_SIZE, y=TITLE_PAD)

    row = 0
    column = 0

    for ind_quant in data.columns.tolist():

        sns.despine(left=True)

        ax = axes[row, column]

        b = sns.distplot(data[ind_quant], ax=ax, rug=True)

        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
        b.set_xlabel(ind_quant,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        if ind_quant in ["saturated-fat_100g", "salt_100g"]:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))
        else:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

        plt.setp(axes, yticks=[])

        plt.tight_layout()

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0

#------------------------------------------

def plotDistplotWithRug(data, column, long, larg):
    '''
        Plots the distribution of the column variable
        in the given dataframe (must be a quantitative column)
        
        Parameters
        ----------------
        data   : dataframe containing a column named column
        
        column : the column for which the distribution should
                 be drawn
                                 
        long   : int
                 The length of the figure for the plot
        
        larg   : int
                  The width of the figure for the plot
                                 
        Returns
        ---------------
        -
    '''
        
    TITLE_SIZE = 30
    TITLE_PAD = 1.05
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    f, axes = plt.subplots(figsize=(long, larg))

    f.suptitle(column + " - DISTRIBUTION", fontweight="bold",
               fontsize=TITLE_SIZE, y=TITLE_PAD)

    sns.despine(left=True)
    
    sns.set_palette(sns.dark_palette("purple", reverse="True"))

    b = sns.distplot(data[column], rug=True)

    b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
    b.set_xlabel(column,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    if column in ["saturated-fat_100g", "salt_100g"]:
        axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))
    else:
        axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

    plt.setp(axes, yticks=[])

    plt.tight_layout()

#------------------------------------------

def plotBoxPlots(data, long, larg, nb_rows, nb_cols):
    '''
        Displays a boxplot for each column of data.
        
        Parameters
        ----------------
        data : dataframe containing exclusively quantitative data
                                 
        long : int
               The length of the figure for the plot
        
        larg : int
               The width of the figure for the plot
               
        nb_rows : int
                  The number of rows in the subplot
        
        nb_cols : int
                  The number of cols in the subplot
                                  
        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 35
    TITLE_PAD = 1.05
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 25
    LABEL_PAD = 10
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    f, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    f.suptitle("VALEURS QUANTITATIVES - DISTRIBUTION", fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)


    row = 0
    column = 0

    for ind_quant in data.columns.tolist():
        ax = axes[row, column]

        sns.despine(left=True)

        b = sns.boxplot(x=data[ind_quant], ax=ax, color="darkviolet")

        plt.setp(axes, yticks=[])

        plt.tight_layout()

        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)

        lx = ax.get_xlabel()
        ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
        if ind_quant == "salt_100g":
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))
        else:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

        ly = ax.get_ylabel()
        ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        ax.tick_params(axis='both', which='major', pad=TICK_PAD)

        ax.xaxis.grid(True)
        ax.set(ylabel="")

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0
                    
#------------------------------------------

def plotLorenz(lorenz_df, long, larg):
    '''
        Plots a Lorenz curve with the given title
        
        ----------------
        - lorenz_df : a dataframe containing the Lorenz values
                      one column = lorenz value for a variable
        - long       : int
                       The length of the figure for the plot
        
        - larg       : int
                       The width of the figure for the plot
        
        Returns
        ---------------
        _
    '''
    
    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 50


    sns.set(style="whitegrid")
    
    f, ax = plt.subplots(figsize=(long, larg))
    
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    plt.title("VARIABLES QUANTITATIVES - COURBES DE LORENZ",
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    sns.set_color_codes("pastel")
    
    b = sns.lineplot(data=lorenz_df, palette="pastel", linewidth=5, dashes=False)
    
    b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)

    b.set_yticklabels(b.get_yticks(), size = TICK_SIZE)

    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))
    
    ax.tick_params(axis='both', which='major', pad=TICK_PAD)

    ax.set_xlabel("")

    # Add a legend and informative axis label
    leg = ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, ncol=1, frameon=True,
             fontsize=LEGEND_SIZE)
    
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)


    # Display the figure
    plt.show()
 
#------------------------------------------
    
def plotContingencyTableHeatmap(data_contingency, data_khi2, long, larg):
    '''
        Plots a heatmap of the Khi-2 table given
        in data_khi2 with annotations from the contingency table
        given in data_contingency
        
        ----------------
        - data : a dataframe containing the Lorenz values
                 one column = lorenz value for a variable
        - long : int
                 The length of the figure for the plot
        
        - larg : int
                 The width of the figure for the plot
        
        Returns
        ---------------
        _
    '''
    TITLE_SIZE = 30
    TITLE_PAD = 0.95
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 25
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    f, axes = plt.subplots(figsize=(long, larg))

    f.suptitle("TABLEAU DE CONTINGENCE\nAVEC MISE EN LUMIÈRE DES RELATIONS PROBABLES (KHI-2)", fontweight="bold",
               fontsize=TITLE_SIZE, y=TITLE_PAD)

    b = sns.heatmap(data=data_khi2.iloc[:-1, :-1], annot=data_contingency.iloc[:-1, :-1], annot_kws={"fontsize":20}, fmt="d")

    xlabels = [item.get_text().upper() for item in axes.get_xticklabels()]
    b.set_xticklabels(xlabels, size=TICK_SIZE, weight="bold")
    b.set_xlabel(data_khi2.iloc[:-1, :-1].columns.name.upper(),fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ylabels = [item.get_text() for item in axes.get_yticklabels()]
    b.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold")
    b.set_ylabel(data_khi2.iloc[:-1, :-1].index.name.upper(),fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.show()
    
#------------------------------------------
    
def plotPairplot(data, height, hue=None):
    '''
        Plots a pairplot of all the quantitative variables
        in data
        
        ----------------
        - data   : a dataframe containing the data
        - height : an int for the height of the plot
        - hue    : qualitative column in data. If given,
                   plot will be colorized according to the
                   values in the hue column.
        
        Returns
        ---------------
        _
    '''
    TITLE_SIZE = 70
    TITLE_PAD = 1.05
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 45
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5
            
    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)  # fontsize of the figure title

    test_palette = {"a":"#287F46", "b":"#78BB42", "c":"#F9C623", "d":"#E6792B", "e":"#DC3D2A"}

    plt.rcParams["font.weight"] = "bold"

    if hue == None:
        b = sns.pairplot(data=data, height=height)
    else:
        b = sns.pairplot(data=data, hue=hue, height=height, palette=test_palette)

    b.fig.suptitle("VARIABLES QUALITATIVES - PAIRPLOT",
                   fontweight="bold",
                   fontsize=TITLE_SIZE, y=TITLE_PAD)



#------------------------------------------

def plotCorrelationHeatMap(data, corr_method, long, larg):
    '''
        Plots a heatmap of the correlation coefficients
        between the quantitative columns in data
        
        ----------------
        - data : a dataframe containing the data
        - corr : the correlation method ("pearson" or "spearman")
        - long : int
                 The length of the figure for the plot
        
        - larg : int
                 The width of the figure for the plot
        
        Returns
        ---------------
        _
    '''
    
    TITLE_SIZE = 40
    TITLE_PAD = 1
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 45
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5
    
    corr = data.corr(method = corr_method)

    f, ax = plt.subplots(figsize=(long, larg))
                
    f.suptitle("COEFFICIENT DE CORRÉLATION DE " + corr_method.upper(), fontweight="bold",
               fontsize=TITLE_SIZE, y=TITLE_PAD)

    b = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                    cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax,
                    annot=corr, annot_kws={"fontsize":20}, fmt=".2f")

    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    b.set_xticklabels(xlabels, size=TICK_SIZE, weight="bold")
    b.set_xlabel(data.columns.name,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    b.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold")
    b.set_ylabel(data.index.name,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.show()
                
#------------------------------------------

def plotKDE(data, column, groupby_col, long, larg):
    '''
        Plots a KDE plot of column in data, grouped by
        groupby_col.
        
        ----------------
        - data : a dataframe containing the data
        - column : the correlation method ("pearson" or "spearman")
        - groupby_col : int
                 The length of the figure for the plot
        
        - larg : int
                 The width of the figure for the plot
        
        Returns
        ---------------
        _
    '''
    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 30
    TICK_PAD = 20
    LABEL_SIZE = 25
    LABEL_PAD = 30
    LEGEND_SIZE = 15
    LINE_WIDTH = 3.5

    sns.set(style="whitegrid")

    sns.despine(left=True)

    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)

    f, ax = plt.subplots(figsize=(long, larg))

    f.suptitle("DISTRIBUTION DU NUTRISCORE PAR GROUPE PNNS 1",
              fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)

    plt.setp(ax, yticks=[])

    for groupby_criterion, data_df in data.groupby([groupby_col]):
        sns.kdeplot(data=data_df[column],
                    label= groupby_criterion, shade=True)
        
    ax.xaxis.grid(False)

    plt.legend()

#------------------------------------------
def plotSwarmplot(data, x_col, y_col, title, long, larg, rot=None):
    '''
        Plots a swarmplot of the columns x_col and y_col in data
        
        ----------------
        - data  : a dataframe containing the data
        - x_col : the name of a quantitative column in data
        - y_col : the name of a qualitative column in data
        - title : the title of the figure
        - long  : the length of the figure
        - larg  : the widht of the figure
        - rot   : the rotation to apply to the x tick labels
        
        Returns
        ---------------
        _
    '''

    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LINE_WIDTH = 3.5
    LEGEND_SIZE = 30

    sns.set(style="whitegrid")

    sns.set_palette("husl", 8)

    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)

    f, ax = plt.subplots(figsize=(long, larg))

    f.suptitle(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)
    
    if x_col != "nova_group":
        b = sns.swarmplot(x=data[x_col].sample(n=20000),
                          y=data[y_col].sample(n=20000),
                          order=data[y_col].sort_values().unique())
    else:
        b = sns.swarmplot(x=data[x_col].sample(n=20000),
                          y=data[y_col].sample(n=20000))
    
    
    if rot != None:
        plt.xticks(rotation=rot)
    
    if x_col != "nova_group":
        b.set_xticklabels(b.get_xticks(), weight="bold")

        ylabels = [item.get_text().upper() for item in ax.get_yticklabels()]
        b.set_yticklabels(ylabels,weight="bold")
    else:
        b.set_yticklabels(b.get_yticks(), weight="bold")

        xlabels = [item.get_text()[0] for item in ax.get_xticklabels()]
        b.set_xticklabels(xlabels,weight="bold")

    lx = ax.get_xlabel()
    ax.set_xlabel(lx.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
    ly = ax.get_ylabel()
    ax.set_ylabel(ly.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.show()

#------------------------------------------

def plotMultipleKDEplotShades(datas, col_x, col_y, alpha, title, long, larg):
    '''
        Plots KDE for the given data with the values of col_x as x axis and
        the values of col_y as y axis
        
        ----------------
        - data  : a dictionary containing :
                    - as keys : the name of the group to plot
                    - as values : a dictionary containing :
                            - the keys : data, legend_info and palette
                            - the values :
                                - the data pertaining to the group
                                - the position of the legend for the group
                                - the palette to be used for the group
                            
        - col_x : the name of a column present in each group data
        - col_y : the name of a column present in each group data
        - title : the title of the figure
        - long  : the length of the figure
        - larg  : the widht of the figure
        
        Returns
        ---------------
        _
    '''
    
    sns.set(style="whitegrid")

    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LINE_WIDTH = 3.5
    LEGEND_SIZE = 30

    # Set up the figure
    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)

    f, ax = plt.subplots(figsize=(long, larg))

    f.suptitle(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)

    for group, group_data in datas.items():
        data_to_plot = group_data["data"]
        palette_name = group_data["palette"]
        legend_x = group_data["legend_info"][0]
        legend_y = group_data["legend_info"][1]
        
        ax=sns.kdeplot(data_to_plot[col_x], data_to_plot[col_y],
                       cmap = palette_name,
                       shade=True, shade_lowest=False, alpha=alpha)

        lx = ax.get_xlabel()
        ax.set_xlabel(lx.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
        ly = ax.get_ylabel()
        ax.set_ylabel(ly.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        ax.text(legend_x, legend_y, group, size=16,
                color=sns.color_palette(palette_name)[-2], fontsize=LEGEND_SIZE)
         
#------------------------------------------

def plotScatterplotNutriScores(data, col_x, col_y, title, long, larg):
    '''
        Plots a scatterplot of col_y as a function of col_x with the
        data in data
        
        ----------------
        - data  : a dataframe containing the col_x and col_y columns
        - col_x : the name of a column present in data
        - col_y : the name of a column present in data
        - title : the title of the figure
        - long  : the length of the figure
        - larg  : the widht of the figure
        
        Returns
        ---------------
        _
    '''

    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LINE_WIDTH = 3.5
    LEGEND_SIZE = 15

    sns.set(style="whitegrid")

    plt.rcParams["font.weight"] = "bold"

    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)

    f, ax = plt.subplots(figsize=(long, larg))

    f.suptitle(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)

    nutri_colors = ["#287F46", "#78BB42", "#F9C623", "#E6792B", "#DC3D2A"]

    b = sns.scatterplot(x=col_x, y=col_y,
                        hue="nutriscore_grade", hue_order=['a', 'b', 'c', 'd', 'e'],
                        size="nutriscore_score", sizes=(1, 15),
                        linewidth=0, palette=sns.color_palette(nutri_colors),
                        data=data)

    lx = ax.get_xlabel()
    ax.set_xlabel(lx.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
    ly = ax.get_ylabel()
    ax.set_ylabel(ly.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ax.legend(bbox_to_anchor=(0.2,-0.4,0.6,0.2), loc="lower left", mode="expand",
              borderaxespad=0, ncol=2, frameon=True, fontsize=LEGEND_SIZE)

    plt.show()

#------------------------------------------

def plotSimpleScatterplot(col_x, col_y, title, long, larg):
    '''
        Plots a scatterplot of col_y as a function of col_x
        
        ----------------
        - col_x : the name of a column
        - col_y : the name of a column
        - title : the title of the figure
        - long  : the length of the figure
        - larg  : the widht of the figure
        
        Returns
        ---------------
        _
    '''

    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LINE_WIDTH = 3.5
    LEGEND_SIZE = 30

    sns.set(style="whitegrid")

    sns.set_palette(sns.dark_palette("purple", reverse=True))

    plt.rcParams["font.weight"] = "bold"

    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)

    f, ax = plt.subplots(figsize=(long, larg))

    f.suptitle(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)

    b = sns.scatterplot(x=col_x, y=col_y)

    lx = ax.get_xlabel()
    ax.set_xlabel(lx.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
    ly = ax.get_ylabel()
    ax.set_ylabel(ly.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.show()

#------------------------------------------

def plotRegplot(data, col_x, col_y, title, long, larg):
    '''
        Plots a regression plot of the columns col_y and
        col_x in data
        
        ----------------
        - data  : a dataframe containing the col_x and col_y columns
        - col_x : the name of a column present in data
        - col_y : the name of a column present in data
        - title : the title of the figure
        - long  : the length of the figure
        - larg  : the widht of the figure
        
        Returns
        ---------------
        _
    '''

    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LINE_WIDTH = 3.5
    LEGEND_SIZE = 30

    sns.set(style="whitegrid")

    sns.set_palette(sns.dark_palette("purple", reverse=True))

    plt.rcParams["font.weight"] = "bold"

    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)

    f, ax = plt.subplots(figsize=(long, larg))

    f.suptitle(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)

    b = sns.regplot(x=col_x, y=col_y, data=data)

    lx = ax.get_xlabel()
    ax.set_xlabel(lx.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
    ly = ax.get_ylabel()
    ax.set_ylabel(ly.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.show()

#------------------------------------------

def plotOptimalK(xtrain, ytrain, xtest, ytest, title, long, larg):
    '''
        Plots the error as a function of K
        
        ----------------
        - xtrain : x in training set
        - ytrain : y in training set
        - xtest  : x in test set
        - ytest  : y in test set
        - title  : the title of the figure
        - long   : the length of the figure
        - larg   : the widht of the figure
        
        Returns
        ---------------
        _
    '''

    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LINE_WIDTH = 3.5
    LEGEND_SIZE = 30

    sns.set(style="whitegrid")

    sns.set_palette(sns.dark_palette("purple", reverse=True))

    plt.rcParams["font.weight"] = "bold"

    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)

    errors = []
    for k in range(2,15):
        knn = neighbors.KNeighborsClassifier(k)
        errors.append(100*(1 - knn.fit(xtrain, ytrain).score(xtest, ytest)))
        
    f, ax = plt.subplots(figsize=(long, larg))

    f.suptitle(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)
              
    plt.plot(range(2,15), errors, 'o-')

    plt.show()

#------------------------------------------

def plotSimpleSwarmplot(col_x, col_y, title, long, larg):
    '''
        Plots a simple swarmplot of col_y as a function
        of col_x
        
        ----------------
        - col_x : a data series
        - col_y : a data series
        - title  : the title of the figure
        - long   : the length of the figure
        - larg   : the widht of the figure
        
        Returns
        ---------------
        _
    '''


    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LINE_WIDTH = 3.5
    LEGEND_SIZE = 30

    sns.set(style="whitegrid")

    sns.set_palette("husl", 8)

    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)

    f, ax = plt.subplots(figsize=(long, larg))

    f.suptitle(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)

    b = sns.swarmplot(x=col_x.sample(n=10000),
                      y=col_y.sample(n=10000),
                      order = ['a', 'b', 'c', 'd', 'e'])

    b.set_xticklabels(b.get_xticks(), weight="bold")

    ylabels = [item.get_text().upper() for item in ax.get_yticklabels()]
    b.set_yticklabels(ylabels,weight="bold")

    lx = ax.get_xlabel()
    ax.set_xlabel(lx.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
    ly = ax.get_ylabel()
    ax.set_ylabel(ly.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.show()

#------------------------------------------

def plotSuperposedScatterplot(data1_x, data1_y, data2_x, data2_y, title, long, larg):
    '''
        Plots a superposed scatterplot of the two given (x,y) pairs
        
        ----------------
        - data1_x : a dataseries
        - data1_y : a dataseries
        - data2_x : a dataseries
        - data2_y : a dataseries
        - title  : the title of the figure
        - long   : the length of the figure
        - larg   : the widht of the figure
        
        Returns
        ---------------
        _
    '''

    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LINE_WIDTH = 3.5
    LEGEND_SIZE = 30

    sns.set(style="whitegrid")

    plt.rcParams["font.weight"] = "bold"

    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)

    f, ax = plt.subplots(figsize=(long, larg))

    f.suptitle(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)

    b = sns.scatterplot(x=data1_x, y=data1_y, color="lavender")
    c = sns.scatterplot(x=data2_x, y=data2_y, color="darkviolet", alpha=0.8)

    lx = ax.get_xlabel()
    ax.set_xlabel(lx.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
    ly = ax.get_ylabel()
    ax.set_ylabel(ly.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.show()
    
#------------------------------------------

def plotSuperposedSwarmplot(data1_x, data1_y, data2_x, data2_y, title, long, larg):
    '''
        Plots a superposed swarmplot of the two given (x,y) pairs
        
        ----------------
        - data1_x : a dataseries
        - data1_y : a dataseries
        - data2_x : a dataseries
        - data2_y : a dataseries
        - title  : the title of the figure
        - long   : the length of the figure
        - larg   : the widht of the figure
        
        Returns
        ---------------
        _
    '''

    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LINE_WIDTH = 3.5
    LEGEND_SIZE = 30

    sns.set(style="whitegrid")

    plt.rcParams["font.weight"] = "bold"

    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)

    f, ax = plt.subplots(figsize=(long, larg))

    f.suptitle(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)

    b = sns.swarmplot(x=data1_x, y=data1_y)
    c = sns.swarmplot(x=data2_x, y=data2_y,alpha=0.8)

    plt.show()
