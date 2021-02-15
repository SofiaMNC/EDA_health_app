# **Concept: A Public Health App**
*Sofia Chevrolat (June 2020)*
___
This study in 2 notebooks aims to estimate the feasability of exploiting <i>Open Food Facts</i>' database, containing more than a million food items, as part of a call for proposals by Santé publique France. 
The French national public health agency would like to be submitted with ideas for innovative applications on the theme of nutrition. 

Given the focus of the call to proposals, this pre-analysis will concentrate on three points :
- **The category of the food items** : their PNNS as well as NOVA groups and subgroups.
- **The nutritional values of the food items** : in particular, the big nutritional groups (proteins, fats and carbohydrates)

The study being carried out for Santé Publique France, the point of reference will be the products sold in France.
___

This study is divided into 2 notebooks: 
- A cleaning up notebook
- An analysis notebook
___
## Notebook 1 : Data Clean Up

This notebook is organised as follows:

**0. Setting Up**
- 0.1 Loading the necessary libraries
- 0.2 Defining custom functions
- 0.3 Loading the data set
- 0.4 Description of the data set

**1. Data Targeting**
- 1.1 Restriction to products sold in France
- 1.2 Selecting the columns
- 1.3 Separating the dataset between alcohol / no-alcohol products

**2. Clean Up**
- 2.1 Deleting totally empty columns and rows   
- 2.2 Handling duplicates
    * 1.2.1 At row level
    * 1.2.2 At column level
- 2.3 Handling NaN values
    * 1.5.1 In qualitative variables
    * 1.5.2 In quantitative variables
- 2.4 Handling outliers

**3. Exporting The Cleaned Up Data**
- 2.1 Exporting no-alcohol product data 
- 2.2 Exporting alcohol product data

___
## Notebook 2 : Data Analysis

This notebook is organized as follows:

**0. Setting Up**
- 0.1 Loading the necessary libraries
- 0.2 Defining custom functions
- 0.3 Loading the data set
- 0.4 Description of the data set
- 0.5 Selecting variables of interest

**1. Univariate Analysis**
- 1.1 Measures of central tendancy
    * 1.1.1 Qualitative variables
    * 1.1.2 Quantitative variables
- 1.2 Measures of dispersion
    * 1.2.1 Qualitative variables
    * 1.2.2 Quantitative variables
- 1.3 Measures of shape
    * 1.3.1 Skewness
    * 1.3.2 Kurtosis
- 1.4 Measures of concentration
    * 1.4.1 Lorenz curves
    * 1.4.2 Gini index

**2. Multivariate Analysis**
- 2.0 Setting Up
    * 2.0.1 Restricting and separating the dataset
    * 2.0.2 Selecting the relevant columns
- 2.1 Study of the relationships between qualitative variables
    * 2.1.1 Between the Nutri-Score letter and the PNNS 1 group
    * 2.1.2 Between the Nutri-Score letter and the PNNS2 group
    * 2.1.3 Between the Nutri-Score letter and the NOVA classification
- 2.2 Study of the relationships between qualitative variables
    * 2.2.1 Pair diagram of all qualitative variables
    * 2.2.2 Pearson correlation coefficients
    * 2.2.3 Spearman correlation coefficients
- 2.3 Study of the relationships between qualitative and quantitaive variables
    * 2.3.1 Between qualitatives variables and the Nutri-Score rating
    * 2.3.2 Between quantitatives variables and the Nutri-Score letter
- 2.4 Conclusions of the multivariate analysis

**3. Application**
- 3.1 Predicting the Nutri-Score rating
    * 3.1.1 Global model using all the data
    * 3.1.2 Improved model with a component on each PNNS 2 group
    * 3.1.3 Graphical representation: test values vs predicted values
- 3.2 Predicting the Nutri-Score letter
    * 3.2.1 Classification using the nutritional values
    * 3.2.2 Classification using the Nutri-Score rating
- 3.3 Model Validation
    * 3.3.1 Validation set for the model predicting the Nutri-Score rating
    * 3.3.2 Validation set for the classification between Nutri-Score letters
- 3.4 The special case of alcoholic beverages
    * 3.4.1 Refinement of the model to account for alcoholic beverages
    * 3.4.2 Final tests

_________

## Requirements

This assumes that you already have an environment allowing you to run Jupyter notebooks. 

The libraries used otherwise are listed in requirements.txt

_________

## Usage

1. The dataset is quite massive. Therefore, the notebook is configured to use a local version of the dataset, saved on disk after the very first download via Pandas. 
Therefore, you can either : 
    - Download the dataset from [Open Food Facts' official website](https://world.openfoodfacts.org/data), and place it at root level with name "open_food_facts_df.csv".
    - Uncomment the cell number 3 and comment cell number 4. This will fetch the data directly from Open Food Fact's website.

2. The data set and the signification of its columns are described in details on [Open Food Facts' official website](https://world.openfoodfacts.org/data/data-fields.txt)

3. Run the following in your terminal to install all required libraries :

```bash
pip3 install -r requirements.txt
```

4. Run the notebooks in order (Notebook 1 first, then Notebook 2).
__________

## Results

For a complete presentation and commentary of the results of this analysis, please see the PowerPoint presentation.

NOTE: The presentation is in French.