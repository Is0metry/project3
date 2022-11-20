# Goals
* Determine predictors of property tax value within zillow data set
* Use those factors to generate a Linear Regression model to predict Tax Value
* Use this model to predict tax value on out of sample data

# Plan
- Acquire data from database
- Prepare data
    - Remove null values from data
    - Discard outliers
	- Create engineered features using existing data:
		- half_baths
- Explore data in search of predictors of Tax Value
	- Answer the following initial questions
		- Does Calculated Square Feet correlate with Tax Value?
        - Does FIPS code affect Tax Value?
        - Do the number of bedrooms affect Tax Value?
		- Does number of add-ons affect churn?
- Model data

- Draw conclusions

# Data Dictionary
Variable Name | `zillow` Database Equivalent | Definition
---|---|---
**bed_count** | bedroomcnt | No. of bedrooms in property
**calc_sqft**|calculatedfinishedsquarefeet | The calculated finished square footage of property
**fips** | fips | FIPS code of property 
**fips_6037**| N/A | Dummy value where `fips == 6037` (used in modeling)
**fips_6059**| N/A | Dummy value where `fips == 6059` (used in modeling)
**fips_6111**| N/A | Dummy value where `fips == 6111` (used in modeling)
**full_baths** | fullbathroomcnt | No. of full bathrooms on property
**half_baths** | N/A | No. of half bathrooms on property 
**tax_value** | taxvaluedollarcnt | Tax value of the property

# Steps to Reproduce:
1. Clone this repo
2. provide env.py file with hostname, username, and password, and database for telco data
3. Run `final_report.ipynb`

# Conclusions
## Exploration
- Calculated Square Feet is the most significant predictor of Tax Value vs previous predictions
- Bedroom Count and FIPS code were also predictors, but to a lesser degree.
- Number of full baths was not a good predictor of Tax Value
## Modeling
- The final model of LASSO + LARS performed the best
- The final model also outperformed the baseline both in and out of sample.

## Next Steps
- broaden predictions tested on to include 2016 data
- Additional data verification.
- Use a classifier model to separate by valu`e
- Formal removal of outliers.