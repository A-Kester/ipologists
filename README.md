# IPOlogists

# Group Members
Anna Kester and Will Zambito

# Abstract

# Motivation and Question (Will)

When Airbnb went public, its stock opened at nearly double its offer price, leaving over $3 billion in potential capital. While extreme, this outcome is not unusual. IPO underpricing has been documented heavily in academic finance literature. On average, IPOs are underpriced by 10-20%, which represented billions of dollars in "money left on the table" annually. 

We are motivated by the idea that IPO underpricing is not purely random, but reflects patterns across different domains: deal characteristics, macroeconomics conditions, and the way firms describe themselves in regulatory filings. These relationships may be complex and nonlinear, making them well-suited for machine learning.

Our central research questions are:
- Can we predict whether an IPO will be underpriced more accurately than simply predicting it will naively?
- What features are most indicative of an IPO being overpriced?

We are motivated by the quality of our data across three domains:
- We have Bloomberg financial data for all IPO deals in the US since 2000 providing us with historical data on: Institution, Industry, Offer Price, Offer to 1st close, % Shares Out, Deal Size, and more relevant deal information.
- Through the FRED API, we have access to macroeconomic time series including interest rate, CPI, and unemployment, which can each be matched to the IPO date to capture market conditions
- Through SEC EDGAR, we have access to S-1 filings for hundreds of IPOs, which can be processed to extract structured signals around risk, growth narratives, and business clarity, which are rarely quantified. 

Together, these three data sources will allow us to construct a classification and regression model for IPO pricing dynamics that combine financial variables, macroeconomic context, and textual information. 


# Planned Deliverables (Anna)
Deliverables for Full Success:
- Python package containing all code and documentation with the following goals:
    - Accurate classification of IPOs as underpriced, within expected range, or overpriced
    - Magnitude prediction, as time allows
    - Learning about and implementing multiple models effectively (Logistic Regression, Pytorch MLP, Random Forest, XGBoost)
    - Robust feature engineering
- A Jupyter notebook for each of our models
- A blog post / GitHub pages site with the following
    - Model comparison and explanation of why differences arise across models
    - Identify key factors that lead to IPO underpricing
    - Explanatory visualizations 

Deliverables for Partial Success:
- Python package containing all code and documentation with the following goals:
    - Working classification of IPOs as underpriced, expected, or overpriced
    - Learning about and implementing at least 2 of the 4 planned models
- A Jupyter notebook for each of our models
- A well organized and presentable blog post of what we were able to accomplish

# Resources Required (Will)

# What You Will Learn (Both)

# Risk Statement (Anna)
Some risks to achieving the full deliverable are potential issues working with S-1 filing data. Since this data is textual, additional complications may arise in cleaning and preparing the data for training. Another potential risk is not having time for magnitude prediction if data cleaning and the implementation of the classification models takes longer than expected.

# Ethics Statement (Will)

# Tentative Timeline (Anna)
After two weeks we plan to have preliminary models working. We plan to have our data collected and cleaned by the end of week 1 and model implementations by the end of week 2.
After four weeks we plan to have our models and deliverable finalized. Time permitting, we will work on any extra add-ons to enhance our project.