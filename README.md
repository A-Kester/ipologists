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

The project required access to financial, macroeconomic, and textual data sources, as well as tools for data processing and feature extraction

1. IPO Financial Data (Bloomberg Terminal)
We will use the Bloomberg Terminal provided through Middlebury College library to obtain all our historical IPO data. The data will be exported from Bloomberg into Excel/CSV format. 

2. SEC EDGAR filings
We will use the SEC EDGAR database to collect IPO S-1 filings. We will use the `sec-edgar-downloader` Python package. These filings will provide the textual data needed to extract our risk, growth, and business complexity features. 
3. Macroeconomic Data (FRED API)
Macroeconomic indicators will be obtained using the Federal Reserve Economic Data (FRED) API, which requires a [free API key](https://fred.stlouisfed.org/docs/api/fred/). 

4. Text Feature Extraction Tools:
To process S-1 filings, we may use:
- OpenAI API
- OR open-source model if costs or token limits make API usage impractical
We will select our approach based on computational cost and scalability after initial experimentation. If too costly, we might revisit how to feature extract from S-1 filings. 
5. Computing Resources
- We believe our personal laptops or computing resources providing by the college will be enough for our data processing and modeling. 
- We will use Google Colab for model training if we require additional computing power. 


# What You Will Learn (Both)

# Risk Statement (Anna)
Some risks to achieving the full deliverable are potential issues working with S-1 filing data. Since this data is textual, additional complications may arise in cleaning and preparing the data for training. Another potential risk is not having time for magnitude prediction if data cleaning and the implementation of the classification models takes longer than expected.

# Ethics Statement (Will)

1. Intended Impact

If our project is successful and deployed, we aim to democratize access to IPO analysis tools that are currently primarily available to institutional investors with access to Bloomberg terminals and large quantitative research teams. By developing a model, we hope to make IPO pricing more understandable and accessible to a broader audience. 

2. Who Benefits?
- Retail and individual investors who do not have access to Bloomberg or institutional grade IPO research.
- Researchers that study IPO markets and could use our open-source models as a baseline
- Companies who are considering going public that want a neutral and data-driven view of IPO pricing.

3. Who may be Harmed?
- Investors in non-US or emerging markets, since our dataset focuses on US IPOs with Bloomberg coverage, it might limit generalizability across global markets
- Users who might misinterpret the models outputs ad financial advice. The models are being designed for educational and analytical purposes only. This could lead to misleading financial decision-making
- Smaller firms or underrepresented sectors: our dataset and SEC EDGAR filings are biased towards larger IPOs which could lead to skewness or less accurate prediction for these less represented groups.

4. Will it make the World a Better Place?
We believe the project could contribute positively to society. We assume that increasing transparency in financial markets is beneficial, and making IPO pricing dynamics more understandable helps reduce information asymmetry between different types of investors. This would allow for more informed decision-making and possibly improve fairness in capital markets. We also assume that by helping explain or predicting IPO underpricing, it could lead to a reduction in mispricing and improve capital allocation in the economy. Which would lead to a more efficient market.

Although  our project is not intended to make recommendations, we do highlight potential forms of algorithmic bias in our work. Because our dataset is limited to US IPOs, our model may overrepresent larger more visible companies and underrepresent smaller firms.

Overall, we believe the project could improve understanding of IPO markets and make financial analysis more accessible, but its positive impacts depends on careful interpretation of the results.

# Tentative Timeline (Anna)
After two weeks we plan to have preliminary models working. We plan to have our data collected and cleaned by the end of week 1 and model implementations by the end of week 2.
After four weeks we plan to have our models and deliverable finalized. Time permitting, we will work on any extra add-ons to enhance our project.