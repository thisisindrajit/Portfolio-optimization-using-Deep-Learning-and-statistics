# Portfolio optimization using Deep Learning and Statistics (Athena)

Portfolio optimization is the process of selecting the best portfolio (asset distribution), out of the set of all portfolios being considered, according to some objective. The objective typically maximizes factors such as expected return, and minimizes costs like financial risk.

Investors want to gain maximum returns from their portfolio while minimizing the risks associated. But purchasing assets without analyzing the fundamentals and merely relying on speculation and market sentiment is a significant problem in portfolio management. This is the problem that the model is trying to solve so that investors can get suggestions from the model and invest wisely.

We intend to propose a novel solution to optimize and enhance portfolio performance using a combination of deep learning and statistical models along with asset fundamentals analysis to obtain maximum returns with minimal risk.


## Uses

- The proposed model can be used to maximize the returns and minimize the risk of a given portfolio (a collection of stocks and other assets) by allocating an optimal weight for each asset in the portfolio.
- It can provide a portfolio that delivers high return per unit risk.
- It can also create a balanced portfolio with many different investments such as stocks, bonds and mutual funds.


## Data collection

For obtaining historical prices of stocks, a python package called YFinance ([yfinance · PyPI](https://pypi.org/project/yfinance/)) is used. 

After the user enters the ticker symbols of all stocks in their portfolio, it is sent to a function which uses APIs from YFinance to get the historical prices of stocks using the ticker symbols provided by the user.

```
def get_data(tickers, period="5y"):

  # checking if the length of the portfolio is greater than the allowed portfolio length
  if port_len > max_port_len:
    print(f'Only {max_port_len} number of assets allowed in portfolio!')
    return None
  
  # download data of ticker symbols
  data = yf.download(tickers, period=period)
  
  # check whether if any ticker symbol has less than minimum required previous days of data
  for i in range(port_len):
    cur_ticker_count = data.iloc[:, i].count()
    print(f'No of rows available for {tickers[i]} is {cur_ticker_count}')

    if cur_ticker_count < min_prev_days:
      print(f'{tickers[i]} has less than {min_prev_days} days of historical price data. Please consider removing the asset or adding some other asset.')
      return None
    
  # having only close prices and removing other data
  data = data.drop("Open", axis = 1)
  data = data.drop("High", axis = 1)
  data = data.drop("Low", axis = 1)
  data = data.drop("Volume", axis = 1)
  data = data.drop("Adj Close", axis = 1)

  return data


df = get_data(tickers, period="5y")
```

Also another API is used to get extra information about the stocks like balance sheets, income statements, financial statements and company information which will later be used for calculating the fundamental analysis scores.


## System Design

![Model architecture](https://user-images.githubusercontent.com/43838718/172583953-185763bd-3fbe-4646-b422-e984c497f792.png)


### Deep Learning Model

The proposed deep learning model is based on LSTM (Long Short Term Memory) networks. Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. LSTMs are explicitly designed to avoid the long-term dependency problem.

The architecture of the proposed deep learning model is shown below:

![dl model archi](https://user-images.githubusercontent.com/43838718/167290331-b869f59e-77d3-4cd2-a1a4-42c7b02d0c76.jpg)

The input to the model contained close prices and daily returns for each stock in the portfolio and the past 5 years of data was used to form a single input. It was observed that keeping daily returns in input data acted as a momentum feature during training of the model.

A custom loss function that maximizes the Sharpe ratio was used as the loss function. The optimizer used was Adam (Adaptive Moment Estimation). Below is the implementation of the custom loss function and compilation of the model:

```
def sharpe_loss(_, y_pred):

  # make all time-series start at 1 (Scaling down prices for fast multiplication)
  data = tf.divide(tf_port_data, tf_port_data[0])  
            
  # value of the portfolio after allocations applied
  # tf.multiply is element wise multiplication
  portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1) 
  
  portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula

  sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns)

  # since we want to maximize Sharpe and gradient descent minimizes the loss, 
  # we can negate Sharpe (the min of a negated function is its max)
  return -sharpe
  
opt_model.compile(loss=sharpe_loss, optimizer='adam')
```

The model was trained on a GPU on Google Colab for 50 epochs with no shuffling. The  output from the model was the portfolio weights that maximized the Sharpe ratio of the portfolio.


### Statistical Models

In the proposed model, two statistical models namely HRP (Hierarchical Risk Parity) and HERC (Hierarchical Equal Risk Contribution) were used. Both these models focused on diversifying risk among all assets in the portfolio while keeping the Sharpe ratio as optimal as possible.


### Asset fundamentals analyser

The asset fundamentals analyser is mainly used to analyse various metrics that can be used to define whether a stock is fundamentally strong or weak. The different metrics used were:

- Piotroski score
- RoE (Return on Equity)
- RoCE (Return on Capital Employed)
- P/E ratio (Price to Earnings ratio)
- P/B ratio (Price to Book ratio)
- Analyst recommendation
- CAGR (Compound Annual Growth Rate)
- Debt to asset ratio
- Is assets 1.5 times greater than liabilities

### Final output generator

The final output generator performs calculations based on the inputs provided and then returns the final optimized weights. In the final optimization step, fundamentally stronger stocks are given more importance during weight allocation, while the risk is kept as diversified as possible to make sure the optimized portfolio is robust, provides good returns with less risk and suits well for long-term investment.

## Results

The proposed model was tested on a portfolio consisting of Indian stocks and the inferences are presented below.
 
The portfolio consists of 10 Indian stocks listed in the National Stock Exchange of India (NSE). All stocks were selected based on analyst recommendations from various investment companies. The stocks in the portfolio along with their ticker symbols is shown below:

<table>
  <tr>
   <td>
   </td>
   <td><strong>Ticker symbols (NSE)</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Aarti Drugs</strong>
   </td>
   <td>AARTIDRUGS
   </td>
  </tr>
  <tr>
   <td><strong>APL Apollo Tubes</strong>
   </td>
   <td>APLAPOLLO
   </td>
  </tr>
  <tr>
   <td><strong>Birlasoft</strong>
   </td>
   <td>BSOFT
   </td>
  </tr>
  <tr>
   <td><strong>Coforge</strong>
   </td>
   <td>COFORGE
   </td>
  </tr>
  <tr>
   <td><strong>Dhampur Sugar Mills</strong>
   </td>
   <td>DHAMPURSUG
   </td>
  </tr>
  <tr>
   <td><strong>ICICI Securities</strong>
   </td>
   <td>ISEC
   </td>
  </tr>
  <tr>
   <td><strong>Lincoln pharma</strong>
   </td>
   <td>LINCOLN
   </td>
  </tr>
  <tr>
   <td><strong>State bank of India</strong>
   </td>
   <td>SBIN
   </td>
  </tr>
  <tr>
   <td><strong>Tata Power</strong>
   </td>
   <td>TATAPOWER
   </td>
  </tr>
  <tr>
   <td><strong>Thirumalai Chemicals Ltd</strong>
   </td>
   <td>TIRUMALCHM
   </td>
  </tr>
</table>

#### Weights allocated by different models (As obtained on 18th May 2022)

This table shows the weights allocated to each stock in the portfolio by different models. As per the proposed model, all of these weights are sent to the final output generator where based on the fundamentals analysis scores, the best weights are chosen for each stock and the final weights are obtained.

<table>
  <tr>
   <td>
   </td>
   <td><strong>DL model</strong>
   </td>
   <td><strong>HRP</strong>
   </td>
   <td><strong>HERC</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Aarti Drugs</strong>
   </td>
   <td>0.090153
   </td>
   <td>0.128619
   </td>
   <td>0.019307
   </td>
  </tr>
  <tr>
   <td><strong>APL Apollo Tubes</strong>
   </td>
   <td>0.35
   </td>
   <td>0.062015
   </td>
   <td>0.011910
   </td>
  </tr>
  <tr>
   <td><strong>Birlasoft</strong>
   </td>
   <td>0.062669
   </td>
   <td>0.066786
   </td>
   <td>0.036506
   </td>
  </tr>
  <tr>
   <td><strong>Coforge</strong>
   </td>
   <td>0.154747
   </td>
   <td>0.266242
   </td>
   <td>0.113753
   </td>
  </tr>
  <tr>
   <td><strong>Dhampur Sugar Mills</strong>
   </td>
   <td>0.090688
   </td>
   <td>0.083187
   </td>
   <td>0.165851
   </td>
  </tr>
  <tr>
   <td><strong>ICICI Securities</strong>
   </td>
   <td>0.017432
   </td>
   <td>0.049992
   </td>
   <td>0.009601
   </td>
  </tr>
  <tr>
   <td><strong>Lincoln pharma</strong>
   </td>
   <td>0.032083
   </td>
   <td>0.086568
   </td>
   <td>0.172592
   </td>
  </tr>
  <tr>
   <td><strong>State bank of India</strong>
   </td>
   <td>0.027805
   </td>
   <td>0.109243
   </td>
   <td>0.209761
   </td>
  </tr>
  <tr>
   <td><strong>Tata Power</strong>
   </td>
   <td>0.149399
   </td>
   <td>0.097192
   </td>
   <td>0.186621
   </td>
  </tr>
  <tr>
   <td><strong>Thirumalai Chemicals Ltd</strong>
   </td>
   <td>0.025025
   </td>
   <td>0.050155
   </td>
   <td>0.074098
   </td>
  </tr>
</table>

#### Comparison of annual expected returns, annual volatility and Sharpe ratio (As obtained on 18th May 2022)

This table shows the annual expected returns, annual volatility and Sharpe ratio that the investors can expect if they optimize this portfolio with the weights allocated by the respective models (as shown in the previous table).

<table>
  <tr>
   <td>
   </td>
   <td><strong>DL model</strong>
   </td>
   <td><strong>HRP</strong>
   </td>
   <td><strong>HERC</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Annual expected returns (in %)</strong>
   </td>
   <td>40.16
   </td>
   <td>35.63
   </td>
   <td>30.88
   </td>
  </tr>
  <tr>
   <td><strong>Annual volatility (in %)</strong>
   </td>
   <td>28.55
   </td>
   <td>27.52
   </td>
   <td>29.74
   </td>
  </tr>
  <tr>
   <td><strong>Sharpe ratio</strong>
   </td>
   <td>1.41
   </td>
   <td>1.29
   </td>
   <td>1.04
   </td>
  </tr>
</table>

From this table, it is evident that the DL model weight allocations correspond to the highest Sharpe ratio. This is because the DL model only tries to maximize the Sharpe ratio and not provide optimal allocation to all assets, whereas the other two statistical models diversify the risk among all assets while trying to keep the Sharpe ratio as optimal as possible.


#### Score from fundamentals analyser (As obtained on 18th May 2022)

This table shows the fundamentals analysis score for each stock in the portfolio. The scores are allocated based on various metrics that provide insights on how fundamentally strong the stock is.


<table>
  <tr>
   <td>
   </td>
   <td><strong>Fundamentals analysis score</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Aarti Drugs</strong>
   </td>
   <td>9.24
   </td>
  </tr>
  <tr>
   <td><strong>APL Apollo Tubes</strong>
   </td>
   <td>6.94
   </td>
  </tr>
  <tr>
   <td><strong>Birlasoft</strong>
   </td>
   <td>6.51
   </td>
  </tr>
  <tr>
   <td><strong>Coforge</strong>
   </td>
   <td>5.84
   </td>
  </tr>
  <tr>
   <td><strong>Dhampur Sugar Mills</strong>
   </td>
   <td>7.01
   </td>
  </tr>
  <tr>
   <td><strong>ICICI Securities</strong>
   </td>
   <td>7.78
   </td>
  </tr>
  <tr>
   <td><strong>Lincoln pharma</strong>
   </td>
   <td>7.77
   </td>
  </tr>
  <tr>
   <td><strong>State bank of India</strong>
   </td>
   <td>6.16
   </td>
  </tr>
  <tr>
   <td><strong>Tata Power</strong>
   </td>
   <td>4
   </td>
  </tr>
  <tr>
   <td><strong>Thirumalai Chemicals Ltd</strong>
   </td>
   <td>6.5
   </td>
  </tr>
</table>

\* Assets with higher scores are considered to be fundamentally stronger compared to the ones with lower scores.


#### Final output (As obtained on 18th May 2022)

This table shows the final weights that are obtained after the final output generator performs calculations for optimizing the portfolio even further based on the fundamentals analysis scores.


<table>
  <tr>
   <td>
   </td>
   <td><strong>Final weights</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Aarti Drugs</strong>
   </td>
   <td>0.136134
   </td>
  </tr>
  <tr>
   <td><strong>APL Apollo Tubes</strong>
   </td>
   <td>0.148824
   </td>
  </tr>
  <tr>
   <td><strong>Birlasoft</strong>
   </td>
   <td>0.062836
   </td>
  </tr>
  <tr>
   <td><strong>Coforge</strong>
   </td>
   <td>0.121268
   </td>
  </tr>
  <tr>
   <td><strong>Dhampur Sugar Mills</strong>
   </td>
   <td>0.120757
   </td>
  </tr>
  <tr>
   <td><strong>ICICI Securities</strong>
   </td>
   <td>0.057508
   </td>
  </tr>
  <tr>
   <td><strong>Lincoln pharma</strong>
   </td>
   <td>0.180107
   </td>
  </tr>
  <tr>
   <td><strong>State bank of India</strong>
   </td>
   <td>0.035320
   </td>
  </tr>
  <tr>
   <td><strong>Tata Power</strong>
   </td>
   <td>0.104707
   </td>
  </tr>
  <tr>
   <td><strong>Thirumalai Chemicals Ltd</strong>
   </td>
   <td>0.032540
   </td>
  </tr>
</table>


This table shows the annual expected returns, annual volatility and Sharpe ratio that the investors can expect if they optimize this portfolio with the weights shown in the above table.


<table>
  <tr>
   <td><strong>Annual expected returns (in %)</strong>
   </td>
    <td><i>34.19</i>
   </td>
  </tr>
  <tr>
   <td><strong>Annual volatility (in %)</strong>
   </td>
    <td><i>27.52</i>
   </td>
  </tr>
  <tr>
   <td><strong>Sharpe ratio</strong>
   </td>
    <td><i>1.24</i>
   </td>
  </tr>
</table>


From this table, we can see that the expected returns and Sharpe ratio corresponding to the final weight allocation is slightly lesser than the expected returns and Sharpe ratios corresponding to the weight allocations by individual models. But the individual models do not take into account the fundamentals of the stocks in the portfolio, thereby making them more vulnerable to losses. 


## Reference Links

### Site links
- [Portfolio Optimization methods](https://blog.quantinsti.com/portfolio-optimization-methods/)

### YouTube links
- [Battle of Portfolio Optimization methods](https://www.youtube.com/watch?v=GW1PASCDOLM&feature=youtu.be)  
- [Portfolio Optimization using Python](https://www.youtube.com/watch?v=xagKMaTjxjk)
- [What is LSTM? Simple explanation of LSTM](https://www.youtube.com/watch?v=LfnrRPFhkuY)
