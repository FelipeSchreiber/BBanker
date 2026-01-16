import io
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import base64
from io import BytesIO


def load_returns_from_file(file_path, sheet_name=None):
    """
    Load returns data from CSV or ODS file.
    Assumes first column is dates and rest are asset returns.
    
    Parameters:
    - file_path: Path to the file
    - sheet_name: For ODS files, specify the sheet name (default: 'log_returns' or first sheet)
    """
    try:
        # Check file extension
        if file_path.endswith('.ods'):
            # Default to 'log_returns' sheet if not specified
            if sheet_name is None:
                sheet_name = 'log_returns'
            try:
                returns_df = pd.read_excel(file_path, engine='odf', sheet_name=sheet_name)
            except:
                # If sheet doesn't exist, try first sheet
                returns_df = pd.read_excel(file_path, engine='odf')
        else:
            # Try to read with semicolon delimiter for CSV
            try:
                returns_df = pd.read_csv(file_path, delimiter=";")
            except:
                # Try with comma delimiter
                returns_df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")
    
    # Get returns (all columns except first which is dates)
    returns = returns_df.iloc[:, 1:].copy()
    
    # Check if dataframe is already in float format, if not convert it
    if not all(returns.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        # Replace commas with periods and convert to float
        for col in returns.columns:
            returns.loc[:, col] = returns.loc[:, col].astype(str).str.replace(',', '.')
        returns = returns.apply(pd.to_numeric, errors='coerce')
    
    return returns


def calculate_optimal_portfolio(returns, risk_free_rate):
    """
    Calculate optimal portfolio weights that maximize Sharpe ratio.
    
    Parameters:
    - returns: DataFrame of asset returns
    - risk_free_rate: Annual risk-free rate (e.g., 0.05 for 5%)
    
    Returns:
    - Dictionary with optimal weights, return, volatility, and Sharpe ratio
    """
    def neg_sharpe_ratio(weights, returns, risk_free_rate):
        portfolio_return = np.sum(returns.mean() * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_std_dev
        return -sharpe
    
    num_assets = returns.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_weights = np.array(num_assets * [1. / num_assets])
    
    optimized_results = minimize(
        neg_sharpe_ratio, 
        initial_weights, 
        args=(returns, risk_free_rate),
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    optimal_weights = optimized_results.x
    optimal_portfolio_return = np.sum(returns.mean() * optimal_weights) * 252
    optimal_portfolio_std_dev = np.sqrt(np.dot(optimal_weights.T, np.dot(returns.cov() * 252, optimal_weights)))
    max_sharpe_ratio = (optimal_portfolio_return - risk_free_rate) / optimal_portfolio_std_dev
    
    # Create weights dictionary
    weights_dict = {}
    for i, asset in enumerate(returns.columns):
        if optimal_weights[i] > 0.001:  # Only show assets with >0.1% allocation
            weights_dict[asset] = optimal_weights[i]
    
    return {
        'weights': weights_dict,
        'annual_return': optimal_portfolio_return,
        'annual_volatility': optimal_portfolio_std_dev,
        'sharpe_ratio': max_sharpe_ratio
    }


def portfolio_volatility(weights, covariance_matrix):
    """Calculate portfolio volatility."""
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))


def minimize_volatility(target_return, expected_returns, covariance_matrix):
    """Minimize volatility for a given target return."""
    num_assets = len(expected_returns)
    initial_weights = np.array(num_assets * [1. / num_assets])
    
    constraints = (
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        {'type': 'eq', 'fun': lambda weights: np.sum(weights * expected_returns) - target_return}
    )
    
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    result = minimize(
        portfolio_volatility, 
        initial_weights, 
        args=(covariance_matrix,),
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    return result.fun


def generate_efficient_frontier(returns, risk_free_rate=None):
    """
    Generate efficient frontier plot.
    
    Parameters:
    - returns: DataFrame of asset returns
    - risk_free_rate: Optional risk-free rate to mark optimal portfolio
    
    Returns:
    - Base64 encoded image string
    """
    expected_returns = returns.mean() * 252
    covariance_matrix = returns.cov() * 252
    
    min_return = expected_returns.min()
    max_return = expected_returns.max()
    target_returns = np.linspace(min_return, max_return, 100)
    
    min_volatilities = []
    for tr in target_returns:
        try:
            min_vol = minimize_volatility(tr, expected_returns, covariance_matrix)
            min_volatilities.append(min_vol)
        except:
            min_volatilities.append(np.nan)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(min_volatilities, target_returns, linestyle='-', marker='o', markersize=3, label='Efficient Frontier')
    
    # Mark optimal portfolio if risk_free_rate is provided
    if risk_free_rate is not None:
        optimal = calculate_optimal_portfolio(returns, risk_free_rate)
        plt.scatter(
            optimal['annual_volatility'], 
            optimal['annual_return'], 
            color='red', 
            s=200, 
            marker='*', 
            label=f'Optimal Portfolio (Sharpe: {optimal["sharpe_ratio"]:.2f})',
            zorder=5
        )
    
    plt.xlabel('Annual Volatility (Standard Deviation)')
    plt.ylabel('Annual Expected Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64


def format_portfolio_results(optimal_portfolio):
    """Format portfolio optimization results as readable text."""
    result_text = "### Optimal Portfolio Allocation\n\n"
    result_text += f"**Annual Return:** {optimal_portfolio['annual_return']*100:.2f}%\n"
    result_text += f"**Annual Volatility:** {optimal_portfolio['annual_volatility']*100:.2f}%\n"
    result_text += f"**Sharpe Ratio:** {optimal_portfolio['sharpe_ratio']:.4f}\n\n"
    result_text += "**Asset Allocation:**\n"
    
    for asset, weight in sorted(optimal_portfolio['weights'].items(), key=lambda x: x[1], reverse=True):
        result_text += f"- {asset}: {weight*100:.2f}%\n"
    
    return result_text
