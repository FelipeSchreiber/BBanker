# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file if it exists
COPY requirements.txt* ./

# Install dependencies
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy application code
COPY . .

# Expose port (adjust as needed)
EXPOSE 7860

# Command to run the application
CMD ["python", "portfolio_agent.py"]
