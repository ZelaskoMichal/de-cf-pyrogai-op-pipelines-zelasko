ARG PYTHON_BASE_IMAGE

# Build the adapter image
FROM ${PYTHON_BASE_IMAGE}

# Define build arguments and environment variables
ARG MODEL_PATH
ENV MODEL_PATH=${MODEL_PATH}
ARG ADAPTER_PATH

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY ${ADAPTER_PATH}/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ${ADAPTER_PATH}/main.py /app/
COPY ${ADAPTER_PATH}/infrastructure /app/infrastructure

# Get model(s)
RUN mkdir -p /app/models
COPY ${MODEL_PATH}/* /app/models

# Set entrypoint and default command
ENTRYPOINT ["dockerentrypoint"]
CMD ["/app/main.py"]