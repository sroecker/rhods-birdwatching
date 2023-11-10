FROM registry.access.redhat.com/ubi8/python-39

# Add application sources with correct permissions for OpenShift
USER 0
ADD streamlit .
RUN chown -R 1001:0 ./
USER 1001

# Install the dependencies
RUN pip install -U "pip>=19.3.1" && \
    pip install -r requirements.txt &&

# Run the application
CMD streamlit run streamlit/app.py
