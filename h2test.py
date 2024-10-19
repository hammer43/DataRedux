# Modify your code to use these initialization settings:
import h2o
h2o.init(port=54323)
print(h2o.cluster().show_status())
try:
    h2o.init(port=54323,  # Try a different port
             start_h2o=True,  
             nthreads=-1,  # Use all CPU threads
             min_mem_size="1G",  # Minimum memory
             max_mem_size="2G",  # Maximum memory
             enable_assertions=False,
             bind_to_localhost=True)
except Exception as e:
    print(f"Error initializing H2O: {e}")