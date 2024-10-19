import h2o
h2o.init(port=54321)
print(h2o.cluster().show_status())