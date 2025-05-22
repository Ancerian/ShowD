import h2o

m = h2o.load_model("123.model")
r = m.predict(h2o.H2OFrame({"x": [1, 2, 3]}))
print(r)
