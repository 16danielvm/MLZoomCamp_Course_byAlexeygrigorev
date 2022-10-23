import numpy as np

import bentoml

from bentoml.io import JSON   
from bentoml.io import NumpyNdarray

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")

model_runner = model_ref.to_runner()

svc = bentoml.Service("ml", runners=[model_runner])

@svc.api(input= NumpyNdarray(), output= NumpyNdarray())
def classify(vector):
    # vector = dv.transform(application_data)
    prediction = model_runner.predict.run(vector)
    print(prediction)
    return prediction[0]

