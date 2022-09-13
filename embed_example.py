import os

import numpy as np
import pandas as pd
import plotly.express as px
import requests
from loguru import logger
from sklearn.manifold import TSNE

headers = {"secret": os.environ.get("TEXT_GENERATOR_SECRET")}
texts_to_embed = [
    "def factorial(n):\n\tif n == 0:\n    \treturn 1\n\treturn factorial(n - 1) * n\n",
    "write a function to return factorial of a number",
    "write a function to print a number twice",
    "def print_twice(x):\n\tprint(x)\n\tprint(x)\n",
    "electrical testing of a switchboard with hand holding a red wire",
    "cat and dog laying on the floor",
    "https://images2.minutemediacdn.com/image/upload/c_fill,w_1080,ar_16:9,f_auto,q_auto,g_auto/shape%2Fcover%2Fsport%2F516438-istock-637689912-981f23c58238ea01a6147d11f4c81765.jpg",
    "https://static.text-generator.io/static/img/Screenshot%20from%202022-09-12%2010-08-50.png",
]
embeddings = []
for text in texts_to_embed:

    data = {
        "text": text,
        "num_features": 230,
    }

    response = requests.post(
        "https://api.text-generator.io/api/v1/feature-extraction", json=data, headers=headers
    )

    json_response_list = response.json()  # the embedding is a list of numbers
    embeddings.append(json_response_list)
    logger.info(embeddings)


# could also choose to embed here using PCA
# from sklearn.decomposition import PCA
# two_dim = PCA(random_state=0).fit_transform(np.array(embeddings))[:,:2]

two_dim = TSNE(
    n_components=3, random_state=0, perplexity=0, learning_rate="auto", n_iter=250
).fit_transform(np.array(embeddings))

df = pd.DataFrame(
    data={
        "x": list(map(lambda embed: embed[0], two_dim)),
        "y": list(map(lambda embed: embed[1], two_dim)),
        "hover_data": texts_to_embed,
    }
)
fig = px.scatter(df, x="x", y="y", hover_data=["hover_data"])
fig.show()
fig.write_html("embed_example2.html")
