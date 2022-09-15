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

labels_for_graph = [
    "factorial code",
    "factorial prompt",
    "printing prompt",
    "printing code",
    "electrical description",
    "cat and dog",
    "image of cat and dog",
    "image of electrical",
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

small_embed = TSNE(
    n_components=3, random_state=0, perplexity=0, learning_rate="auto", n_iter=250
).fit_transform(
    np.array(embeddings)
)  # takes .15s for 250k features .03s for 2.5k

df = pd.DataFrame(
    data={
        "x": list(map(lambda embed: embed[0], small_embed)),
        "y": list(map(lambda embed: embed[1], small_embed)),
        "hover_data": texts_to_embed,
    }
)
# 2d plot
fig = px.scatter(df, x="x", y="y", hover_data=["hover_data"])
fig.show()
fig.write_html("embed_example2.html")


import numpy as np
from datetime import datetime
from datetime import timedelta

# could also choose to embed here using PCA
from sklearn.decomposition import PCA

start_time = datetime.now()
# small_embed = PCA(random_state=0).fit_transform(np.array(embeddings))[:,:2]

end_time = datetime.now()
print(f"TSNE time taken {end_time - start_time}")
print(small_embed)
df = pd.DataFrame(
    data={
        "x": list(map(lambda embed: embed[0], small_embed)),
        "y": list(map(lambda embed: embed[1], small_embed)),
        "z": list(map(lambda embed: embed[2], small_embed)),
        "hover_data": labels_for_graph,
    }
)
fig = px.scatter_3d(df, x="x", y="y", z="z", hover_data=["hover_data"])
fig.show()
fig.write_html("embed_example8.html", include_plotlyjs=False)


from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean


def m_euclid(v1, v2):
    return euclidean(v1, v2)


dist_list = []
for j1 in embeddings:
    dist_list.append([m_euclid(j1, j2) for j2 in embeddings])
dist_matrix = pd.DataFrame(dist_list)
dist_matrix.columns = labels_for_graph
print(dist_matrix)
fig2 = px.imshow(dist_matrix, y=labels_for_graph)
fig2.show()
