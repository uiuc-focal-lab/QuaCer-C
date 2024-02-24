# from utils import *

# dbpedia_reader = DBPediaReader(max_threads=6, request_timeout=10)
# graph = dbpedia_reader.run("IMDb", "IMDb", max_depth=10, save_file='imdb_tiny.json')
# print(len(graph))
# print(graph.keys())
# graph = load_graph('imdb_tiny.json')
# # # print(graph.keys())
# graph_algos = GraphAlgos(graph)
# # diameter = graph_algos.compute_diameter()
# # print(diameter)
# # print(graph_algos.dfs_path("Chandler_Bing", 3))
# # some_path = graph_algos.dfs_path("Chandler_Bing", 8)
# # print(some_path)
# # # # print(graph_algos.get_relations())
# # print(f"{some_path[0]} -> {graph_algos.get_relation_for_vertex(some_path[0], some_path[1])} -> {some_path[1]}")
# # print(f"{some_path[1]} -> {graph_algos.get_relation_for_vertex(some_path[1], some_path[2])} -> {some_path[2]}")
# # print(f"query: {graph_algos.generate_query_for_vertices(some_path[0], some_path[-3], k=5)}")
# for i in range(20):
#     print(f"query: {graph_algos.generate_random_query(k=5)}")
# wiki_text = WikiText('imdb_tiny_documents')
# for vertex in graph_algos.get_vertices():
#     wiki_text.save_wiki_text(vertex)

from statsmodels.stats.proportion import proportion_confint

# results = [40, 38, 44, 30, 28, 48, 35]
results1 = [0.6, 0.62, 0.64, 0.68, 0.56, 0.54]
# for result in results:
#     print(proportion_confint(result, 100, method='beta'))

for result in results1:
    result_num = int(result * 1000)
    print(proportion_confint(108, 182, method='beta'))