from utils import *

dbpedia_reader = DBPediaReader(max_threads=15, request_timeout=8)
graph = dbpedia_reader.run("Albert_Einstein", "Albert Einstein", max_depth=5, save_file='graph.json')
print(graph.keys())
graph = load_graph('graph.json')
print(graph.keys())
graph_algos = GraphAlgos(graph)
diameter = graph_algos.compute_diameter()
print(diameter)
print(graph_algos.dfs_path("Albert_Einstein", 3))
some_path = graph_algos.dfs_path("Albert_Einstein", diameter - 3)
print(some_path)
print(graph_algos.get_relations())
print(f"{some_path[0]} -> {graph_algos.get_relation_for_vertex(some_path[0], some_path[1])} -> {some_path[1]}")
print(f"{some_path[1]} -> {graph_algos.get_relation_for_vertex(some_path[1], some_path[2])} -> {some_path[2]}")
wiki_text = WikiText('documents')
for vertex in some_path:
    wiki_text.save_wiki_text(vertex)
    print(f"Saved {vertex}")