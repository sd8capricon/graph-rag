expand_chunk = """//cypher
WITH $chunks AS chunk_ids
MATCH (c:Chunk)
WHERE c.id IN chunk_ids
OPTIONAL MATCH (c)-[:SIMILAR]-(similar)
WITH
  collect({ text: c.text, source_id: c.source_id }) +
  collect({ text: similar.text, source_id: similar.source_id }) AS all_chunks
UNWIND all_chunks AS chunk
RETURN collect(DISTINCT chunk) AS chunks
"""
