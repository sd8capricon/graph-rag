get_triplets_by_community_id = """//cypher
// Get triplets by community id
WITH "f2c121e4969c439791f572ba029e8676_9" AS community_id

// Get the community node
MATCH (c:Community {id: community_id})<-[:IN_COMMUNITY]-(e)
WITH c, collect(e) AS community_entities

// Get relationships among entities inside the same community
MATCH (src)-[r]->(tgt)
WHERE src IN community_entities
  AND tgt IN community_entities
  AND type(r) <> 'IN_COMMUNITY'

WITH c, src, tgt, r
ORDER BY src.id, type(r), tgt.id

WITH
  c,
  collect({
    source: {
      labels: labels(src),
      properties: apoc.map.clean(properties(src), ['community_id'], [])
    },
    relationship: type(r),
    target: {
      labels: labels(tgt),
      properties: apoc.map.clean(properties(tgt), ['community_id'], [])
    }
  }) AS triplets

RETURN {
  id: c.id,
  triplets: triplets
} AS result
"""
