import duckdb

con = duckdb.connect()

query = """
SELECT
    'S1' AS source,
    COUNT(*) AS rows
FROM read_parquet('features/normalized/train_source1.parquet')

UNION ALL

SELECT
    'S2' AS source,
    COUNT(*) AS rows
FROM read_parquet('features/normalized/train_source2.parquet')

UNION ALL

SELECT
    'S3' AS source,
    COUNT(*) AS rows
FROM read_parquet('features/normalized/train_source3.parquet')
"""

result = con.execute(query).fetchdf()

print(result.to_string(index=False))