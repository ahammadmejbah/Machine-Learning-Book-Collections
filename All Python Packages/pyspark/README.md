# PySpark

## Optimal PySpark Config
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder\
    .config('spark.executor.instances', '17')\
    .config('spark.yarn.executor.memoryOverhead', '4096')\
    .config('spark.executor.memory', '35g')\
    .config('spark.yarn.driver.memoryOverhead', '4096')\
    .config('spark.driver.memory', '35g')\
    .config('spark.executor.cores', '5')\
    .config('spark.driver.cores', '5')\
    .config('spark.default.parallelism', '170')\
    .getOrCreate()
```