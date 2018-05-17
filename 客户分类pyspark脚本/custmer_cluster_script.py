'''
This script aim to cluster custmers into 6 gorup using pyspark. 
We focus on the pipeline of the whole process rather than the algorithm.
Three steps schema is designed according to the structure of Sino bigdata system as follows:

1. Read data from sql server 
2. Appling model to dataframe
3. Write result to mysql 

'''
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from datetime import datetime
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans

if __name__ == '__main__':
    conf = SparkConf().setAppName("app")
    sc = SparkContext(conf=conf)
    sqlsc = SQLContext(sc)
    ### 1. Read from sql server
    mssql_df = sqlsc.read.format('jdbc').options(url='jdbc:sqlserver://10.144.129.20:1433;databaseName=FCS_DB;user=card;password=card_123456',dbtable='PORTAL_MEMBER').load() # read from sql server

    mssql_df.cache() # cache to offline analysis

    ### 2. Appling model to dataframe
    ### Data Preprocing
    df = mssql_df.select("pc_card_no","oil_qty","amount","last_trade_time")
    today = datetime.today()
    func = udf(lambda x:(today-datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
    df_recent = df.withColumn('R', func(df.last_trade_time).cast(IntegerType())) # create the recentency column

    df_RFM = df_recent.drop('last_trade_time') # drop useless data
    df_filter = df_RFM.filter(df_RFM.R > 0).filter(df_RFM.amount > 0) # filter out outliers
    drop_na_df = df_filter.na.drop() # drop nas

    featuresUsed = ['oil_qty','amount','R']
    assembler = VectorAssembler(inputCols = featuresUsed, outputCol ="features_unscaled")
    assembled = assembler.transform(drop_na_df) # feature vector used for clustering

    scaler = StandardScaler(inputCol = "features_unscaled", outputCol = "features", withStd =True, withMean =True)
    scaleModel = scaler.fit(assembled)
    scaleData = scaleModel.transform(assembled) # Standerlize the data

    ### Model
    scaledDataFeat = scaleData.select('features')
    scaledDataFeat.persist() # persist in memory to speed up the algorithm

    kmeans = KMeans(k=6, seed =1)
    model =kmeans.fit(scaledDataFeat)
    prediction = model.transform(scaledDataFeat)

    ### Result
    custmer_list = scaleData.select('pc_card_no').collect() #custmer id list
    pred_list = prediction.select('prediction').collect()# prediction list
    result_tuples  = [(custmer_list[i][0], pred_list[i][0]) for i in range(len(custmer_list))] # pair custmer id and predidction to produce the result tuples
    result = sqlsc.createDataFrame(result_tuples, schema=["ids", "cluster"]) # Shape the result to dataframe 

    ### 3. Write result to mysql 
    result.write.format('jdbc').options(url='jdbc:mysql://10.144.131.241:5010/report',user='root',password='F6xWfzUAPef8maPD',dbtable='test').mode('append').save() # save into mysql 



