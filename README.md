# Visualization_and_Catch_feature_in_industry
在產線機台變數上實現outlier pattern抓取，以及將機台參數視覺化

根據不同batch進行切分，將機台原始數值列在背景，SMA為Moving Average線，黑色虛線為前80%batch其90%上下界，紅點為後20%超過黑色虛線的資料點。

![image](https://github.com/ShaoTingHsu/Visualization_and_Catch_feature_in_industry/blob/main/Pictures/Visualization_1.png)

其中以DBSCAN分群法抓出較為不同pattern的batch。

![image](https://github.com/ShaoTingHsu/Visualization_and_Catch_feature_in_industry/blob/main/Pictures/Visualization_2.png)

