# tensorflow
se 讲的是cnn过程中维度的变化  ###
Iri_one 是对Iri数据集进行最简单的x*w+b的预测  #####
batch 和楼上一样 ，关键在于 next batch 的构造，以及 label 的输入最好用自己的one-hot的函数，用 tf自带的会出现莫名的问题。  ###
