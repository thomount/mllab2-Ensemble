使用说明：
	//需要手动建立data,input,output文件夹并将train.csv和test.csv放在input中

	cd src
	python preprocess.py
	python main.py -bagging/-ada -svm/-dt -t <turns> -b <branch_size>

先通过预处理将原始数据处理成向量（较慢），再执行集成学习。输出结果在output/predict.csv中


学号: 2017011315
Kaggle ID: thomount
邮箱: tjpby@163.com
手机: 13132034474
