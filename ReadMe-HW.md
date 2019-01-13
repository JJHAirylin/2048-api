# Read Me 代码与模型说明
## 2048-api/

    data_collect.py:产生训练集训练数据，保存在csv文件里
	
    model_get64_5.h5：game.score<64分时使用的模型权重（version:5.0)
	
    model_get128_1.h5：64分<=game.score<128分时使用的模型权重（version:1.0)
	
    model_get256_2.h5：128分<=game.score<256分时使用的模型权重（version:2.0)
	
    model_get512_2.h5：256分<=game.score<512分时使用的模型权重（version:2.0)
	
    model_get1024.h5：512分<=game.score<1024分时使用的模型权重
	
## game2048/

    models.py:深度神经网络模型定义
	
	load.py:一次性导入五个模型的权重并保存
	
    agents.py:LearnAgent的定义，
	          predict()函数使用训练好的深度学习模型决策移动方向
			  
	getdata.py:包括将board转换为One-hot Board棋盘的函数
	           和从csv文件里读取训练数据的函数
			   
	continue_train.py:训练模型的函数
	
	game.py:增加了将board转换为One-hot Board棋盘的函数
	

  
  