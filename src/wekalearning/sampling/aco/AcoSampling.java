package wekalearning.sampling.aco;

import java.lang.Math;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.io.*;

import weka.core.converters.ConverterUtils.DataSink;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.*;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.lazy.IBk;
import weka.core.neighboursearch.KDTree;

//过采样算法枚举类型
enum OSS {
	SMOTE, // 少数类样本合成过采样
	ROS, // 少数类样本随机过采样
	BSO, // Borderline_SMOTE,少数类样本边界域MOTE
	AQ_BSO, // 量化类边界域的BSO
	ADA_SYN
}

public class AcoSampling {	
	// 工作路径
	private static String workPath = "D:/黎敏工作/不平衡学习";
	// 维护一个闭锁
	private CountDownLatch latch;	
	// 原始样本集合
	private Instances DInstances;
	//小类的样本集合的类别索引
	private int minClass=0;
	private int neigh=5;
	// 过采样算法
	OSS ossStyle;
	// 原始样本集合少数类样本过采样后的样本集合
	private Instances OverDInstances;
	// 样本路径的信息素(每个样本包含选择和不选择两条路径)
	private double[][] pheromone;
	// 根据pheromone计算样本被选择的概率(样本选择路径的信息系除以两条路径信息素之和)，
	private double[] selectedProbs;
	// 分类器的名称
	private String classifierName;
	// 迭代的次数
	private int iterationTimes;
	// 蚂蚁的数量
	private int numbersOfAnt;
	// 蚂蚁群
	private Ant Ants[];
	// 蒸发率
	private double rate;
	// 完成一次迭代得到的全局最优路径（样本选择的向量，如：[0,1,...]表示第1个样本未先中，第2个样本被先中，）
	private int Gtour[];
	// 完成一次迭代得到的全局最优值(根据适应度函数)
	private double GFit = -1.0;
	// 完成iterationTimes次迭代得到的全局最优值(根据适应度函数)
	private double GnFit = -1;
	private double GG_mean=-1;
	private double GnG_mean=-1;
	// 完成iterationTimes次迭代得到的全局最优路径（样本选择的向量，如：[0,1,...]表示第1个样本未先中，第2个样本被先中，）
	private int Gntour[];

	// 完成一次迭代得到的最优样本集合
	private Instances GInstances;
	// 完成iterationTimes次迭代得到的最优样本集合
	private Instances GnInstances;
	// 信息素上限
	private double upperPhero = 2.0;
	// 信息素下限
	private double lowerPhero = 0.5;
	// 蚁群迭代是否结束
	private boolean done = false;
	//KDTre索引，用于KNN分类器
	private KDTree kdtree;
	
	public void setminClass(int minClassIndex){
		minClass=minClassIndex;
	}
	//设置小类索引
	private  int SetminClassIndex(Instances data){
		int minClass=0;
		double min=data.numInstances();		
	    // Figure out how much data there is per class
	    double[] numInstancesPerClass = new double[data.numClasses()];
	    for (Instance instance : data) {
	      numInstancesPerClass[(int)instance.classValue()]++;
	    }

	    //找出最小类别的记数和索引
	    for (int i = 0; i < data.numClasses(); i++) {
	    	if (min> numInstancesPerClass[i])
	    		min=numInstancesPerClass[i];
	    		minClass=i;
	    }
	    return minClass;
	}
	
	AcoSampling(OSS overStyle, String ClsName, Instances train, int iterationTimes, int numbersOfAnt, double rate,int neigh) {
		this.ossStyle = overStyle;
		this.classifierName = ClsName;
		// 初始化样本集合
		this.DInstances = train;

		// 迭代次数
		this.iterationTimes = iterationTimes;
		// 设置蚂蚁数量
		this.numbersOfAnt = numbersOfAnt;
		this.neigh=neigh;
		// 给蚂蚁数组分配空间
		this.Ants = new Ant[numbersOfAnt];

		// 生成过采样样本集合
		doOverSampling(this.ossStyle);
		kdtree=new KDTree(OverDInstances);
		// 给最佳路径数组分配空间
		this.Gtour = new int[OverDInstances.numInstances()];
		// 初始化每个样本被选中的概率
		this.selectedProbs = new double[OverDInstances.numInstances()];

		// 设置蒸发率
		this.rate = rate;
		// 信息素初始化
		Initpheromone();
	}

	// 对少数类样本过采样
	void doOverSampling(OSS oss) {
		switch (oss) {
		case SMOTE:
			doSmote();
			break;
		case ROS:
			doROS();
			break;
		case BSO:
			doBSO();
			break;
		case AQ_BSO:// 量化类概率边界域SMOTE，待黄顺完成自动设置参数版
			doAQ_BSO();
			break;
		case ADA_SYN: // 待黄顺完成自动平衡版
			doADA_SYN();
			break;
		}
	}

	void doSmote() {
		SMOTEPro Overconvert = new SMOTEPro();
		try {
			Overconvert.setClassValue("-1");
			Overconvert.setNearestNeighbors(neigh);
			Overconvert.setInputFormat(DInstances);
			OverDInstances = Filter.useFilter(DInstances, Overconvert);
			/*
			DataSink.write(
					workPath + "/experiments/" + DInstances.relationName() + "(" + this.ossStyle + ").arff",
					OverDInstances);
			*/
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	void doBSO() {
		BSO1Pro Overconvert = new BSO1Pro();
		try {
			Overconvert.setClassValue("-1");
			Overconvert.setNearestNeighbors(neigh);
			Overconvert.setInputFormat(DInstances);
			OverDInstances = Filter.useFilter(DInstances, Overconvert);
			/*
			DataSink.write(
					workPath + "/experiments/" + DInstances.relationName() + "(" + this.ossStyle + ").arff",
					OverDInstances);
			*/
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	void doROS() {
		ResampleofBalance Overconvert = new ResampleofBalance();
		try {
			// SampleSizePercent = -1 : 随机过采样平衡数据集
			// SampleSizePercent = -2 : 随机降采样平衡数据集
			// else : 等价于 sample
			Overconvert.setSampleSizePercent(-1.0);
			Overconvert.setInputFormat(DInstances);
			OverDInstances = Filter.useFilter(DInstances, Overconvert);
			/*
			DataSink.write(
					workPath + "/experiments/" + DInstances.relationName() + "(" + this.ossStyle + ").arff",
					OverDInstances);
			*/
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	void doADA_SYN() {
		ADA_SYN_Pro Overconvert = new ADA_SYN_Pro();
		try {
			Overconvert.setClassValue("-1");
			Overconvert.setNearestNeighbors(neigh);
			Overconvert.setInputFormat(DInstances);
			OverDInstances = Filter.useFilter(DInstances, Overconvert);
			/*
			DataSink.write(
					workPath + "/experiments/" + DInstances.relationName() + "(" + this.ossStyle + ").arff",
					OverDInstances);
			*/
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	void doAQ_BSO() {
		AQ_BSO1_Pro Overconvert = new AQ_BSO1_Pro();
		try {
			Overconvert.setClassValue("-1");
			Overconvert.setNearestNeighbors(neigh);
			Overconvert.setInputFormat(DInstances);
			OverDInstances = Filter.useFilter(DInstances, Overconvert);
			/*
			DataSink.write(
					workPath + "/experiments/" + DInstances.relationName() + "(" + this.ossStyle + ").arff",
					OverDInstances);
			*/
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	// 初始化pheromone矩阵
	void Initpheromone() {
		double inipheromone = 1;
		// 每个样本有选择和不选择两条路径可选择
		pheromone = new double[OverDInstances.numInstances()][2];
		// 给pheronome赋值
		for (int i = 0; i < OverDInstances.numInstances(); i++) {
			for (int j = 0; j < 2; j++)
				// pheromone[i][0]代表第i个样本被选中的信息素，pheromone[i][1]代表第i个样本未被选中的信息素，
				pheromone[i][j] = inipheromone;
		}
		// 初始每个样本选中边的概率
		for (int i = 0; i < OverDInstances.numInstances(); i++) {
			selectedProbs[i] = (pheromone[i][0]) / (pheromone[i][0] + pheromone[i][1]);
		}
	}

	// 蚂蚁（内部类）
	class Ant {
		// 样本选择的向量，如：[0,1,...]表示第1个样本未先中，第2个样本被先中，
		private int[] tour;
		// 选择的样本集合
		private Instances antInstances;
		// 蚂蚁找出的路径的适应度函数值
		private double fit = -1;
		//蚂蚁找出的路径的G-mean值
		private double fG_mean=-1;
		// 蚂蚁的分类器
		private Classifier antcls;

		Ant() {
			// 初始化蚂蚁选择的样本选择向量
			tour = new int[OverDInstances.numInstances()];
			// 初始化选择的样本集合，生成一个与DInstances同构的空的样本集合
			antInstances = new Instances(OverDInstances, 0);
			try {
				antcls = (Classifier) Class.forName(classifierName).newInstance();
				if (classifierName == "weka.classifiers.functions.LibSVM") {
					((LibSVM) antcls).setProbabilityEstimates(true);
					((LibSVM) antcls).setNormalize(true);

				} else if (classifierName == "weka.classifiers.bayes.NaiveBayes") {
					((NaiveBayes) antcls).setUseKernelEstimator(true);

				} else if (classifierName == "weka.classifiers.lazy.IBk") {
					((IBk) antcls).setKNN(3);
					((IBk) antcls).setNearestNeighbourSearchAlgorithm(new KDTree(OverDInstances));

				}
			
			} catch (Exception e) {

			}
		}

		// 构造蚂蚁的样本选择向量,赋值选择的样本集合
		private void constructTour() {
			double rs = 0;// 轮盘赌产生的随机数
			for (int i = 0; i < selectedProbs.length; i++) {
				rs = Math.random();
				// 按轮盘赌方式确定是否选择样本
				if (rs <= selectedProbs[i]) {// 选择当前样本
					tour[i] = 1;
					antInstances.add(OverDInstances.instance(i));
				} else {// 不选择当前样本
					tour[i] = 0;
				}
			}
		}

		// 评估蚂蚁选择的样本集合，计算适应度函数值
		private void evaluateTour() throws Exception {
        	double wfit=0.0;
        	double g_mean=0.0;
    		Evaluation Eval=new Evaluation(antInstances);		//评估器初始化
			//用选出的样本集合训练分类器
			antcls.buildClassifier(antInstances);
			//用原始数据对分类器进行测度
			Eval.evaluateModel(antcls, DInstances);	
						
			//计算G-mean
			if (!Double.isNaN(Eval.truePositiveRate(0))&&!Double.isNaN(Eval.truePositiveRate(1))){
			   g_mean=Math.pow((Eval.truePositiveRate(0)*Eval.truePositiveRate(1)),0.5);
			   fG_mean=g_mean;
			   //fit=g_mean;
			}
			
			/*
			int nclass=DInstances.numClasses();
			//计算平均F-measure作为适应度值
			for (int i=0;i<nclass;i++){
				if (!Double.isNaN(Eval.fMeasure(i)))
				wfit=wfit+Eval.fMeasure(i);
			}
			fit=wfit/nclass;
			*/
			
			//fit=g_mean;
			
			/*
			if (!Double.isNaN(Eval.fMeasure(minClass)))
				fit=(1.0/3)*Eval.areaUnderROC(0)+(1.0/3)*g_mean+(1.0/3)*((Eval.fMeasure(0)+Eval.fMeasure(1))/2);
			else
				fit=0;
			*/
						
			fit=Eval.areaUnderROC(0);

			//an approximation measure of AUC
			//fit=(1.0+Eval.truePositiveRate(0)-Eval.falsePositiveRate(0))/2.0;

	}
	}

	// 单只蚂蚁要做的工作放这里，这是一个(内部类)
	class AntThread extends Thread {
		// 需要传入一只蚂蚁
		private Ant ant;
		public AntThread(Ant ant) {
			this.ant = ant;
		}
		@Override
		public void run() {
			ant.constructTour();// 蚂蚁构建路径，找出一个对应的样本集合
			try { // evaluateTour()这个方法往上抛了一个异常，run()这里好像不能往上抛，那就捕获一下
				ant.evaluateTour();// 蚂蚁对找出的路径进行评估
			} catch (Exception e) {
				e.printStackTrace();
			}
			// 线程结束，告诉闭锁
			latch.countDown();
		}
	}

	// 所有蚂蚁完成一次旅行后，找到本次最优路径
	private void SelectbestPath() {
		int i, j;
		int t = -1;
		double bestfit = -1;
		double bestG_mean=-1;
		Instances bestInstances = null;
		// 遍历所有蚂蚁，找到本次蚁群的最佳路径
		for (i = 0; i < numbersOfAnt; i++) {
			if (Ants[i].fit > bestfit) {
				bestfit = Ants[i].fit;
				bestG_mean=Ants[i].fG_mean;
				bestInstances = Ants[i].antInstances;
				t = i;
			} else if (Ants[i].fit < bestfit) {

			} else {// 相等的情况
				/*
				if (bestInstances.numInstances() < Ants[i].antInstances.numInstances()) {
					bestfit = Ants[i].fit;
					bestInstances = Ants[i].antInstances;
					t = i;
				}
				*/
				
				if (bestG_mean < Ants[i].fG_mean) {
					bestfit = Ants[i].fit;
					bestG_mean=Ants[i].fG_mean;
					bestInstances = Ants[i].antInstances;
					t = i;
				}
				
			}
		}
		GFit = bestfit;
		// if (t!=-1){
		GG_mean=Ants[t].fG_mean ;
		Gtour = Ants[t].tour; // 保存最佳路径
		GInstances = Ants[t].antInstances; // 保存一次蚁群迭代得到的最优采样样本集合
		// }
	}

	// 根据最优的路径更新信息素
	private void UpdatePheromone() {
		// 计算信息素的增加量
		// double delta=GFit/(0.1*numbersOfAnt);
		double delta = GFit / (0.1 * 50);
		// 对每个样本的两条边的Pheromone进行更新
		for (int i = 0; i < OverDInstances.numInstances(); i++) {
			if (Gtour[i] == 1) {// 若第i个样本被选中,选择的边信息素得到增强，不选择的边信息素减弱
				pheromone[i][0] = Math.min(rate * pheromone[i][0] + delta, upperPhero);
				pheromone[i][1] = Math.max(rate * pheromone[i][1], lowerPhero);
			} else {// 若第i个样本未被选中,选择的边信息素得到减弱，不选择的边信息素增强
				pheromone[i][0] = Math.max(rate * pheromone[i][0], lowerPhero);
				pheromone[i][1] = Math.min(rate * pheromone[i][1] + delta, upperPhero);
			}

		}
		// 更新每个样本选中边的概率
		for (int i = 0; i < OverDInstances.numInstances(); i++) {
			selectedProbs[i] = (pheromone[i][0]) / (pheromone[i][0] + pheromone[i][1]);
		}
	}

	public void go() throws Exception {
		int i, j = 1;
		GnFit = 0;

		// 搞一个线程池
		ThreadPoolExecutor threadPool = (ThreadPoolExecutor) Executors.newFixedThreadPool(12);
		
		while (j <= iterationTimes) {
			// 每次迭代都重新年初始化新的一批蚂蚁
			for (i = 0; i < numbersOfAnt; i++) {
				Ants[i] = new Ant();
			}

			// 上闭锁，不放过一只小蚂蚁
			latch = new CountDownLatch(numbersOfAnt);

			// 进行一次迭代（即让所有的蚂蚁构建一条路径）
			for (i = 0; i < numbersOfAnt; i++) {
				// 把小蚂蚁扔给线程池
				threadPool.execute(new AntThread(Ants[i]));
			}

			// 暂时阻塞主线程，直到所有小蚂蚁跑完
			latch.await();
			
			// 找到本次抚迭代的最优路径
			SelectbestPath();

			// 完成一次迭代后更新信息素
			UpdatePheromone();

			if (GnFit < GFit) {
				GnFit = GFit;
				GnG_mean=GG_mean;
				Gntour = Gtour;
				GnInstances = GInstances;
			} else if (GnFit > GFit) {

			} else {// 如果此次的最优适应度与前面的最优值相等
				/*
				if (GnInstances.numInstances() < GInstances.numInstances()) {
					GnFit = GFit;
					Gntour = Gtour;
					GnInstances = GInstances;
				}
				*/
				if (GG_mean > GnG_mean) {
					GnFit = GFit;
					GnG_mean=GG_mean;
					Gntour = Gtour;
					GnInstances = GInstances;
				}
			}

			j++;
		}
		
		// 关闭线程池
		threadPool.shutdown();


		System.out.println("best fitness： " + GnFit);

		done = true;
	}


	//得到过采样后再降采样的最佳集合
	public Instances getBestInstances() throws Exception {
		if (done)
			return GnInstances;
		else {
			go();
			return GnInstances;
		}
	}

	public Instances getSmoteInstances()  {
		if (OverDInstances == null)
			doOverSampling(this.ossStyle);
		
		return OverDInstances;

	}

	public Instances getOverInstances()   {
		if (OverDInstances == null)
			doOverSampling(this.ossStyle);
		
			return OverDInstances;

	}

	public static void main(String[] args) throws Exception {
		//String clsName = "weka.classifiers.trees.J48";
		// String clsName="weka.classifiers.bayes.NaiveBayes";
		 String clsName="weka.classifiers.functions.LibSVM";

		//DataSource source = new DataSource(workPath + "/twoclass-imbalances/ionosphere.arff");

		// =============================================================================================
		 DataSource source = new DataSource(workPath + "/artificial_datasets/imbalance(200-20).arff");
	
		Instances train = source.getDataSet();

		train.setClassIndex(train.numAttributes() - 1);
		AcoSampling acoSampling1 = new AcoSampling(OSS.SMOTE, clsName, train, 100, 50, 0.9,5);
		acoSampling1.setminClass(acoSampling1.SetminClassIndex(train));

		long startTime = System.currentTimeMillis();
		acoSampling1.go();

		long endTime = System.currentTimeMillis();
		System.out.println("running time： "+(endTime-startTime)+"ms");
	}

}