package wekalearning.sampling.aco;
import java.io.*;
import java.lang.reflect.Array;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;




import jxl.*;

import java.util.*;

import weka.core.Utils;
import jxl.read.biff.BiffException;
import jxl.write.*;
import jxl.write.biff.RowsExceededException;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.evaluation.output.prediction.*;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;

public class AcoSamplingClassify {
	// 
	private static String workPath = "D:/Limin";
	//ants number
	public static int ants=50;
	//iteration number
	public static int iterates=50;
	//number of cross-validation 
	public static int runs=1;
	//number of folds
	public static int folds=10;
	public static double rate=0.95;
	public static int neigh=5;
    //classify name
    private static String clsName="weka.classifiers.trees.J48";
   // private static String clsName="weka.classifiers.bayes.NaiveBayes";
   // private static String clsName="weka.classifiers.lazy.IBk";
    //private static String clsName="weka.classifiers.functions.LibSVM";
    //private static String clsName="weka.classifiers.functions.MultilayerPerceptron";
   
   
	public static String dataFilename="";
	
	public static WritableWorkbook xlsResults;
	public static WritableSheet sheet;
	public static String rstExcel="";
	public static String rstTxt="";
	
	
	private static int SetminClassIndex(Instances data){
		int minClass=0;
		double min=data.numInstances();		
	    // Figure out how much data there is per class
	    double[] numInstancesPerClass = new double[data.numClasses()];
	    for (Instance instance : data) {
	      numInstancesPerClass[(int)instance.classValue()]++;
	    }

	   
	    for (int i = 0; i < data.numClasses(); i++) {
	    	if (min> numInstancesPerClass[i])
	    		min=numInstancesPerClass[i];
	    		minClass=i;
	    }
	    return minClass;
	}
	
	public static void initialxlsResults() throws IOException, RowsExceededException, WriteException{

		xlsResults= Workbook.createWorkbook(new File(rstExcel));

		sheet=xlsResults.createSheet(clsName,0);
		sheet.addCell(new Label(0,0,"Data Set"));
		sheet.addCell(new Label(1,0,"Methods"));
		sheet.addCell(new Label(3,0,"OA"));
		sheet.addCell(new Label(3,0,"majority class"));		
		sheet.addCell(new Label(4,0,"Precision"));	
		sheet.addCell(new Label(5,0,"Recall"));
		sheet.addCell(new Label(6,0,"F-measure"));
		sheet.addCell(new Label(7,0,"minority class"));		
		sheet.addCell(new Label(8,0,"Precision"));	
		sheet.addCell(new Label(9,0,"Recall"));
		sheet.addCell(new Label(10,0,"F-measure"));		
		sheet.addCell(new Label(11,0,"AUC"));
		sheet.addCell(new Label(12,0,"samples"));

	}
	
	public static void sheetAddRow(int row,String dataName,String methodName,Evaluation eval,int samples) throws RowsExceededException, WriteException, IOException, BiffException{
	    
		sheet.addCell(new Label(0,row,dataName));
		sheet.addCell(new Label(1,row,methodName));
		sheet.addCell(new jxl.write.Number(2,row,eval.pctCorrect()));
		sheet.addCell(new jxl.write.Number(4,row,eval.precision(0)));
		sheet.addCell(new jxl.write.Number(5,row,eval.recall(0)));
		sheet.addCell(new jxl.write.Number(6,row,eval.fMeasure(0)));
		sheet.addCell(new jxl.write.Number(8,row,eval.precision(1)));
		sheet.addCell(new jxl.write.Number(9,row,eval.recall(1)));
		sheet.addCell(new jxl.write.Number(10,row,eval.fMeasure(1)));
		sheet.addCell(new jxl.write.Number(11,row,eval.areaUnderROC(1)));
		sheet.addCell(new jxl.write.Number(12,row,samples));
	}
	

	
	public static void main(String[] args) throws Exception {
		SimpleDateFormat sdf = new SimpleDateFormat();
        sdf.applyPattern("yyyyMMddHHmm");
		Date btime= new Date();
		rstExcel="D:\\Limin\\results\\AcoSampling-Results("+clsName+"-neigh="+neigh+")(runs="+runs+"folds="+folds+")("+sdf.format(btime)+").xls";
		initialxlsResults();
		int row=1;
		long startTime = System.currentTimeMillis();
		BufferedReader br1 = new BufferedReader(new FileReader("D:\\Limin\\dataName.txt"));
		dataFilename=br1.readLine();
		while(dataFilename!=null){
			Instances Originaldata=DataSource.read(dataFilename);
			Originaldata.setClassIndex(Originaldata.numAttributes()-1);	
			int minClass= SetminClassIndex(Originaldata);
			
			/*
			ReplaceMissingValues  replace= new ReplaceMissingValues();					
			replace.setInputFormat(Originaldata);
	    	Instances inputData = Filter.useFilter(Originaldata, replace); 
			 */
			Instances inputData=new Instances(Originaldata);
			
			Classifier clsOriginal=(Classifier)Class.forName(clsName).newInstance();	
			Evaluation evalOriginal=new Evaluation(inputData);   //评估器初始化 
			int OriginalCount=inputData.numInstances()*9/10;
			
			//用SMOTE数据折训练分类器
			Classifier clsSmote=(Classifier)Class.forName(clsName).newInstance();	
			Evaluation evalSmote=new Evaluation(inputData);   //评估器初始化 	 
			//用蚁群搜索数据折训练分类器
			Classifier clsAcoSMOTE=(Classifier)Class.forName(clsName).newInstance();
			Evaluation evalAcoSMOTE=new Evaluation(inputData);   //评估器初始化 	
			int SMOTECount=0,AcoSMOTECount=0; 
			
			//用ROS数据折训练分类器
			Classifier clsROS=(Classifier)Class.forName(clsName).newInstance();	
			Evaluation evalROS=new Evaluation(inputData);   //评估器初始化 	 
			//用蚁群搜索数据折训练分类器
			Classifier clsAcoROS=(Classifier)Class.forName(clsName).newInstance();
			Evaluation evalAcoROS=new Evaluation(inputData);   //评估器初始化 	 
			int ROSCount=0,AcoROSCount=0;
		
			//用BSO数据折训练分类器
			Classifier clsBSO=(Classifier)Class.forName(clsName).newInstance();	
			Evaluation evalBSO=new Evaluation(inputData);   //评估器初始化 	 
			//用蚁群搜索数据折训练分类器
			Classifier clsAcoBSO=(Classifier)Class.forName(clsName).newInstance();
			Evaluation evalAcoBSO=new Evaluation(inputData);   //评估器初始化 		
			int BSOCount=0,AcoBSOCount=0;
			
			//用ADA_SYN数据折训练分类器
			Classifier clsADA_SYN=(Classifier)Class.forName(clsName).newInstance();	
			Evaluation evalADA_SYN=new Evaluation(inputData);   //评估器初始化 	 
			//用蚁群搜索数据折训练分类器
			Classifier clsAcoADA_SYN=(Classifier)Class.forName(clsName).newInstance();
			Evaluation evalAcoADA_SYN=new Evaluation(inputData);   //评估器初始化 		
			int ADA_SYNCount=0,AcoADA_SYNCount=0;
			
			/*
			//用AQ_BSO数据折训练分类器
			Classifier clsAQ_BSO=(Classifier)Class.forName(clsName).newInstance();	
			Evaluation evalAQ_BSO=new Evaluation(inputData);   //评估器初始化 	 
			//用蚁群搜索数据折训练分类器
			Classifier clsAcoAQ_BSO=(Classifier)Class.forName(clsName).newInstance();
			Evaluation evalAcoAQ_BSO=new Evaluation(inputData);   //评估器初始化 	
			int AQ_BSOCount=0,AcoAQ_BSOCount=0;
			*/	
			
			if (clsName == "weka.classifiers.functions.LibSVM") {
				((LibSVM) clsOriginal).setProbabilityEstimates(true);
				((LibSVM) clsOriginal).setNormalize(true);

				((LibSVM) clsSmote).setProbabilityEstimates(true);
				((LibSVM) clsSmote).setNormalize(true);
				((LibSVM) clsAcoSMOTE).setProbabilityEstimates(true);
				((LibSVM) clsAcoSMOTE).setNormalize(true);

				((LibSVM) clsROS).setProbabilityEstimates(true);
				((LibSVM) clsROS).setNormalize(true);
				((LibSVM) clsAcoROS).setProbabilityEstimates(true);
				((LibSVM) clsAcoROS).setNormalize(true);

				((LibSVM) clsBSO).setProbabilityEstimates(true);
				((LibSVM) clsBSO).setNormalize(true);
				((LibSVM) clsAcoBSO).setProbabilityEstimates(true);
				((LibSVM) clsAcoBSO).setNormalize(true);

				((LibSVM) clsADA_SYN).setProbabilityEstimates(true);
				((LibSVM) clsADA_SYN).setNormalize(true);
				((LibSVM) clsAcoADA_SYN).setProbabilityEstimates(true);
				((LibSVM) clsAcoADA_SYN).setNormalize(true);

			} else if (clsOriginal.getClass().getName() == "weka.classifiers.bayes.NaiveBayes") {
				((NaiveBayes) clsOriginal).setUseKernelEstimator(true);
				((NaiveBayes) clsSmote).setUseKernelEstimator(true);	
				((NaiveBayes) clsAcoSMOTE).setUseKernelEstimator(true);
				((NaiveBayes) clsROS).setUseKernelEstimator(true);
				((NaiveBayes) clsAcoROS).setUseKernelEstimator(true);
				((NaiveBayes) clsBSO).setUseKernelEstimator(true);
				((NaiveBayes) clsAcoBSO).setUseKernelEstimator(true);
				((NaiveBayes) clsADA_SYN).setUseKernelEstimator(true);
				((NaiveBayes) clsAcoADA_SYN).setUseKernelEstimator(true);
			} else if (clsOriginal.getClass().getName() == "weka.classifiers.lazy.IBk") {
				 int k=3;
				((IBk) clsOriginal).setKNN(k);
				((IBk) clsOriginal).setNearestNeighbourSearchAlgorithm(new weka.core.neighboursearch.KDTree());
				((IBk) clsSmote).setKNN(k);
				((IBk) clsSmote).setNearestNeighbourSearchAlgorithm(new weka.core.neighboursearch.KDTree());
				((IBk) clsAcoSMOTE).setKNN(k);
				((IBk) clsAcoSMOTE).setNearestNeighbourSearchAlgorithm(new weka.core.neighboursearch.KDTree());
				
				((IBk) clsROS).setKNN(k);
				((IBk) clsROS).setNearestNeighbourSearchAlgorithm(new weka.core.neighboursearch.KDTree());
				((IBk) clsAcoROS).setKNN(k);
				((IBk) clsAcoROS).setNearestNeighbourSearchAlgorithm(new weka.core.neighboursearch.KDTree());

				((IBk) clsBSO).setKNN(k);
				((IBk) clsBSO).setNearestNeighbourSearchAlgorithm(new weka.core.neighboursearch.KDTree());
				((IBk) clsAcoBSO).setKNN(k);
				((IBk) clsAcoBSO).setNearestNeighbourSearchAlgorithm(new weka.core.neighboursearch.KDTree());

				((IBk) clsADA_SYN).setKNN(k);
				((IBk) clsADA_SYN).setNearestNeighbourSearchAlgorithm(new weka.core.neighboursearch.KDTree());
				((IBk) clsAcoADA_SYN).setKNN(k);
				((IBk) clsAcoADA_SYN).setNearestNeighbourSearchAlgorithm(new weka.core.neighboursearch.KDTree());
				//((IBk) clsAcoADA_SYN).setNearestNeighbourSearchAlgorithm(weka.core.neighboursearch.KDTree());
			}
			/*else if (clsOriginal.getClass().getName() == "weka.classifiers.trees.J48") {
				((J48) clsOriginal).setUnpruned(true);		
				((J48) clsSmote).setUnpruned(true);	
				((J48) clsAcoSMOTE).setUnpruned(true);	
				((J48) clsROS).setUnpruned(true);	
				((J48) clsAcoROS).setUnpruned(true);
				((J48) clsBSO).setUnpruned(true);	
				((J48) clsAcoBSO).setUnpruned(true);
				((J48) clsADA_SYN).setUnpruned(true);	
				((J48) clsAcoADA_SYN).setUnpruned(true);	
			}
			*/
			PlainText plText = new PlainText();
			StringBuffer sbSmote= new StringBuffer();
			StringBuffer sbAcoSmote= new StringBuffer();	
			StringBuffer sbBSO= new StringBuffer();
			StringBuffer sbAcoBSO= new StringBuffer();
			StringBuffer sbROS= new StringBuffer();
			StringBuffer sbAcoROS= new StringBuffer();
			StringBuffer sbADA_SYN= new StringBuffer();
			StringBuffer sbAcoADA_SYN= new StringBuffer();
			plText.setOutputDistribution(true);
	
			
			//循环runs次，每次folds-cross validation
			for(int r=0;r<runs;r++){
				//每次得得排序不同的数据
				int seed=r*r+10;
				Random rand= new Random(seed);
				Instances randData=new Instances(inputData);
				randData.randomize(rand);  //样本随机化
				randData.stratify(folds);  //产生folds折数据
				
				//folds-cross validation
				for (int n=0;n<folds;n++){
					Instances originalFolds=randData.trainCV(folds, n); //获得训练样本折构成的集合
					Instances test=randData.testCV(folds, n);          //获得测度折构成的集合
				
					//SMOTE采样，ACO寻优
					AcoSampling acoSMOTESampling = new AcoSampling(OSS.SMOTE,clsName,originalFolds,iterates,ants,rate,neigh); 
					acoSMOTESampling.setminClass(minClass);
					//acoSMOTESampling.go();  
					Instances SmoteFolds=acoSMOTESampling.getOverInstances();     //获得训练样本折Smote后构成的集合
					Instances AcoSmoteFolds=acoSMOTESampling.getBestInstances();
					SMOTECount=SMOTECount+SmoteFolds.numInstances();
					AcoSMOTECount= AcoSMOTECount+AcoSmoteFolds.numInstances();
				
					//ROS采样，ACO寻优,
					AcoSampling acoROSSampling = new AcoSampling(OSS.ROS,clsName,originalFolds,iterates,ants,rate,neigh); 
					acoROSSampling.setminClass(minClass);
					acoROSSampling.go();  
					Instances RosFolds=acoROSSampling.getOverInstances();     //获得训练样本折ROS后构成的集合
					Instances AcoRosFolds=acoROSSampling.getBestInstances();
					ROSCount=ROSCount+RosFolds.numInstances();
					AcoROSCount=AcoROSCount+AcoRosFolds.numInstances();
				
					//Borderline_SMOTE采样，ACO寻优
					AcoSampling acoBSOSampling = new AcoSampling(OSS.BSO,clsName,originalFolds,iterates,ants,rate,neigh);  
					acoBSOSampling.setminClass(minClass);
					acoBSOSampling.go();  
					Instances BsoFolds=acoBSOSampling.getOverInstances();     //获得训练样本折BSO后构成的集合
					Instances AcoBsoFolds=acoBSOSampling.getBestInstances();
					BSOCount= BSOCount+BsoFolds.numInstances();
					AcoBSOCount=AcoBSOCount+AcoBsoFolds.numInstances();
									
					//ADA_SYN采样，ACO寻优
					AcoSampling acoADA_SYNSampling = new AcoSampling(OSS.ADA_SYN,clsName,originalFolds,iterates,ants,rate,neigh); 
					acoADA_SYNSampling.setminClass(minClass);
					acoADA_SYNSampling.go();  
					Instances ADA_SYNFolds=acoADA_SYNSampling.getOverInstances();     //获得训练样本折ADA_SYN后构成的集合
					Instances AcoADA_SYNFolds=acoADA_SYNSampling.getBestInstances();	
					ADA_SYNCount=ADA_SYNCount+ADA_SYNFolds.numInstances();
					AcoADA_SYNCount=AcoADA_SYNCount+AcoADA_SYNFolds.numInstances();
					
					/*
					//AQ_BSO采样，ACO寻优
					AcoSampling acoAQ_BSOSampling = new AcoSampling(OSS.AQ_BSO,clsName,originalFolds,iterates,ants,rate,neigh);  
					acoAQ_BSOSampling.setminClass(minClass);
					acoAQ_BSOSampling.go();  
					Instances AQ_BSOFolds=acoADA_SYNSampling.getOverInstances();     //获得训练样本折ROS后构成的集合
					Instances AcoAQ_BSOFolds=acoADA_SYNSampling.getBestInstances();
					AQ_BSOCount= AQ_BSOCount+AQ_BSOFolds.numInstances();
					AcoAQ_BSOCount=AcoAQ_BSOCount+AcoAQ_BSOFolds.numInstances();
					*/
					
					//原始数据折评估
					clsOriginal.buildClassifier(originalFolds);
					evalOriginal.evaluateModel(clsOriginal, test);			
					
					//Smote数据折评估
					clsSmote.buildClassifier(SmoteFolds); 
					plText.setHeader(test);
					plText.setBuffer(sbSmote);
					evalSmote.evaluateModel(clsSmote, test,plText);
					//pTSmote.setOutputFile(new File(workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(Smote).txt"));
					//保存predictions
					/*rstTxt=workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(Smote).txt";
					BufferedWriter bwSmote = new BufferedWriter(new FileWriter(rstTxt));
					bwSmote.append(sbSmote.toString());
					bwSmote.flush();
					bwSmote.close();				
					*/
	
					//AcoSmote数据折评估
					clsAcoSMOTE.buildClassifier(AcoSmoteFolds);
					plText.setHeader(test);
					plText.setBuffer(sbAcoSmote);
					evalAcoSMOTE.evaluateModel(clsAcoSMOTE, test,plText);	
					//保存predictions
					/*rstTxt=workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(AcoSmote).txt";
					BufferedWriter bwAcoSmote = new BufferedWriter(new FileWriter(rstTxt));
					bwAcoSmote.append(sbAcoSmote.toString());
					bwAcoSmote.flush();
					bwAcoSmote.close();	
					*/
				
					//ROS数据折评估
					clsROS.buildClassifier(RosFolds); 	
					plText.setHeader(test);
					plText.setBuffer(sbROS);
					evalROS.evaluateModel(clsROS, test,plText);		
					//保存predictions
					/*rstTxt=workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(ROS).txt";
					BufferedWriter bwROS = new BufferedWriter(new FileWriter(rstTxt));
					bwROS.append(sbROS.toString());
					bwROS.flush();
					bwROS.close();	
					*/
					//AcoROS数据折评估
					clsAcoROS.buildClassifier(AcoRosFolds);
					plText.setHeader(test);
					plText.setBuffer(sbAcoROS);
					evalAcoROS.evaluateModel(clsAcoROS, test,plText);		
					//保存predictions
				/*	rstTxt=workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(AcoROS).txt";
					BufferedWriter bwAcoROS = new BufferedWriter(new FileWriter(rstTxt));
					bwAcoROS.append(sbAcoROS.toString());
					bwAcoROS.flush();
					bwAcoROS.close();	
			  		*/
					//BSO数据折评估
					clsBSO.buildClassifier(BsoFolds); 
					plText.setHeader(test);
					plText.setBuffer(sbBSO);
					evalBSO.evaluateModel(clsBSO,test,plText);		
					//保存predictions
					/*rstTxt=workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(BSO).txt";
					BufferedWriter bwBSO = new BufferedWriter(new FileWriter(rstTxt));
					bwBSO.append(sbBSO.toString());
					bwBSO.flush();
					bwBSO.close();	
					*/

					//AcoBSO数据折评估
					clsAcoBSO.buildClassifier(AcoBsoFolds);
					plText.setHeader(test);
					plText.setBuffer(sbAcoBSO);
					evalAcoBSO.evaluateModel(clsAcoBSO, test,plText);		
					//保存predictions
					/*rstTxt=workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(AcoBSO).txt";
					BufferedWriter bwAcoBSO = new BufferedWriter(new FileWriter(rstTxt));
					bwAcoBSO.append(sbAcoBSO.toString());
					bwAcoBSO.flush();
					bwAcoBSO.close();	
					*/
					//ADA_SYN数据折评估
					clsADA_SYN.buildClassifier(ADA_SYNFolds); 	
					plText.setHeader(test);
					plText.setBuffer(sbADA_SYN);
					evalADA_SYN.evaluateModel(clsADA_SYN,test,plText);	
					//保存predictions
					/*rstTxt=workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(ADA_SYN).txt";
					BufferedWriter bwADA_SYN = new BufferedWriter(new FileWriter(rstTxt));
					bwADA_SYN.append(sbADA_SYN.toString());
					bwADA_SYN.flush();
					bwADA_SYN.close();
					*/
					//AcoADA_SYN数据折评估
					clsAcoADA_SYN.buildClassifier(AcoADA_SYNFolds);
					plText.setHeader(test);
					plText.setBuffer(sbAcoADA_SYN);
					evalAcoADA_SYN.evaluateModel(clsAcoADA_SYN, test,plText);		
					//保存predictions
					/*rstTxt=workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(AcoADA_SYN).txt";
					BufferedWriter bwAcoADA_SYN = new BufferedWriter(new FileWriter(rstTxt));
					bwAcoADA_SYN.append(sbAcoADA_SYN.toString());
					bwAcoADA_SYN.flush();
					bwAcoADA_SYN.close();
					*/
					/*
					//AQ_BSO数据折评估
					clsAQ_BSO.buildClassifier(AQ_BSOFolds); 	
					evalAQ_BSO.evaluateModel(clsAQ_BSO,test);						
					//AcoAQ_BSO数据折评估
					clsAcoAQ_BSO.buildClassifier(AcoAQ_BSOFolds);
					evalAcoAQ_BSO.evaluateModel(clsAcoAQ_BSO, test);	
					*/					
				}
			}
			//保存实验结果
			/*rstTxt="D:\\黎敏工作\\不平衡学习\\results\\"+inputData.relationName()+"("+clsName+")(result)(runs="+runs+"folds="+folds+")("+sdf.format(btime)+").txt";
			BufferedWriter bw1 = new BufferedWriter(new FileWriter(rstTxt));
			bw1.append("分类器：+"+clsName);
			bw1.append("原始数据折**********************************************");
			bw1.append(evalOriginal.toSummaryString());
			bw1.append(evalOriginal.toMatrixString());
			bw1.append(evalOriginal.toClassDetailsString());  
			bw1.append("原始训练样本数量："+OriginalCount+'\n');
			bw1.append("*************************************************"+'\n');     
        
			bw1.append("SMOTE数据折**********************************************");
			bw1.append(evalSmote.toSummaryString());
			bw1.append(evalSmote.toMatrixString());
			bw1.append(evalSmote.toClassDetailsString());   
			bw1.append("AcoSmote数据折*******************************************");
			bw1.append(evalAcoSMOTE.toSummaryString());
			bw1.append(evalAcoSMOTE.toMatrixString());
			bw1.append(evalAcoSMOTE.toClassDetailsString());  
			bw1.append("SMOTE训练样本数量："+SMOTECount/10+",AcoSMOTE训练样本数量："+AcoSMOTECount/10+'\n');
			bw1.append("*************************************************"+'\n');

			bw1.append("BSO数据折*************************************************");
			bw1.append(evalBSO.toSummaryString());
			bw1.append(evalBSO.toMatrixString());
			bw1.append(evalBSO.toClassDetailsString());  
			bw1.append("AcoBSO数据折**********************************************");
			bw1.append(evalAcoBSO.toSummaryString());
			bw1.append(evalAcoBSO.toMatrixString());
			bw1.append(evalAcoBSO.toClassDetailsString());  
			bw1.append("BSO训练样本数量："+BSOCount/10+",AcoBSO训练样本数量："+AcoBSOCount/10+'\n');
			bw1.append("**********************************************************"+'\n');
        
			bw1.append("ROS数据折*************************************************");
			bw1.append(evalROS.toSummaryString());
			bw1.append(evalROS.toMatrixString());
			bw1.append(evalROS.toClassDetailsString());  
			bw1.append("AcoROS数据折*************************************************");
			bw1.append(evalAcoROS.toSummaryString());
			bw1.append(evalAcoROS.toMatrixString());
			bw1.append(evalAcoROS.toClassDetailsString()); 
			bw1.append("ROS训练样本数量："+ROSCount/10+",AcoROS训练样本数量："+AcoROSCount/10+'\n');
			bw1.append("**********************************************************"+'\n');
			
			bw1.append("ADA_SYN数据折**********************************************");
			bw1.append(evalADA_SYN.toSummaryString());
			bw1.append(evalADA_SYN.toMatrixString());
			bw1.append(evalADA_SYN.toClassDetailsString());  
			bw1.append("AcoADA_SYN数据折*******************************************");
			bw1.append(evalAcoADA_SYN.toSummaryString());
			bw1.append(evalAcoADA_SYN.toMatrixString());
			bw1.append(evalAcoADA_SYN.toClassDetailsString());   
			bw1.append("ADA_SYN训练样本数量："+ADA_SYNCount/10+",AcoADA_SYN训练样本数量："+AcoADA_SYNCount/10+'\n');
			bw1.append("**********************************************************");
			 
			/*
			bw1.append("AQ_BSO数据折**********************************************");
			bw1.append(evalAQ_BSO.toSummaryString());
			bw1.append(evalAQ_BSO.toMatrixString());
			bw1.append(evalAQ_BSO.toClassDetailsString());  
			bw1.append("AcoAQ_BSO数据折*******************************************");
			bw1.append(evalAcoAQ_BSO.toSummaryString());
			bw1.append(evalAcoAQ_BSO.toMatrixString());
			bw1.append(evalAcoAQ_BSO.toClassDetailsString());   
			bw1.append("AQ_BSO训练样本数量："+AQ_BSOCount/10+",AcoAQ_BSO训练样本数量："+AcoAQ_BSOCount/10+'\n');
			bw1.append("**********************************************************");
            */
			
			//保存prediction，用于绘制ROC曲线
	      /*  ThresholdCurve tc=new ThresholdCurve();

	        Instances curve =tc.getCurve(evalSmote.predictions(),minClass);
	        doubleToFile(curve,workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(Smote).txt");
	        curve =tc.getCurve(evalAcoSMOTE.predictions(),minClass);
	        doubleToFile(curve,workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(AcoSmote).txt");
	
	        curve =tc.getCurve(evalROS.predictions(),minClass);
	        doubleToFile(curve,workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(ROS).txt");	
	        curve =tc.getCurve(evalAcoROS.predictions(),minClass);
	        doubleToFile(curve,workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(AcoROS).txt");
	 

	        curve =tc.getCurve(evalBSO.predictions(),minClass);
	        doubleToFile(curve,workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(BSO).txt");
	        curve =tc.getCurve(evalAcoBSO.predictions(),minClass);
	        doubleToFile(curve,workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(AcoBSO).txt");
	    	
	        curve =tc.getCurve(evalADA_SYN.predictions(),minClass);
	        doubleToFile(curve,workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(ADA_SYN).txt");
	        curve =tc.getCurve(evalAcoADA_SYN.predictions(),minClass);
	        doubleToFile(curve,workPath +"/ROC/"+inputData.relationName()+"("+clsName+")(AcoADA_SYN).txt");
	  		
			*/					
			//bw1.close();
			
			sheetAddRow(row++,inputData.relationName(),"Original",evalOriginal,OriginalCount);
			sheetAddRow(row++,inputData.relationName(),"SMOTE",evalSmote,SMOTECount/10);
			sheetAddRow(row++,inputData.relationName(),"AcoSMOTE",evalAcoSMOTE,AcoSMOTECount/10);
			sheetAddRow(row++,inputData.relationName(),"BSO",evalBSO,BSOCount/10);
			sheetAddRow(row++,inputData.relationName(),"AcoBSO",evalAcoBSO,AcoBSOCount/10);
			sheetAddRow(row++,inputData.relationName(),"ROS",evalROS,ROSCount/10);
			sheetAddRow(row++,inputData.relationName(),"AcoROS",evalAcoROS,AcoROSCount/10);
			sheetAddRow(row++,inputData.relationName(),"ADA_SYN",evalADA_SYN,ADA_SYNCount/10);
			sheetAddRow(row++,inputData.relationName(),"AcoADA_SYN",evalAcoADA_SYN,AcoADA_SYNCount/10);
			
			//xlsResults.write(); 
			System.out.println(dataFilename+"...... completed！");
			
			dataFilename=br1.readLine();
		}
			
			xlsResults.write();  xlsResults.close();
			System.out.println("comleted！");
			long endTime = System.currentTimeMillis();   
			System.out.println("running time：" + (endTime - startTime) + "ms");    //输出程序运行时间
	}
	
	public static void doubleToFile(Instances curve, String fileName) throws IOException{
        File file = new File(fileName); //存放数组数据的文件
        FileWriter out = new FileWriter(file); //文件写入流

        int tpIndex = curve.attribute(ThresholdCurve.TP_RATE_NAME).index();	
        int fpIndex = curve.attribute(ThresholdCurve.FP_RATE_NAME).index();
        double[] tpRate = curve.attributeToDoubleArray(tpIndex);
        double[] fpRate = curve.attributeToDoubleArray(fpIndex);
       
        //将数组中的数据写入到文件中。每行各数据之间TAB间隔
        for(int i=0;i<tpRate.length;i++){
        		out.write(tpRate[i]+",");
        		out.write(fpRate[i]+"\r\n");
        }
        out.close();
	}
}
