/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    Resample.java
 *    Copyright (C) 2002-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.filters.supervised.instance;

import java.util.Enumeration;
import java.util.Random;
import weka.core.*;

/**
 * <!-- globalinfo-start --> Produces a random subsample of a dataset using
 * either sampling with replacement or without replacement.<br/>
 * The original dataset must fit entirely in memory. The number of instances in
 * the generated dataset may be specified. The dataset must have a nominal class
 * attribute. If not, use the unsupervised version. The filter can be made to
 * maintain the class distribution in the subsample, or to bias the class
 * distribution toward a uniform distribution. When used in batch mode (i.e. in
 * the FilteredClassifier), subsequent batches are NOT resampled.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -S &lt;num&gt;
 *  Specify the random number seed (default 1)
 * </pre>
 * 
 * <pre>
 * -Z &lt;num&gt;
 *  The size of the output dataset, as a percentage of
 *  the input dataset (default 100)
 * </pre>
 * 
 * <pre>
 * -B &lt;num&gt;
 *  Bias factor towards uniform class distribution.
 *  0 = distribution in input data -- 1 = uniform distribution.
 *  (default 0)
 * </pre>
 * 
 * <pre>
 * -no-replacement
 *  Disables replacement of instances
 *  (default: with replacement)
 * </pre>
 * 
 * <pre>
 * -V
 *  Inverts the selection - only available with '-no-replacement'.
 * </pre>
 * 
 * <!-- options-end -->
 * 
/**
 * SampleSizePercent = -1 : 随机过采样平衡数据集
 * SampleSizePercent = -2 : 随机降采样平衡数据集
 *            else : 等价于 sample
 * @author - 黎敏
 * @date 2019年4月22日下午2:46:44
 * 
 * 20200110修改: 针对过采样部分提供获取新增样本的接口
 *  其中对原代码部分逻辑做出如下修改
 *      原:不保留原集合样本,从零开始采用有放回的随机抽样方法,可能存在的问题,虽然是过采样,但依然有可能出现有样本未被采样进去的情况.
 *      新:默认全部选中原集合样本一次,从 100% 开始采用有放回的随机抽样方法,并通对每个类别的取样百分比全部做减 100.0 的处理来抵消.
 */
public class ResampleofBalance extends Resample{
  static final double epsilon =0.000001;
  /** for serialization. */
  static final long serialVersionUID = 7079064953548300689L;

  //保存所有新增样本
  protected Instances S_newNum = null;
  
  //获取合成的样本
  public Instances getS_newNum() throws Exception {
      if (S_newNum != null) {
          return S_newNum;
      } else {
          throw new Exception("请先调用Filter.useFilter()合成样本!");
      }
  }
  
  //Override
  protected void createSubsample() {
        //等于-1，随机过采样平衡
        if (Math.abs(m_SampleSizePercent+1.0)<epsilon) {
            createSubsampleofBalance(-1);
        }//等于-2，随机降采样平衡
        else if (Math.abs(m_SampleSizePercent+2.0)<epsilon) {
            createSubsampleofBalance(-2);
        }
        else {
            super.createSubsample();
        }
  }

    //对数据随机降采样或随机过采样，使得整个数据集平衡
    protected void createSubsampleofBalance(int OverorUnder) {
        int maxIndex=0;
        double max=1.0;
        int minIndex=0;
        
        Instances data = getInputFormat();
        double min=data.numInstances();     
         
        // Figure out how much data there is per class
        double[] numInstancesPerClass = new double[data.numClasses()];
        for (Instance instance : data) {
          numInstancesPerClass[(int)instance.classValue()]++;
        }

        //找出最大类别和最小类别的记数和索引
        for (int i = 0; i < data.numClasses(); i++) {
            if (max< numInstancesPerClass[i]){
                max=numInstancesPerClass[i];
                maxIndex=i;
            }else if (min> numInstancesPerClass[i])
                min=numInstancesPerClass[i];
                minIndex=i;
        }       
        
        //计算每个类别的取样百分比
        double[] sampleSizePercentPerClass= new double[data.numClasses()];
        if (OverorUnder==-1){ //如果是过采样平衡
            for (int i = 0; i < data.numClasses(); i++) {
                if (i==maxIndex) 
                    sampleSizePercentPerClass[i]=100.0; 
                else {
                    if (numInstancesPerClass[i]>0) 
                    sampleSizePercentPerClass[i]=((max*100.0)/numInstancesPerClass[i]);
                }
            }
            // 20200110修改:针对过采样部分的调整,保留全部原始样本,所以每个类别的取样百分比全部减 100.0
            for (int i = 0; i < sampleSizePercentPerClass.length; i++) {
                sampleSizePercentPerClass[i] = sampleSizePercentPerClass[i] - 100.0;
            }
         }
        else if (OverorUnder==-2){//如果是降采样平衡
            for (int i = 0; i < data.numClasses(); i++) {
                if (i==minIndex) 
                    sampleSizePercentPerClass[i]=100.0; 
                else {
                    if (numInstancesPerClass[i]>0) 
                    sampleSizePercentPerClass[i]=((min*100.0)/(numInstancesPerClass[i]*1.0));
                }
            }                        
        }     
        
        // Collect data per class
        //获取每个类别的实例集合
        Instance[][] instancesPerClass = new Instance[data.numClasses()][];
        int numActualClasses = 0;
        for (int i = 0; i < data.numClasses(); i++) {
           //初始化每个类别的实例集合   
          instancesPerClass[i] = new Instance[(int)numInstancesPerClass[i]];
          if (numInstancesPerClass[i] > 0) {
            numActualClasses++;
          }
        }
        int[] counterPerClass = new int[data.numClasses()];
        for (Instance instance : data) {
          int classValue = (int)instance.classValue();
          instancesPerClass[classValue][counterPerClass[classValue]++] = instance;
        }

        // Determine how much data we want for each class
        //获得每个类别的取样数目
        int[] numInstancesToSample = new int[data.numClasses()];
        for (int i = 0; i < data.numClasses(); i++) {

          // Can't sample any data if there is no data for the class
          if (numInstancesPerClass[i] == 0) {
            continue;
          }

          // Blend observed prior and uniform prior based on user-defined blending parameter
          int sampleSize = (int)((sampleSizePercentPerClass[i]/ 100.0) * ((1 - m_BiasToUniformClass) * numInstancesPerClass[i] +
            m_BiasToUniformClass * data.numInstances() / numActualClasses));
          if (getNoReplacement() && sampleSize > numInstancesPerClass[i]) {
            System.err.println("警告: 没有足够的样本 " + data.classAttribute().value(i) + 
              " for selected value of bias parameter in supervised Resample filter when sampling without replacement.");
            sampleSize = (int)numInstancesPerClass[i];
          }
          //获得第i个类别的取样数目
          numInstancesToSample[i] = (int)sampleSize;
        }

        // Sample data
        Random random = new Random(m_RandomSeed);
        if (!getNoReplacement()) {
          // 先将输入的数据集压入输出
          Enumeration instanceEnum = getInputFormat().enumerateInstances();
          while (instanceEnum.hasMoreElements()) {
              Instance instance = (Instance) instanceEnum.nextElement();
              push((Instance) instance.copy());
          }
          // 初始化 S_newNum
          S_newNum = getInputFormat().stringFreeStructure();
          for (int i = 0; i < data.numClasses(); i++) {
            int numEligible = (int)numInstancesPerClass[i];
            for (int j = 0; j < numInstancesToSample[i]; j++) {
              // Sampling with replacement
              // 随机挑选一个样本
              Instance new_instance = instancesPerClass[i][random.nextInt(numEligible)];
              // 加入新增样本集
              S_newNum.add(new_instance);
              // 压入输出
              push(new_instance);
            }
          }
        } else {
          for (int i = 0; i < data.numClasses(); i++) {
            int numEligible = (int)numInstancesPerClass[i];
            
            // Set up array of indices
            int[] selected = new int[numEligible];
            for (int j = 0; j < numEligible; j++) {
              selected[j] = j;
            }
            for (int j = 0; j < numInstancesToSample[i]; j++) {

              // 不重复采样
              int chosenLocation = random.nextInt(numEligible);
              int chosen = selected[chosenLocation];
              numEligible--;
              //chosenLocation和chosen交换
              selected[chosenLocation] = selected[numEligible];
              selected[numEligible] = chosen; //每一次把选中的样本放在最后
            }

            // 确定转换
            if (getInvertSelection()) {

              // 以第一个numqualified实例为例，
              //  索引从0到umEligible-1的样本是未选中的样本， 因为它们没有被选中
              for (int j = 0; j < numEligible; j++) {
               //调用父类Filter的push方法，把样本压入输出队列
                push(instancesPerClass[i][selected[j]]);
              }
            } else {
               //对选中的样本进行处理 
              // Take the elements that have been selected
              for (int j = numEligible; j < (int)numInstancesPerClass[i]; j++) {
                //调用父类Filter的push方法，把样本压入输出队列
                push(instancesPerClass[i][selected[j]]);
              }
            }
          }
        }
    }

    

  
}
