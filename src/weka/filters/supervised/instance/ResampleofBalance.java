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
 * SampleSizePercent = -1 : ���������ƽ�����ݼ�
 * SampleSizePercent = -2 : ���������ƽ�����ݼ�
 *            else : �ȼ��� sample
 * @author - ����
 * @date 2019��4��22������2:46:44
 * 
 * 20200110�޸�: ��Թ����������ṩ��ȡ���������Ľӿ�
 *  ���ж�ԭ���벿���߼����������޸�
 *      ԭ:������ԭ��������,���㿪ʼ�����зŻص������������,���ܴ��ڵ�����,��Ȼ�ǹ�����,����Ȼ�п��ܳ���������δ��������ȥ�����.
 *      ��:Ĭ��ȫ��ѡ��ԭ��������һ��,�� 100% ��ʼ�����зŻص������������,��ͨ��ÿ������ȡ���ٷֱ�ȫ������ 100.0 �Ĵ���������.
 */
public class ResampleofBalance extends Resample{
  static final double epsilon =0.000001;
  /** for serialization. */
  static final long serialVersionUID = 7079064953548300689L;

  //����������������
  protected Instances S_newNum = null;
  
  //��ȡ�ϳɵ�����
  public Instances getS_newNum() throws Exception {
      if (S_newNum != null) {
          return S_newNum;
      } else {
          throw new Exception("���ȵ���Filter.useFilter()�ϳ�����!");
      }
  }
  
  //Override
  protected void createSubsample() {
        //����-1�����������ƽ��
        if (Math.abs(m_SampleSizePercent+1.0)<epsilon) {
            createSubsampleofBalance(-1);
        }//����-2�����������ƽ��
        else if (Math.abs(m_SampleSizePercent+2.0)<epsilon) {
            createSubsampleofBalance(-2);
        }
        else {
            super.createSubsample();
        }
  }

    //����������������������������ʹ���������ݼ�ƽ��
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

        //�ҳ����������С���ļ���������
        for (int i = 0; i < data.numClasses(); i++) {
            if (max< numInstancesPerClass[i]){
                max=numInstancesPerClass[i];
                maxIndex=i;
            }else if (min> numInstancesPerClass[i])
                min=numInstancesPerClass[i];
                minIndex=i;
        }       
        
        //����ÿ������ȡ���ٷֱ�
        double[] sampleSizePercentPerClass= new double[data.numClasses()];
        if (OverorUnder==-1){ //����ǹ�����ƽ��
            for (int i = 0; i < data.numClasses(); i++) {
                if (i==maxIndex) 
                    sampleSizePercentPerClass[i]=100.0; 
                else {
                    if (numInstancesPerClass[i]>0) 
                    sampleSizePercentPerClass[i]=((max*100.0)/numInstancesPerClass[i]);
                }
            }
            // 20200110�޸�:��Թ��������ֵĵ���,����ȫ��ԭʼ����,����ÿ������ȡ���ٷֱ�ȫ���� 100.0
            for (int i = 0; i < sampleSizePercentPerClass.length; i++) {
                sampleSizePercentPerClass[i] = sampleSizePercentPerClass[i] - 100.0;
            }
         }
        else if (OverorUnder==-2){//����ǽ�����ƽ��
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
        //��ȡÿ������ʵ������
        Instance[][] instancesPerClass = new Instance[data.numClasses()][];
        int numActualClasses = 0;
        for (int i = 0; i < data.numClasses(); i++) {
           //��ʼ��ÿ������ʵ������   
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
        //���ÿ������ȡ����Ŀ
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
            System.err.println("����: û���㹻������ " + data.classAttribute().value(i) + 
              " for selected value of bias parameter in supervised Resample filter when sampling without replacement.");
            sampleSize = (int)numInstancesPerClass[i];
          }
          //��õ�i������ȡ����Ŀ
          numInstancesToSample[i] = (int)sampleSize;
        }

        // Sample data
        Random random = new Random(m_RandomSeed);
        if (!getNoReplacement()) {
          // �Ƚ���������ݼ�ѹ�����
          Enumeration instanceEnum = getInputFormat().enumerateInstances();
          while (instanceEnum.hasMoreElements()) {
              Instance instance = (Instance) instanceEnum.nextElement();
              push((Instance) instance.copy());
          }
          // ��ʼ�� S_newNum
          S_newNum = getInputFormat().stringFreeStructure();
          for (int i = 0; i < data.numClasses(); i++) {
            int numEligible = (int)numInstancesPerClass[i];
            for (int j = 0; j < numInstancesToSample[i]; j++) {
              // Sampling with replacement
              // �����ѡһ������
              Instance new_instance = instancesPerClass[i][random.nextInt(numEligible)];
              // ��������������
              S_newNum.add(new_instance);
              // ѹ�����
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

              // ���ظ�����
              int chosenLocation = random.nextInt(numEligible);
              int chosen = selected[chosenLocation];
              numEligible--;
              //chosenLocation��chosen����
              selected[chosenLocation] = selected[numEligible];
              selected[numEligible] = chosen; //ÿһ�ΰ�ѡ�е������������
            }

            // ȷ��ת��
            if (getInvertSelection()) {

              // �Ե�һ��numqualifiedʵ��Ϊ����
              //  ������0��umEligible-1��������δѡ�е������� ��Ϊ����û�б�ѡ��
              for (int j = 0; j < numEligible; j++) {
               //���ø���Filter��push������������ѹ���������
                push(instancesPerClass[i][selected[j]]);
              }
            } else {
               //��ѡ�е��������д��� 
              // Take the elements that have been selected
              for (int j = numEligible; j < (int)numInstancesPerClass[i]; j++) {
                //���ø���Filter��push������������ѹ���������
                push(instancesPerClass[i][selected[j]]);
              }
            }
          }
        }
    }

    

  
}
