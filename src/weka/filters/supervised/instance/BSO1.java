package weka.filters.supervised.instance;

import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Borderline-SMOTE 1
 * @author - 叶川
 * @date 2019年4月11日晚上10:24:29
 */
public class BSO1 extends SMOTE {

	/** for serialization. */
	static final long serialVersionUID = -1653880819059250366L;

	// 新增样本缓存
	protected Instances S_new = null;

	@Override
	protected void doSMOTE() throws Exception {
		int minIndex = 0;
		int min = Integer.MAX_VALUE;
		// 找出少数类
		if (m_DetectMinorityClass) {// 自动检测少数类
			int[] classCounts = getInputFormat().attributeStats(getInputFormat().classIndex()).nominalCounts;
			for (int i = 0; i < classCounts.length; i++) {
				if (classCounts[i] != 0 && classCounts[i] < min) {
					min = classCounts[i];
					minIndex = i;
				}
			}
		} else {// 用户手动输入少数类
			String classVal = getClassValue();
			if (classVal.equalsIgnoreCase("first")) {
				minIndex = 1;
			} else if (classVal.equalsIgnoreCase("last")) {
				minIndex = getInputFormat().numClasses();
			} else {
				minIndex = Integer.parseInt(classVal);
			}
			if (minIndex > getInputFormat().numClasses()) {
				throw new Exception("value index must be <= the number of classes");
			}
			minIndex--; // make it an index
		}

		Instances S = getInputFormat().stringFreeStructure();
		S_new = getInputFormat().stringFreeStructure();
		Instances S_minus = getInputFormat().stringFreeStructure();
		Instances S_plus = getInputFormat().stringFreeStructure();
		
		// 从训练集S中取出全部多数类样本S_minus和少数类样本S_plus
		Enumeration instanceEnum = getInputFormat().enumerateInstances();
		while (instanceEnum.hasMoreElements()) {
			Instance instance = (Instance) instanceEnum.nextElement();
			push((Instance) instance.copy());
			S.add(instance);
			if ((int) instance.classValue() == minIndex) {
				S_plus.add(instance);
			} else {
				S_minus.add(instance);
			}
		}

		// 计算标称特征值距离度量矩阵
		Map vdmMap = new HashMap();
		Enumeration attrEnum = getInputFormat().enumerateAttributes();
		while (attrEnum.hasMoreElements()) {
			Attribute attr = (Attribute) attrEnum.nextElement();
			if (!attr.equals(getInputFormat().classAttribute())) {
				if (attr.isNominal() || attr.isString()) {
					double[][] vdm = new double[attr.numValues()][attr.numValues()];
					vdmMap.put(attr, vdm);
					int[] featureValueCounts = new int[attr.numValues()];
					int[][] featureValueCountsByClass = new int[getInputFormat().classAttribute().numValues()][attr
							.numValues()];
					instanceEnum = getInputFormat().enumerateInstances();
					while (instanceEnum.hasMoreElements()) {
						Instance instance = (Instance) instanceEnum.nextElement();
						int value = (int) instance.value(attr);
						int classValue = (int) instance.classValue();
						featureValueCounts[value]++;
						featureValueCountsByClass[classValue][value]++;
					}
					for (int valueIndex1 = 0; valueIndex1 < attr.numValues(); valueIndex1++) {
						for (int valueIndex2 = 0; valueIndex2 < attr.numValues(); valueIndex2++) {
							double sum = 0;
							for (int classValueIndex = 0; classValueIndex < getInputFormat()
									.numClasses(); classValueIndex++) {
								double c1i = (double) featureValueCountsByClass[classValueIndex][valueIndex1];
								double c2i = (double) featureValueCountsByClass[classValueIndex][valueIndex2];
								double c1 = (double) featureValueCounts[valueIndex1];
								double c2 = (double) featureValueCounts[valueIndex2];
								double term1 = c1i / c1;
								double term2 = c2i / c2;
								sum += Math.abs(term1 - term2);
							}
							vdm[valueIndex1][valueIndex2] = sum;
						}
					}
				}
			}
		}

		int nearestNeighbors = getNearestNeighbors();
		Instances DANGER = getInputFormat().stringFreeStructure();
		// 得到边界少数类样本集(DANGER)
		for (int i = 0; i < S_plus.numInstances(); i++) {
			Instance instanceI = S_plus.instance(i);
			// 计算每个实例与其他实例的距离
			List distanceToInstance = new LinkedList();
			for (int j = 0; j < S.numInstances(); j++) {
				Instance instanceJ = S.instance(j);
				if (!(instanceI.equals(instanceJ))) {
					double distance = 0;
					attrEnum = getInputFormat().enumerateAttributes();
					while (attrEnum.hasMoreElements()) {
						Attribute attr = (Attribute) attrEnum.nextElement();
						if (!attr.equals(getInputFormat().classAttribute())) {
							double iVal = instanceI.value(attr);
							double jVal = instanceJ.value(attr);
							if (attr.isNumeric()) {
								distance += Math.pow(iVal - jVal, 2);
							} else {
								distance += ((double[][]) vdmMap.get(attr))[(int) iVal][(int) jVal];
							}
						}
					}
					distance = Math.pow(distance, .5);
					distanceToInstance.add(new Object[] { distance, instanceJ });
				}
			}

			// 根据距离对邻居进行排序
			Collections.sort(distanceToInstance, new Comparator() {
				public int compare(Object o1, Object o2) {
					double distance1 = (Double) ((Object[]) o1)[0];
					double distance2 = (Double) ((Object[]) o2)[0];
					return Double.compare(distance1, distance2);
				}
			});

			// 计算其多数类近邻数N_maj
			Iterator entryIterator = distanceToInstance.iterator();
			int N_maj = 0;
			int j = 0;
			while (entryIterator.hasNext() && j < nearestNeighbors) {
				Instance instance = (Instance) ((Object[]) entryIterator.next())[1];
				if ((int) instance.classValue() != minIndex) {
					N_maj++;
				}
				j++;
			}

			// 判断instanceI是否属于边界少数类样本(DANGER)
			if (nearestNeighbors > N_maj && (N_maj > (nearestNeighbors / 2) || N_maj == ((nearestNeighbors + 1) / 2))) {
				DANGER.add(instanceI);
			}
		}

		//计算匹配DANGER集后的相对采样率
		if (DANGER.numInstances()>0) {
			double percentage = (double) S_plus.numInstances()
					*getPercentage()/DANGER.numInstances();
			setPercentage(percentage);
		}
		
		// 对于所有需要的随机性，使用这个随机源
		Random rand = new Random(getRandomSeed());

		// 如果百分数不能被100整除，找出要使用的一组额外索引
		List extraIndices = new LinkedList();
		double percentageRemainder = (getPercentage() / 100) - Math.floor(getPercentage() / 100.0);
		int extraIndicesCount = (int) (percentageRemainder * S_plus.numInstances());
		if (extraIndicesCount >= 1) {
			for (int i = 0; i < S_plus.numInstances(); i++) {
				extraIndices.add(i);
			}
		}
		Collections.shuffle(extraIndices, rand);
		extraIndices = extraIndices.subList(0, extraIndicesCount);
		Set extraIndexSet = new HashSet(extraIndices);

		// 得到该实例在少数类样本集中的真实近邻值
		if (min <= getNearestNeighbors()) {
			nearestNeighbors = min - 1;
		} else {
			nearestNeighbors = getNearestNeighbors();
		}
		if (nearestNeighbors < 1)
			throw new Exception("Cannot use 0 neighbors!");

		// 对边界少数类样本集(DANGER)中的数据过采样
		Instance[] nnArray = new Instance[nearestNeighbors];
		for (int i = 0; i < DANGER.numInstances(); i++) {
			// 在DANGER集中随机选出一个主样本instanceI
			Instance instanceI = DANGER.instance(rand.nextInt(DANGER.numInstances()));
			// 计算该实例与S_plus中样本的距离
			List distanceToInstance = new LinkedList();
			for (int j = 0; j < S_plus.numInstances(); j++) {
				Instance instanceJ = S_plus.instance(j);
				if (!(instanceI.equals(instanceJ))) {
					double distance = 0;
					attrEnum = getInputFormat().enumerateAttributes();
					while (attrEnum.hasMoreElements()) {
						Attribute attr = (Attribute) attrEnum.nextElement();
						if (!attr.equals(getInputFormat().classAttribute())) {
							double iVal = instanceI.value(attr);
							double jVal = instanceJ.value(attr);
							if (attr.isNumeric()) {
								distance += Math.pow(iVal - jVal, 2);
							} else {
								distance += ((double[][]) vdmMap.get(attr))[(int) iVal][(int) jVal];
							}
						}
					}
					distance = Math.pow(distance, .5);
					distanceToInstance.add(new Object[] { distance, instanceJ });
				}
			}

			// 根据距离对邻居进行排序
			Collections.sort(distanceToInstance, new Comparator() {
				public int compare(Object o1, Object o2) {
					double distance1 = (Double) ((Object[]) o1)[0];
					double distance2 = (Double) ((Object[]) o2)[0];
					return Double.compare(distance1, distance2);
				}
			});

			// 取出最近的K个近邻
			Iterator entryIterator = distanceToInstance.iterator();
			int j = 0;
			while (entryIterator.hasNext() && j < nearestNeighbors) {
				nnArray[j] = (Instance) ((Object[]) entryIterator.next())[1];
				j++;
			}

			// 创建合成的样本
			int n = (int) Math.floor(getPercentage() / 100);
			while (n > 0 || extraIndexSet.remove(i)) {
				double[] values = new double[S_plus.numAttributes()];
				int nn = rand.nextInt(nearestNeighbors);
				attrEnum = getInputFormat().enumerateAttributes();
				while (attrEnum.hasMoreElements()) {
					Attribute attr = (Attribute) attrEnum.nextElement();
					if (!attr.equals(getInputFormat().classAttribute())) {
						if (attr.isNumeric()) {
							double dif = nnArray[nn].value(attr) - instanceI.value(attr);
							double gap = rand.nextDouble();
							values[attr.index()] = (double) (instanceI.value(attr) + gap * dif);
						} else if (attr.isDate()) {
							double dif = nnArray[nn].value(attr) - instanceI.value(attr);
							double gap = rand.nextDouble();
							values[attr.index()] = (long) (instanceI.value(attr) + gap * dif);
						} else {
							int[] valueCounts = new int[attr.numValues()];
							int iVal = (int) instanceI.value(attr);
							valueCounts[iVal]++;
							for (int nnEx = 0; nnEx < nearestNeighbors; nnEx++) {
								int val = (int) nnArray[nnEx].value(attr);
								valueCounts[val]++;
							}
							int maxIndex = 0;
							int max = Integer.MIN_VALUE;
							for (int index = 0; index < attr.numValues(); index++) {
								if (valueCounts[index] > max) {
									max = valueCounts[index];
									maxIndex = index;
								}
							}
							values[attr.index()] = maxIndex;
						}
					}
				}
				values[S_plus.classIndex()] = minIndex;
				Instance synthetic = new DenseInstance(1.0, values);
				S_new.add(synthetic);
				push(synthetic);
				n--;
			}
		}
	}

	/**
	 * Main method for running this filter.
	 * @param args
	 *            should contain arguments to the filter: use -h for help
	 */
	public static void main(String[] args) {
		runFilter(new BSO1(), args);
	}
}
