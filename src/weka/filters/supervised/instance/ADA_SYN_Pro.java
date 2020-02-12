package weka.filters.supervised.instance;

import java.util.Enumeration;

import weka.core.Instance;
import weka.core.Instances;

/**
 * classValue = -1 : 自动平衡数据集
 * 			  else : 等价于 ADA_SYN
 * @author - 叶川
 * @date 2019年5月12日下午3:07:33
 */
public class ADA_SYN_Pro extends ADA_SYN {

	/** for serialization. */
	static final long serialVersionUID = -1653880819059250368L;

	// 保存所有新增样本
	protected Instances S_newNum = null;
	
	// 获取合成的样本
	public Instances getS_newNum() throws Exception {
		if (S_newNum != null) {
			return S_newNum;
		} else {
			throw new Exception("请先调用Filter.useFilter()合成样本!");
		}
	}

	@Override
	protected void doSMOTE() throws Exception {
		if (m_ClassValueIndex.equals("-1")) {
			doADA_SYN_Pro();
		} else {
			super.doSMOTE();
		}
	}

	// 对目标类别不平衡训练集自动运行ADA_SYN使其平衡。
	protected void doADA_SYN_Pro() {
		int maxIndex = 0;
		int max = 0;
		int[] classCounts = getInputFormat().attributeStats(getInputFormat().classIndex()).nominalCounts;
		// 找到数据最多的那一项
		for (int i = 0; i < classCounts.length; i++) {
			if (classCounts[i] != 0 && classCounts[i] > max) {
				max = classCounts[i];
				maxIndex = i;
			}
		}

		S_newNum = getInputFormat().stringFreeStructure();
		for (int i = 0; i < classCounts.length; i++) {
			// 对除最大项以外的项过采样
			if (i != maxIndex) {
				// 计算过采样倍数
				double percentage = ((double) max / classCounts[i] - 1) * 100;
				setPercentage(percentage);
				setClassValue(String.valueOf(i + 1));
				try {
					super.doSMOTE();
					// 保存本次执行完成后生成的新样本
					for (Instance instance : S_new) {
						S_newNum.add(instance);
					}
					// 清空新增样本缓存
					S_new.delete();
				} catch (Exception e) {
					e.printStackTrace();
					continue;
				} finally {
					// 清空输出队列
					while (output() != null) {
					}
				}
			}
		}

		// 将输入的数据集压入输出
		Enumeration instanceEnum = getInputFormat().enumerateInstances();
		while (instanceEnum.hasMoreElements()) {
			Instance instance = (Instance) instanceEnum.nextElement();
			push((Instance) instance.copy());
		}

		// 将新生成的样本压入输出
		instanceEnum = S_newNum.enumerateInstances();
		while (instanceEnum.hasMoreElements()) {
			Instance instance = (Instance) instanceEnum.nextElement();
			push((Instance) instance.copy());
		}
	}

	/**
	 * Main method for running this filter.
	 * @param args
	 *            should contain arguments to the filter: use -h for help
	 */
	public static void main(String[] args) {
		runFilter(new ADA_SYN_Pro(), args);
	}
}
