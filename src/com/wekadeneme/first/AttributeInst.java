package com.wekadeneme.first;

import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.experiment.Stats;

public class AttributeInst {

	public static void main(String[] args) throws Exception {
		// Load Data
		DataSource dataSource = new DataSource("./sources/datasets/iris.arff");
		Instances data = dataSource.getDataSet();

		// set class index as the last attribute
		if (data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes() - 1);
		}

		// get number of attributes (notice class is not counted)
		int numAttr = data.numAttributes() - 1;
		for (int i = 0; i < numAttr; i++) {
			// check if current attr is of type nominal
			if (data.attribute(i).isNominal()) {
				System.out.println("The " + i + "th Attribute is Nominal");
				// get number of values
				int n = data.attribute(i).numValues();
				System.out.println("The " + i + "th Attribute has: " + n + " values");
			}

			// get an AttributeStats object
			AttributeStats attrStats = data.attributeStats(i);
			int distinctValue = attrStats.distinctCount;
			System.out.println("The " + i + "th Attribute has: " + distinctValue + " distinct values");

			// get a Stats object from the AttributeStats
			if (data.attribute(i).isNumeric()) {
				System.out.println("The " + i + "th Attribute is Numeric");
				Stats s = attrStats.numericStats;
				System.out.println("The " + i + "th Attribute has min value: " + s.min + " and max value: " + s.max
						+ " and mean value: " + s.mean);
			}
		}
		// get number of instances
		int numInst = data.numInstances();
		// loop through all instances
		for (int j = 0; j < numInst; j++) {
			// get the j'th instance
			Instance instance = data.instance(j);
			// check if 1st attr is missing from the j'th instance
			if (instance.isMissing(0)) {
				System.out.println("The " + 0 + "th Attribute is missing");
			}
			// check if the class is missing from the j'th instance
			if (instance.classIsMissing()) {
				System.out.println("The class is missing in the " + j + "th Instance");
			}
			// if you want to access the value of Class in your data
			// notice classes of type nominal and string are given ID's
			double classValue = instance.classValue();
			System.out.println(instance.classAttribute().value((int) classValue));

		}
	}

}
