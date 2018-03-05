package com.wekadeneme.first;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Classification {
	public static void main(String args[]) throws Exception {
		// load dataset
		DataSource dataSource = new DataSource("./sources/datasets/iris.arff");
		Instances data = dataSource.getDataSet();

		// set class index to the last attribute
		/**
		 * classIndex() --> Returns the class attribute's index. Returns negative number
		 * if it's undefined.
		 */
		data.setClassIndex(data.numAttributes() - 1);

		// create and build the classifier!
		NaiveBayes bayes = new NaiveBayes();
		bayes.buildClassifier(data);

		// print out capabilities
		System.out.println(bayes.getCapabilities().toString());
		System.out.println(bayes.toString());

		SMO svm = new SMO();
		svm.buildClassifier(data);
		System.out.println(svm.getCapabilities().toString());
		System.out.println(svm.toString());

		String[] options = new String[4];

		// -C <pruning confidence>
		// Set confidence threshold for pruning.
		options[0] = "-C";
		options[1] = "0.11";
		// -M <minimum number of instances>
		// Set minimum number of instances per leaf.
		options[2] = "-M";
		options[3] = "3";

		J48 tree = new J48();
		tree.setOptions(options);
		tree.buildClassifier(data);

		System.out.println(tree.getCapabilities().toString());
		System.out.println(tree.graph());

	}
}