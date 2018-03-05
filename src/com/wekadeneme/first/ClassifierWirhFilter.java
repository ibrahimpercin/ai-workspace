package com.wekadeneme.first;

import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

public class ClassifierWirhFilter {

	public static void main(String[] args) throws Exception {
		// load dataset
		DataSource dataSource = new DataSource("./sources/datasets/iris.arff");
		Instances data = dataSource.getDataSet();

		// set class index to the last attribute
		/**
		 * classIndex() --> Returns the class attribute's index. Returns negative number
		 * if it's undefined.
		 */
		data.setClassIndex(data.numAttributes() - 1);

		Remove filter = new Remove();
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = "1";
		filter.setOptions(options);

		J48 tree = new J48();

		// Create filtered classifier object
		FilteredClassifier classifier = new FilteredClassifier();
		// Specify the filter
		classifier.setFilter(filter);
		// specify the classifier
		classifier.setClassifier(tree);
		// Build the meta-classifier
		classifier.buildClassifier(data);

		System.out.println(tree.graph());
	}
}
