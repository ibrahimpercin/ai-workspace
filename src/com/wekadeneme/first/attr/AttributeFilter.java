package com.wekadeneme.first.attr;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class AttributeFilter {

	public static void main(String[] args) throws Exception {

		// Load Dataset
		DataSource dataSource = new DataSource("./sources/datasets/glass.arff");
		Instances data = dataSource.getDataSet();

		// User a simple filter to remove a certain attribute with "-R" in index 1
		String[] options = new String[] { "-R", "1" };

		// Create a remove object <-- it means filter class
		Remove filter = new Remove();

		// Set the filter options
		filter.setOptions(options);

		// Send the dataset to the filter
		filter.setInputFormat(data);

		// Apply filter
		Instances filteredData = Filter.useFilter(data, filter);

		// Save filtered dataset to new file
		ArffSaver saver = new ArffSaver();
		saver.setInstances(filteredData);
		saver.setFile(new File("./sources/results/filtered.arff"));
		saver.writeBatch();
	}

}
