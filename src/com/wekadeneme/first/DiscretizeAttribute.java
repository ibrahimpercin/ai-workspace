package com.wekadeneme.first;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class DiscretizeAttribute {

	public static void main(String[] args) throws Exception {

		DataSource dataSource = new DataSource("./sources/datasets/cpu.arff");
		Instances data = dataSource.getDataSet();

		// Set Options
		String[] options = new String[4];
		options[0] = "-B";
		options[1] = "4";
		options[2] = "-R";
		options[3] = "0";

		// Apply Discretization
		Discretize discretize = new Discretize();
		discretize.setOptions(options);
		discretize.setInputFormat(data);

		Instances discFilteredData = Filter.useFilter(data, discretize);

		ArffSaver saver = new ArffSaver();
		saver.setInstances(discFilteredData);
		saver.setFile(new File("./sources/results/discFilter.arff"));
		saver.writeBatch();
	}

}
