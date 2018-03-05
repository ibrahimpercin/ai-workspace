package com.wekadeneme.first;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.NonSparseToSparse;

public class Sparse {
	
	public static void main(String[] args) throws Exception {
		// Load dataset
		DataSource dataSource = new DataSource("./sources/datasets/iris.arff");
		Instances data = dataSource.getDataSet();
		
		// Create NonSparseToSparse object to save in sparse Arff format
		NonSparseToSparse sparse = new NonSparseToSparse();
		
		// Specify the dataset
		sparse.setInputFormat(data);
		
		// Apply
		Instances sparseData = Filter.useFilter(data, sparse);
		
		// Save sparse Arff
		ArffSaver saver = new ArffSaver();
		saver.setInstances(sparseData);
		saver.setFile(new File("./sources/results/sparse.arff"));
		saver.writeBatch();
	}
	
}
