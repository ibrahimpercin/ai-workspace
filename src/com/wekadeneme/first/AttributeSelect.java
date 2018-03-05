package com.wekadeneme.first;

import java.io.File;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class AttributeSelect {

	public static void main(String[] args) throws Exception {
		// Load Data
		DataSource dataSource = new DataSource("./sources/datasets/iris.arff");
		Instances data = dataSource.getDataSet();

		// Create Atrribute Selection
		AttributeSelection selectionFilter = new AttributeSelection();

		// Create evaluator and search algorithm objects
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		// set the algorithm to search backward
		search.setSearchBackwards(true);
		// set the filter to use evaluator and search algorithm
		selectionFilter.setEvaluator(eval);
		selectionFilter.setSearch(search);

		// specify the dataset
		selectionFilter.setInputFormat(data);
		// apply filter to dataset
		Instances filteredData = Filter.useFilter(data, selectionFilter);

		// save
		ArffSaver saver = new ArffSaver();
		saver.setInstances(filteredData);
		saver.setFile(new File("./sources/results/attributeSelectionResult.arff"));
		saver.writeBatch();

		System.out.println("Build Successfull !");
	}

}
