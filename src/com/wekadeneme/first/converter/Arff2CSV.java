package com.wekadeneme.first.converter;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;

public class Arff2CSV {

	public static void main(String[] args) throws Exception {
		// load ARFF
		DataSource dataSource = new DataSource("./sources/datasets/glass.arff");
		Instances data = dataSource.getDataSet();// get instances object

		// save CSV
		CSVSaver saver = new CSVSaver();
		saver.setInstances(data);// set the dataset we want to convert
		// and save as CSV
		saver.setFile(new File("./sources/results/arff2csv.csv"));
		saver.writeBatch();
	}
}