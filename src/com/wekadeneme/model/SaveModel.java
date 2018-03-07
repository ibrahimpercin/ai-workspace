package com.wekadeneme.model;

import weka.classifiers.functions.SMOreg;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class SaveModel {

	public static void main(String[] args) throws Exception {
		// Load Data
		DataSource dataSource = new DataSource("./sources/datasets/cpu.arff");
		Instances data = dataSource.getDataSet();
		
		data.setClassIndex(data.numAttributes() - 1);
		
		// Build model
		SMOreg smo = new SMOreg();
		// Apply
		smo.buildClassifier(data);
		
		System.out.println(smo);
		
		// Save model
		SerializationHelper.write("./sources/results/smo_model.model", smo);

	}

}
