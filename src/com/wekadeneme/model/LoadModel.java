package com.wekadeneme.model;

import weka.classifiers.functions.SMOreg;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class LoadModel {

	public static void main(String[] args) throws Exception {
		// Observe the type-cast
		SMOreg smo = (SMOreg) SerializationHelper.read("./sources/results/smo_model.model");
		
		// Load Data
		DataSource dataSource = new DataSource("./sources/datasets/cpu.arff");
		Instances data = dataSource.getDataSet();
		// set class index as last attribute
		if (data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes() - 1);
		}
		
		// get class double value for first instance
		double actualValue = data.instance(0).classValue();
		// get Instance object of first Instance
		Instance newInstance = data.instance(0);
		// call classiftInstance, whic return a double value for the class
		double predictedSMO = smo.classifyInstance(newInstance);
		
		System.out.println("Actual value --> " + actualValue + "\nPredicted Value --> " + predictedSMO);
	}

}
