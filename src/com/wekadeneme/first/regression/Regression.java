package com.wekadeneme.first.regression;

import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.trees.M5P;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Regression {

	public static void main(String[] args) throws Exception {
		DataSource dataSource = new DataSource("./sources/datasets/cpu.arff");
		Instances data = dataSource.getDataSet();
		//set class index to the last attribute
		data.setClassIndex(data.numAttributes()-1);
		
		//build model
		LinearRegression regression = new LinearRegression();
		regression.buildClassifier(data);
		//output model
		System.out.println(regression);	
		
		SMOreg smo = new SMOreg();
		smo.buildClassifier(data);
		System.out.println(smo);
		System.out.println("------------------");
		
		M5P m5p = new M5P();
		m5p.buildClassifier(data);
		System.out.println(m5p);
	
	}

}
