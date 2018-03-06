package com.wekadeneme.first.assoc;

import weka.associations.Apriori;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class AssocRules {

	public static void main(String[] args) throws Exception {
		// Load Data
		DataSource dataSource = new DataSource("./sources/datasets/vote.arff");
		Instances data = dataSource.getDataSet();

		// Set number of rules
		int numberOfRules = 10;

		// Create apriori object and Set number of rules
		Apriori apriori = new Apriori();
		apriori.setNumRules(numberOfRules);

		// Apply
		apriori.buildAssociations(data);

		System.out.println(apriori);

	}
}
