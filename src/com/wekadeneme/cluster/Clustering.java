package com.wekadeneme.cluster;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Clustering {

	public static void main(String[] args) throws Exception {
		// Load Data
		DataSource dataSource = new DataSource("./sources/datasets/weather.nominal.arff");
		// get instances object
		Instances data = dataSource.getDataSet();

		// new instance of clusterer
		SimpleKMeans model = new SimpleKMeans();// Simple EM (expectation maximisation)
		// number of clusters
		model.setNumClusters(4);
		// set distance function
		// model.setDistanceFunction(new weka.core.ManhattanDistance());
		// build the clusterer
		model.buildClusterer(data);
		System.out.println(model);

		// to cluster an instance .. returns cluster number as int
		// model.clusterInstance(instance);

		// returns an array containing the estimated membership probabilities of the
		// test instance in each cluster (this should sum to at most 1)
		// model.distributionForInstance(instance);

		/*
		 * We can evaluate a clusterer with the ClusterEvaluation class For instance,
		 * separate train and test datasets can be used we can print out the number of
		 * clusters found
		 */
		ClusterEvaluation clusterEvaluation = new ClusterEvaluation();
		// Load  Test Data
		DataSource dataSourceTest = new DataSource("./sources/datasets/weather-test.arff");
		
		// get instances object
		Instances dataTest = dataSourceTest.getDataSet();

		clusterEvaluation.setClusterer(model);
		clusterEvaluation.evaluateClusterer(dataTest);// this should be a test dataset!

		System.out.println("# of clusters: " + clusterEvaluation.getNumClusters());
	}

}
