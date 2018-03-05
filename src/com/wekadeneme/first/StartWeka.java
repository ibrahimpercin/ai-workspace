package com.wekadeneme.first;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class StartWeka {

	public static void main(String[] args) throws Exception {
		DataSource dataSource = new DataSource("./sources/datasets/iris.arff");

		Instances trainData = dataSource.getDataSet();

		// Setting class attribute
		/**
		 * classIndex() --> Returns the class attribute's index. Returns negative number
		 * if it's undefined.
		 */
		if (trainData.classIndex() == -1)
			trainData.setClassIndex(trainData.numAttributes() - 1);

		// Naive Bayes
		NaiveBayes naiveBayes = new NaiveBayes();
		naiveBayes.buildClassifier(trainData);

		// Logistic Regression
		Logistic logistic = new Logistic();
		logistic.buildClassifier(trainData);

		// OneR
		OneR oneR = new OneR();
		oneR.buildClassifier(trainData);

		// Random Forest
		RandomForest forest = new RandomForest();
		forest.buildClassifier(trainData);

		// Linear Regression
		// LinearRegression linearRegression = new LinearRegression();
		// linearRegression.buildClassifier(train);

		J48 j48 = new J48();
		j48.buildClassifier(trainData);

		Evaluation eval = new Evaluation(trainData);
		eval.crossValidateModel(j48, trainData, 10, new Random(1));

		// Evaluation evaluation = new EM();

		System.out.println(eval.toSummaryString("\nResults\n======\n", true));

		System.out.println("FMeasure => " + eval.fMeasure(1) + " - Precision => " + eval.precision(1) + " - Recall => "
				+ eval.recall(1) + "\nFNegative => " + eval.falseNegativeRate(1) + " - FPositive => "
				+ eval.falsePositiveRate(1));
	}

}
