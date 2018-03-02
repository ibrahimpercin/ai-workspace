package com.wekadeneme.first;

import java.io.IOException;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class CrossValidation {

	public static void main(String[] args) throws Exception, IOException {

		DataSource source = new DataSource("C:\\Program Files\\Weka-3-8\\data\\glass.arff");

		Instances data = source.getDataSet();

		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);

		double precision = 0;
		double recall = 0;
		double fmeasure = 0;
		double error = 0;

		int size = data.numInstances() / 10;
		int begin = 0;
		int end = size - 1;

		for (int i = 0; i <= 10; i++) {
			System.out.println("Iteration: " + i);
			Instances trainingInstances = new Instances(data);
			Instances testIntances = new Instances(data, begin, (end - begin));
			for (int j = 0; j < (end - begin); j++) {
				trainingInstances.delete(begin);
			}

			NaiveBayes naiveBayes = new NaiveBayes();
			naiveBayes.buildClassifier(trainingInstances);

			Evaluation evaluation = new Evaluation(testIntances);
			evaluation.evaluateModel(naiveBayes, testIntances);

			System.out.println("P: " + evaluation.precision(1));
			System.out.println("R: " + evaluation.recall(1));
			System.out.println("F: " + evaluation.fMeasure(1));
			System.out.println("e: " + evaluation.errorRate());

			precision += evaluation.precision(1);
			recall += evaluation.recall(1);
			fmeasure += evaluation.fMeasure(1);
			error += evaluation.errorRate();

			// UPDATE
			begin = end + 1;
			end += size;
			if (i == (9)) {
				end = data.numInstances();
			}
		}
		System.out.println("Precision: " + precision / 10.0);
		System.out.println("Recall: " + recall / 10.0);
		System.out.println("FMeasure: " + fmeasure / 10.0);
		System.out.println("Error: " + error / 10.0);
	}

}
