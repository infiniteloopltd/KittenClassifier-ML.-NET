﻿using Microsoft.ML;
using Microsoft.ML.Data;

namespace MultiOutputModelTraining
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // Load data
            IDataView dataView = mlContext.Data.LoadFromTextFile<InputData>("data.csv", separatorChar: ',', hasHeader: true);

            // Split data into training and testing sets
            var dataSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var trainData = dataSplit.TrainSet;
            var testData = dataSplit.TestSet;

            // Define data preprocessing pipeline
            var dataPipeline = mlContext.Transforms.Conversion.MapValueToKey("LabelOrange", "IsOrange")
                                                 .Append(mlContext.Transforms.Conversion.MapValueToKey("LabelGrey", "IsGrey"))
                                                 .Append(mlContext.Transforms.Conversion.MapValueToKey("LabelMale", "IsMale"))
                                                 .Append(mlContext.Transforms.Categorical.OneHotEncoding("CodeEncoded", "Code"))
                                                 .Append(mlContext.Transforms.Concatenate("Features", "CodeEncoded"))
                                                 .Append(mlContext.Transforms.Conversion.ConvertType("FeaturesFloat", "Features"));

            // Define training pipeline
            var trainingPipeline = dataPipeline.Append(mlContext.Transforms.Concatenate("NonKeyOutput", "LabelOrange", "LabelGrey", "LabelMale"))
                                               .Append(mlContext.Transforms.Conversion.MapKeyToValue("Orange", "Orange"))
                                               .Append(mlContext.Transforms.Conversion.MapKeyToValue("Grey", "Grey"))
                                               .Append(mlContext.Transforms.Conversion.MapKeyToValue("Male", "Male"))
                                               .Append(mlContext.Transforms.Concatenate("KeyOutput", "Orange", "Grey", "Male"))
                                               .Append(mlContext.Transforms.CopyColumns("Label", "KeyOutput"))
                                               .Append(mlContext.Transforms.NormalizeMinMax("FeaturesFloat", "FeaturesFloat"))
                                               .Append(mlContext.BinaryClassification.Trainers.SdcaNonCalibrated("Model", maximumNumberOfIterations: 100));

            // Train the model
            var trainedModel = trainingPipeline.Fit(trainData);

            // Evaluate the model
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Model");

            Console.WriteLine($"Accuracy: {metrics.MacroAccuracy}");
        }
    }

    public class InputData
    {
        [LoadColumn(0)]
        public string Code;

        [LoadColumn(1)]
        public bool IsOrange;

        [LoadColumn(2)]
        public bool IsGrey;

        [LoadColumn(3)]
        public bool IsMale;
    }

    public class OutputData
    {
        public bool Orange;

        public bool Grey;

        public bool Male;
    }
}