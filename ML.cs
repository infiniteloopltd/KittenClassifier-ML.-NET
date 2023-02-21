using System;
using System.Collections.Generic;
using System.Formats.Asn1;
using System.IO;
using System.Linq;
using Accord.IO;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math;
using Accord.Statistics.Kernels;

namespace AccordNETExample
{
    class Program
    {
        static void Main(string[] args)
        {
            // Read the CSV file
            string filePath = @"kittens.csv";
            var fileContents = File.ReadAllText(filePath);
            var data = CsvReader.FromText(fileContents, hasHeaders: true).ToTable();
            // Prepare the inputs and outputs

  
            double[][] inputs = data.Columns["Code"].ToArray<string>().Select(c => new[] { (double)c.ToString().GetHashCode() }).ToArray();
            int[] outputsOrange = data.Columns["IsOrange"].ToArray<string>().Select(c => c.ToString().ToLower() == "true" ? 1 : 0).ToArray();
            int[] outputsGrey = data.Columns["IsGrey"].ToArray<string>().Select(c => c.ToString().ToLower() == "true" ? 1 : 0).ToArray();
            int[] outputsMale = data.Columns["IsMale"].ToArray<string>().Select(c => c.ToString().ToLower() == "true" ? 1 : 0).ToArray();

            // Train the models
            var teacherOrange = new SequentialMinimalOptimization<Gaussian>()
            {
                Complexity = 100
            };
            var modelOrange = teacherOrange.Learn(inputs, outputsOrange);

            var teacherGrey = new SequentialMinimalOptimization<Gaussian>()
            {
                Complexity = 100
            };
            var modelGrey = teacherGrey.Learn(inputs, outputsGrey);

            var teacherMale = new SequentialMinimalOptimization<Gaussian>()
            {
                Complexity = 100
            };
            var modelMale = teacherMale.Learn(inputs, outputsMale);

            // Save the models
            string modelFilePath = @"path/to/save/your/models/";
            if (!Directory.Exists(modelFilePath))
                Directory.CreateDirectory(modelFilePath);
            Serializer.Save(modelOrange, Path.Combine(modelFilePath, "modelOrange.bin"));
            Serializer.Save(modelGrey, Path.Combine(modelFilePath, "modelGrey.bin"));
            Serializer.Save(modelMale, Path.Combine(modelFilePath, "modelMale.bin"));
        }
    }
}