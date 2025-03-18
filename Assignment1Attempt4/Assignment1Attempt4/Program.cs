using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace Assignment1_Attempt2
{
    class Program
    {
        static void Main(string[] args)
        {
            SalaryList SL = new SalaryList();
            SL.LoadIndividuals();
            SalaryList.Normalize(SL.newInd);

            // Split data into training and validation sets
            (Individual[] trainingData, Individual[] validationData) = SplitData(SL.newInd, 0.8);

            // Define learning rates to test
            double[] learningRates = new double[] { 0.001, 0.01, 0.1, 1.0 };

            using (StreamWriter sw = new StreamWriter("results.csv"))
            {
                sw.WriteLine("LearningRate,Epoch,MSE,SSE");

                foreach (double lr in learningRates)
                {
                    // Initialize Perceptron with current learning rate and Train
                    Perceptron perceptron = new Perceptron(inputSize: 7, learningRate: lr);
                    perceptron.Train(trainingData, epochs: 1000000, validationData: validationData);

                    // Write results to CSV
                    for (int epoch = 0; epoch < perceptron.MseHistory.Count; epoch++)
                    {
                        sw.WriteLine($"{lr},{epoch + 1},{perceptron.MseHistory[epoch]},{perceptron.SseHistory[epoch]}");
                    }
                }
            }

            // Output the predicted vs. actual values for a subset of validation data
            Perceptron finalPerceptron = new Perceptron(inputSize: 7, learningRate: 0.01);
            finalPerceptron.Train(trainingData, epochs: 1000000, validationData: validationData);
            OutputPredictions(finalPerceptron, validationData);

            Console.WriteLine("Training complete. Results saved to results.csv.");
            Console.ReadLine();
        }

        public static void OutputPredictions(Perceptron perceptron, Individual[] validationData)
        {
            Console.WriteLine("Predicted vs Actual Salary:");
            foreach (var individual in validationData)
            {
                double[] inputs = { individual.Education, individual.NrSupervise, individual.NrPositionHeld,
                                    individual.PercResponsibility, individual.NumChildren, individual.Age,
                                    individual.YearsExperience };

                double predictedSalary = perceptron.Predict(inputs);
                double denormalizedPredictedSalary = SalaryList.DenormalizeSalary(predictedSalary);
                double denormalizedActualSalary = SalaryList.DenormalizeSalary(individual.Salary);

                Console.WriteLine($"Predicted: {denormalizedPredictedSalary:F2}, Actual: {denormalizedActualSalary:F2}");
            }
        }

        public static (Individual[], Individual[]) SplitData(Individual[] data, double trainingRatio)
        {
            int trainingSize = (int)(data.Length * trainingRatio);
            Individual[] trainingData = new Individual[trainingSize];
            Individual[] validationData = new Individual[data.Length - trainingSize];

            Array.Copy(data, 0, trainingData, 0, trainingSize);
            Array.Copy(data, trainingSize, validationData, 0, data.Length - trainingSize);

            return (trainingData, validationData);
        }
    }

    public class SalaryList
    {
        public Individual[] newInd = new Individual[2000];
        private int size = 0;
        public static double minSalary;
        public static double maxSalary;

        private void AddToList(Individual b)
        {
            newInd[size] = b; // adds the element b to the back of array.
            size++; //increment size.
        }

        public void LoadIndividuals()
        {
            StreamReader sr = new StreamReader("SalData.csv");
            String line = sr.ReadLine();
            line = sr.ReadLine();
            while (line != null)
            {
                String[] elements = line.Split(",");
                Individual temp = new Individual(double.Parse(elements[0]), double.Parse(elements[1]), double.Parse(elements[2]), double.Parse(elements[3]), double.Parse(elements[4]), double.Parse(elements[5]), double.Parse(elements[6]), double.Parse(elements[7]));
                AddToList(temp);
                line = sr.ReadLine();
            }
            sr.Close();
        }

        public void LoadEvalIndividuals()
        {
            StreamReader sr = new StreamReader("SalData.csv");
            String line = sr.ReadLine();
            line = sr.ReadLine();
            while (line != null)
            {
                String[] elements = line.Split(",");
                Individual temp = new Individual(double.Parse(elements[0]), double.Parse(elements[1]), double.Parse(elements[2]), double.Parse(elements[3]), double.Parse(elements[4]), double.Parse(elements[5]), double.Parse(elements[6]), double.Parse(elements[7]));
                AddToList(temp);
                line = sr.ReadLine();
            }
            sr.Close();
        }

        public static void Normalize(Individual[] individuals)
        {
            minSalary = individuals.Min(i => i.Salary);
            maxSalary = individuals.Max(i => i.Salary);

            foreach (var property in typeof(Individual).GetProperties().Where(p => p.Name != "Salary"))
            {
                double min = individuals.Min(i => (double)property.GetValue(i));
                double max = individuals.Max(i => (double)property.GetValue(i));

                foreach (var individual in individuals)
                {
                    double value = (double)property.GetValue(individual);
                    property.SetValue(individual, (value - min) / (max - min));
                }
            }

            foreach (var individual in individuals)
            {
                individual.Salary = (individual.Salary - minSalary) / (maxSalary - minSalary);
            }
        }

        public static double DenormalizeSalary(double normalizedSalary)
        {
            return normalizedSalary * (maxSalary - minSalary) + minSalary;
        }
    }

    public class Individual
    {
        public double Salary { get; set; }
        public double Education { get; set; }
        public double NrSupervise { get; set; }
        public double NrPositionHeld { get; set; }
        public double PercResponsibility { get; set; }
        public double NumChildren { get; set; }
        public double Age { get; set; }
        public double YearsExperience { get; set; }

        // Constructor for training with salary
        public Individual(double salary, double education, double nrSupervise, double nrPositionHeld,
                          double percResponsibility, double numChildren, double age, double yearsExperience)
        {
            Salary = salary;
            Education = education;
            NrSupervise = nrSupervise;
            NrPositionHeld = nrPositionHeld;
            PercResponsibility = percResponsibility;
            NumChildren = numChildren;
            Age = age;
            YearsExperience = yearsExperience;
        }

        // Constructor for prediction without salary
        public Individual(double education, double nrSupervise, double nrPositionHeld,
                          double percResponsibility, double numChildren, double age, double yearsExperience)
        {
            Education = education;
            NrSupervise = nrSupervise;
            NrPositionHeld = nrPositionHeld;
            PercResponsibility = percResponsibility;
            NumChildren = numChildren;
            Age = age;
            YearsExperience = yearsExperience;
        }
    }

    public class Perceptron
    {
        private double[] weights;
        private double bias;
        private double learningRate;
        public List<double> MseHistory { get; private set; }
        public List<double> SseHistory { get; private set; }

        public Perceptron(int inputSize, double learningRate = 0.01)
        {
            weights = new double[inputSize];
            bias = 0.0;
            this.learningRate = learningRate;
            MseHistory = new List<double>();
            SseHistory = new List<double>();
            InitializeWeights();
        }

        private void InitializeWeights()
        {
            Random rnd = new Random();
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = rnd.NextDouble(); // Initialize weights randomly between 0 and 1
            }
            bias = rnd.NextDouble(); // Initialize bias randomly between 0 and 1
        }

        public void Train(Individual[] trainingData, int epochs, Individual[] validationData)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var individual in trainingData)
                {
                    double[] inputs = { individual.Education, individual.NrSupervise, individual.NrPositionHeld,
                                        individual.PercResponsibility, individual.NumChildren, individual.Age,
                                        individual.YearsExperience };

                    double target = individual.Salary;

                    double prediction = Predict(inputs);

                    double error = target - prediction;

                    // Update weights and bias
                    for (int i = 0; i < weights.Length; i++)
                    {
                        weights[i] += learningRate * error * inputs[i];
                    }

                    bias += learningRate * error;
                }

                // Evaluate MSE and SSE for the current epoch
                double mse = EvaluateMSE(validationData);
                double sse = EvaluateSSE(validationData);

                MseHistory.Add(mse);
                SseHistory.Add(sse);
            }
        }

        public double Predict(double[] inputs)
        {
            double sum = bias;
            for (int i = 0; i < weights.Length; i++)
            {
                sum += weights[i] * inputs[i];
            }
            return sum;
        }

        public double EvaluateMSE(Individual[] validationData)
        {
            double totalError = 0.0;
            foreach (var individual in validationData)
            {
                double[] inputs = { individual.Education, individual.NrSupervise, individual.NrPositionHeld,
                                    individual.PercResponsibility, individual.NumChildren, individual.Age,
                                    individual.YearsExperience };

                double target = individual.Salary;
                double prediction = Predict(inputs);
                totalError += Math.Pow(target - prediction, 2); // Mean Squared Error (MSE)
            }
            return totalError / validationData.Length;
        }

        public double EvaluateSSE(Individual[] validationData)
        {
            double totalError = 0.0;
            foreach (var individual in validationData)
            {
                double[] inputs = { individual.Education, individual.NrSupervise, individual.NrPositionHeld,
                                    individual.PercResponsibility, individual.NumChildren, individual.Age,
                                    individual.YearsExperience };

                double target = individual.Salary;
                double prediction = Predict(inputs);
                totalError += Math.Pow(target - prediction, 2); // Sum Squared Error (SSE)
            }
            return totalError;
        }
    }
}