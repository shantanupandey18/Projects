# Pandey Shantanu, Parekh Gunjal, Shukla Akash, Yaji Manu
# MIS 545 - Group 22
# ProjectGroup22PandeyParekhShuklaYaji.R
# This code evaluates different models on VehicleInsurance.csv file. 
# We perform Data pre-processing, data exploration and feature engineering. 
# Created the Training and Testing datasets.We have built KNN, Logistic 
# regression, Decision Tree, Naive Bayes and Neural network models. 

# Install required packages
# comment the install commands once they are installed
# install.packages("tidyverse")
# install.packages("fastDummies")
# install.packages("rpart.plot")
# install.packages("corrplot")
# install.packages("ggplot2")
# install.packages("smotefamily")
# install.packages("e1071")
# install.packages("class")
# install.packages("neuralnet")

# Load all the libraries
library(ggplot2)
library(tidyverse)
library(fastDummies)
library(e1071)
library(corrplot)
library(smotefamily)
library(rpart.plot)
library(class)
library(neuralnet)

# Set working directory
setwd("C:/Users/ual-laptop/Downloads/RStudio/545 Project")

# Read the CSV File and set datatypes
vehicleInsurance <- read_csv(file="VehicleInsurance.csv",
                             col_types = "fninicinnni",
                             col_names = TRUE)

# Display vehicleInsurance
print(vehicleInsurance)

# Summary of vehicleInsurance
print(summary(vehicleInsurance))

# Structure of vehicleInsurance
print(str(vehicleInsurance))

# Create displayAllHistograms using function
displayAllHistograms <- function(tibbleDataset){
  tibbleDataset %>% keep(is.numeric) %>% gather() %>% ggplot()+
    geom_histogram(mapping = aes(x=value,fill=key),
                   color = "black") + facet_wrap(~key,scales = "free") +
    theme_minimal()
}

# Display Histogram plots Initial
displayAllHistograms(vehicleInsurance)

# Interesting Queries
# Query to get data about Vehicles which are not previously insured
query1 <- vehicleInsurance %>%
  filter(Previously_Insured == 0) %>%
  group_by(Vehicle_Age, Vehicle_Damage) %>%
  tally() %>%
  mutate(Vehicle_Damage = ifelse(Vehicle_Damage == 0, "Not Damaged", "Damaged"))
print(query1)

# Query to calculate Age-wise Annual Premium
query2 <- vehicleInsurance %>%
  filter(Age >= 21) %>%
  filter(Age <=75) %>%
  group_by(Age) %>%
  summarize(AverageAnnualPremium = mean(Annual_Premium))
print(query2)

# Query to Count Number of Vehicle Damage cases based on Gender
query3 <- vehicleInsurance %>%
  filter(Vehicle_Damage == 1) %>%
  group_by(Gender) %>%
  tally()
print(query3)

# Plot for Query 1
ggplot(query1, aes(fill=Vehicle_Damage, y=n, x=Vehicle_Age)) +
  geom_bar(position=position_dodge(), stat="identity", width = 0.5) +
  xlab("Vehicle Age") +
  ylab("Number of Incidents") +
  geom_text(position=position_dodge(width= 0.5), aes(label = n), vjust = -0.2) +
  ggtitle("Number of incidents on previously uninsured vehicles by Vehicle Age")

# Plot for Query 2
ggplot(query2, aes(x= Age, y= AverageAnnualPremium)) +
  geom_line(color="blue") +
  xlab("Age (Years)") +
  ylab("Average Annual Premium in INR") +
  geom_smooth(method = lm, color="red", level=0) +
  ggtitle("Average Annual Premium for Customers (21 to 75 years)")

# Plot for Query 3
ggplot(data=query3, aes(x=Gender, y=n)) +
  geom_bar(stat="identity", color = "blue", fill = "blue", width = 0.2) +
  xlab("Gender") +
  ylab("Count") +
  ggtitle("Vehicle Damage by Gender") +
  geom_text(aes(label = n), vjust = -0.2)

# Remove rows with null
drop_na(vehicleInsurance)

# Data Exploration
# Dummy code Gender
vehicleInsurance <- vehicleInsurance %>%
  mutate(Gender = ifelse(Gender == "Male", 0,1))

# Set ranges to Age
vehicleInsurance <- vehicleInsurance %>%
  mutate(Age = ifelse(Age <=30, 1,
                      ifelse(Age <= 40, 2,
                             ifelse(Age <= 50, 3, 
                                    ifelse(Age <= 60, 4, 
                                           ifelse(Age <= 70, 5, 
                                                  ifelse(Age <= 80, 6,
                                                         ifelse(
                                                           Age <= 90, 7, 8)
                                                         )))))))

# Set range to Vehicle age
vehicleInsurance <- vehicleInsurance %>%
  mutate(Vehicle_Age = ifelse(Vehicle_Age == "< 1 Year", 0,
                              ifelse(Vehicle_Age == "1-2 Year", 1, 2)))

# Correlation Plot
corrplot(cor(vehicleInsurance),
         method = "number",
         type="lower",
         number.cex = 0.5)

# Showing the correlations on console
round(cor(vehicleInsurance %>%
            keep(is.numeric)),2)

# Create labels for KNN
vehicleInsuranceLabels <- vehicleInsurance %>%
  select(Response)

# Feature Engineering
# Outliers check
outlierMin<- quantile(vehicleInsurance$Annual_Premium, 0.25) -
  (IQR(vehicleInsurance$Annual_Premium) * 1.5)
outlierMax <- quantile(vehicleInsurance$Annual_Premium, 0.75) +
  (IQR(vehicleInsurance$Annual_Premium) * 1.5)

# Removing outliers
outliers <- vehicleInsurance %>%
  filter(Annual_Premium < outlierMin | Annual_Premium > outlierMax)
vehicleInsurance <- vehicleInsurance %>%
  filter(Annual_Premium > outlierMin & Annual_Premium < outlierMax)

# Dummy code for vehicle gender
vehicleInsurance <- dummy_cols(vehicleInsurance,select_columns = "Gender")

# Dummy code for vehicle age
vehicleInsurance <- dummy_cols(vehicleInsurance,select_columns = "Vehicle_Age")

# Dummy code for age
vehicleInsurance <- dummy_cols(vehicleInsurance,select_columns = "Age")

# Dummy code for Driving license
vehicleInsurance <- dummy_cols(
  vehicleInsurance,select_columns = "Driving_License")

# Dummy code for Previously insured
vehicleInsurance <- dummy_cols(
  vehicleInsurance,select_columns = "Previously_Insured")

# Dummy code for Vehicle Damage
vehicleInsurance <- dummy_cols(
  vehicleInsurance,select_columns = "Vehicle_Damage")

# Convert Vintage to Years
vehicleInsurance <-  vehicleInsurance %>%
  mutate(Vintage = Vintage/365)

# Normalize using min-max Annual_Premium
vehicleInsurance <- vehicleInsurance %>%
  mutate(Annual_Premium = (Annual_Premium - min(Annual_Premium))/
           (max(Annual_Premium) - min(Annual_Premium)))

# Remove columns to reduce multicollinearity
vehicleInsurance <- vehicleInsurance %>%
  select(-Region_Code, -Vehicle_Age, -Policy_Sales_Channel, -Age)
vehicleInsurance <- vehicleInsurance %>%
  select(-Vehicle_Age_2)
vehicleInsurance <- vehicleInsurance %>%
  select(-Driving_License, -Driving_License_1)
vehicleInsurance <- vehicleInsurance %>%
  select(-Previously_Insured, -Previously_Insured_1)
vehicleInsurance <- vehicleInsurance %>%
  select(-Vehicle_Damage, -Vehicle_Damage_1)
vehicleInsurance <- vehicleInsurance %>%
  select(-Gender, -Gender_1, -Vehicle_Age_1,-Age_1,-Previously_Insured_0)

# Display Histogram final 
displayAllHistograms(vehicleInsurance)

# Display Correlation plot and create training & testing set
# Correlation plot
corrplot(cor(vehicleInsurance),
         method = "number",
         type="lower",
         number.cex = 0.5)

# Show correlations in the console
round(cor(vehicleInsurance%>%
      keep(is.numeric)),2)

# Set seed, create sample set
set.seed(125)
sampleset <- sample(nrow(vehicleInsurance),
                    round(nrow(vehicleInsurance)*0.75),
                    replace = FALSE)

# Create a training data set for 75% of sample set
vehicleInsuranceTraining <- vehicleInsurance[sampleset,]
vehicleInsuranceLabelTraining <- vehicleInsuranceLabels[sampleset,]

# Create a test data set for 25% of sample set
vehicleInsuranceTesting <- vehicleInsurance[-sampleset,]
vehicleInsuranceLabelTesting <- vehicleInsuranceLabels[-sampleset,]

# Check class imbalance in Response
dim(vehicleInsuranceTraining %>% filter(Response == 1))
dim(vehicleInsuranceTraining %>% filter(Response == 0))
summary(vehicleInsuranceTraining$Response)

# Deal with class imbalance in training dataset
vehicleInsuranceTrainingSmoted <-
  tibble(SMOTE(X = data.frame(vehicleInsuranceTraining),
               target = vehicleInsuranceTraining$Response,
               dup_size = 4)$data)

# remove Class column from smoted set
vehicleInsuranceTrainingSmoted <- vehicleInsuranceTrainingSmoted %>%
  select(-class)

# Run a summary and structure on the vehicleInsuranceTrainingSmoted tibble
print(summary(vehicleInsuranceTrainingSmoted))
print(str(vehicleInsuranceTrainingSmoted))

# Generate models
# Neural network
vehicleInsuranceNeuralNet <- neuralnet(
  formula = Response ~.,
  data = vehicleInsuranceTrainingSmoted,
  hidden = 3,
  act.fct = "logistic",
  linear.output = FALSE)

# Logistic Regression
vehicleInsuranceGlm <- glm(data=vehicleInsuranceTrainingSmoted,
                           family = "binomial",
                           formula = Response~.)

# Naive-Bayes: smoothing and model generation
# Creating bins
vehicleInsurance_bin <- vehicleInsuranceTrainingSmoted
vehicleInsurance_bin<- vehicleInsurance_bin %>% 
  mutate(Annual_Premium = round(Annual_Premium,digits = 2))
vehicleInsurance_bin<-drop_na(vehicleInsurance_bin)
vehicleInsurance_bin <- vehicleInsurance_bin %>% mutate(
  Annual_Premium = cut(Annual_Premium, breaks=c(-0.1, 0.25, 0.50, 0.75,1.00)))
vehicleInsurance_bin <- vehicleInsurance_bin %>% mutate(
  Vintage = cut(Vintage, breaks=c(0, 0.25, 0.50, 0.75,1.00)))

# Generate the Naive Bayes Model
vehicleInsuranceNaiveBayes <- naiveBayes(
  formula = Response~.,
  data = vehicleInsurance_bin,
  laplace = 1)

# generate KNN model
vehicleInsuranceKNN <- knn(train= vehicleInsuranceTraining,
                           test=vehicleInsuranceTesting,
                           cl= vehicleInsuranceLabelTraining$Response,
                           k=42)

# Decision tree
# Run the model with different complexity and uncomment other Decision Tree 
# models. Only have one decision tree model running at any time.
vehicleInsuranceDecisionTreeModel <- rpart(formula = Response~.,
                                           method = "class",
                                           cp = 0.01,
                                           data = 
                                           vehicleInsuranceTrainingSmoted)

# vehicleInsuranceDecisionTreeModel <- rpart(formula = Response~.,
#                                            method = "class",
#                                            cp = 0.02,
#                                            data = 
#                                            vehicleInsuranceTrainingSmoted)

# vehicleInsuranceDecisionTreeModel <- rpart(formula = Response~.,
#                                            method = "class",
#                                            cp = 0.04,
#                                            data = 
#                                            vehicleInsuranceTrainingSmoted)

# Model Summaries
summary(vehicleInsuranceGlm)
summary(vehicleInsuranceNaiveBayes)
summary(vehicleInsuranceDecisionTreeModel)
summary(vehicleInsuranceKNN)
summary(vehicleInsuranceNeuralNet)

## Generate prediction for models
# GLM model prediction
vehicleInsuranceGlmPrediction <- predict(vehicleInsuranceGlm,
                                      vehicleInsuranceTesting,
                                      type = "response")
vehicleInsuranceGlmPrediction <- ifelse(
  vehicleInsuranceGlmPrediction >= 0.5,1,0)

# Naive-Bayes Prediction
vehicleInsuranceNaiveBayesPrediction <- predict(vehicleInsuranceNaiveBayes,
                                                vehicleInsuranceTesting,
                                                type = "class")

# Decision Tree prediction
vehicleInsuranceDecisionTreePrediction <- predict(
  vehicleInsuranceDecisionTreeModel,vehicleInsuranceTesting,type = "class")

# Probabilities for Neural Network
vehicleInsuranceNeuralNetProbabilities <- compute(vehicleInsuranceNeuralNet,
                                                  vehicleInsuranceTesting)
vehicleInsuranceNeuralNetPrediction <-
  ifelse(vehicleInsuranceNeuralNetProbabilities$net.result> 0.5,1,0)

# Prediction summary
summary(vehicleInsuranceGlmPrediction)
summary(vehicleInsuranceNaiveBayesPrediction)
summary(vehicleInsuranceDecisionTreePrediction)
summary(vehicleInsuranceKNN)
print(vehicleInsuranceNeuralNetProbabilities$net.result)

# Generate Plots
# Display  plots
# Make sure you have enough figure margins in the Plots tab in the bottom right 
# hand side of the RStudio
rpart.plot(vehicleInsuranceDecisionTreeModel)
plot(vehicleInsuranceNeuralNet)

# Generate confusion matrix & calculate predictive accuracy
# Create a confusion matrix for Logistic regression model
vehicleInsuranceGlmConfusionMatrix <- table(vehicleInsuranceTesting$Response,
                                            vehicleInsuranceGlmPrediction)

# Create a confusion matrix for Naive Bayes Model
vehicleInsuranceNaiveBayesConfusionMatrix <- table(
  vehicleInsuranceTesting$Response,vehicleInsuranceNaiveBayesPrediction)

# Create a confusion matrix for Decision Tree Model
vehicleInsuranceDecisionTreeConfusionMatrix <- table(
  vehicleInsuranceTesting$Response,vehicleInsuranceDecisionTreePrediction)

# Create a confusion matrix for KNN Model
vehicleInsuranceKNNConfusionMatrix <- table(
  vehicleInsuranceTesting$Response,vehicleInsuranceKNN)

# Create a confusion matrix for Neural Network Model
vehicleInsuranceNeuralNetConfusionMatrix <- table(
  vehicleInsuranceTesting$Response,vehicleInsuranceNeuralNetPrediction)

# Calculate false positive
# Calculate flase positive rate for Logistic Regression Model
vehicleInsuranceGlmConfusionMatrix[1,2] /
  (vehicleInsuranceGlmConfusionMatrix[1,1] +
     vehicleInsuranceGlmConfusionMatrix[1,2])

# Calculate flase positive rate for Naive Bayes Model
vehicleInsuranceNaiveBayesConfusionMatrix[1,2] /
  (vehicleInsuranceNaiveBayesConfusionMatrix[1,1] +
     vehicleInsuranceNaiveBayesConfusionMatrix[1,2])

# Calculate flase positive rate for Decision Tree Model
vehicleInsuranceDecisionTreeConfusionMatrix[1,2] /
  (vehicleInsuranceDecisionTreeConfusionMatrix[1,1] +
     vehicleInsuranceDecisionTreeConfusionMatrix[1,2])

# Calculate flase positive rate for KNN model
vehicleInsuranceKNNConfusionMatrix[1,2] /
  (vehicleInsuranceKNNConfusionMatrix[1,1] +
     vehicleInsuranceKNNConfusionMatrix[1,2])

# Calculate flase positive rate for Neural Network model
vehicleInsuranceNeuralNetConfusionMatrix[1,2] /
  (vehicleInsuranceNeuralNetConfusionMatrix[1,1] +
     vehicleInsuranceNeuralNetConfusionMatrix[1,2])

# Calculate false negative
# Calculate false negative for Logistic Regression Model
vehicleInsuranceGlmConfusionMatrix[2,1] /
  (vehicleInsuranceGlmConfusionMatrix[2,1] +
     vehicleInsuranceGlmConfusionMatrix[2,2])

# Calculate false negative for Naive Bayes Model
vehicleInsuranceNaiveBayesConfusionMatrix[2,1] /
  (vehicleInsuranceNaiveBayesConfusionMatrix[2,1] +
     vehicleInsuranceNaiveBayesConfusionMatrix[2,2])

# Calculate false negative for Decision Tree Model
vehicleInsuranceDecisionTreeConfusionMatrix[2,1] /
  (vehicleInsuranceDecisionTreeConfusionMatrix[2,1] +
     vehicleInsuranceDecisionTreeConfusionMatrix[2,2])

# Calculate false negative for KNN Model
vehicleInsuranceKNNConfusionMatrix[2,1] /
  (vehicleInsuranceKNNConfusionMatrix[2,1] +
     vehicleInsuranceKNNConfusionMatrix[2,2])

# Calculate false negative for Neural Network Model
vehicleInsuranceNeuralNetConfusionMatrix[2,1] /
  (vehicleInsuranceNeuralNetConfusionMatrix[2,1] +
     vehicleInsuranceNeuralNetConfusionMatrix[2,2])

# Display the confusion matrix for all models
print(vehicleInsuranceGlmConfusionMatrix)
print(vehicleInsuranceNaiveBayesConfusionMatrix)
print(vehicleInsuranceDecisionTreeConfusionMatrix)
print(vehicleInsuranceKNNConfusionMatrix)
print(vehicleInsuranceNeuralNetConfusionMatrix)

# Predictive accuracies 
# Calculate predictive accuracy for Logistice Resgression Model
vehicleInsuranceGlmPredictiveAccuracy <- sum(
  diag(vehicleInsuranceGlmConfusionMatrix))/nrow(vehicleInsuranceTesting)

# Calculate predictive accuracy for Naive Bayes Model
vehicleInsuranceNaiveBayesPredictiveAccuracy <- sum(diag(
  vehicleInsuranceNaiveBayesConfusionMatrix))/
  nrow(vehicleInsuranceTesting)

# Calculate predictive accuracy for Decision Tree Model
vehicleInsuranceDecisionTreePredictiveAccuracy <- sum(diag(
  vehicleInsuranceDecisionTreeConfusionMatrix))/
  nrow(vehicleInsuranceTesting)

# Calculate predictive accuracy for KNN Model
vehicleInsuranceKNNPredictiveAccuracy <- sum(diag(
  vehicleInsuranceKNNConfusionMatrix))/
  nrow(vehicleInsuranceTesting)

# Calculate predictive accuracy for Neural Net Model
vehicleInsuranceNeuralNetPredictiveAccuracy <- sum(diag(
  vehicleInsuranceNeuralNetConfusionMatrix))/
  nrow(vehicleInsuranceTesting)

# Odds ratios for all attributes given to Logistice Regression
exp(coef(vehicleInsuranceGlm)["Annual_Premium"])
exp(coef(vehicleInsuranceGlm)["Vintage"])
exp(coef(vehicleInsuranceGlm)["Vehicle_Age_0"])
exp(coef(vehicleInsuranceGlm)["Age_2"])
exp(coef(vehicleInsuranceGlm)["Age_3"])
exp(coef(vehicleInsuranceGlm)["Age_4"])
exp(coef(vehicleInsuranceGlm)["Age_5"])
exp(coef(vehicleInsuranceGlm)["Age_6"])
exp(coef(vehicleInsuranceGlm)["Driving_License_0"])
exp(coef(vehicleInsuranceGlm)["Vehicle_Damage_0"])

# Display predictive accuracy for all models
print(vehicleInsuranceGlmPredictiveAccuracy)
print(vehicleInsuranceNaiveBayesPredictiveAccuracy)
print(vehicleInsuranceDecisionTreePredictiveAccuracy)
print(vehicleInsuranceKNNPredictiveAccuracy)
print(vehicleInsuranceNeuralNetPredictiveAccuracy)

# Find optimal K Value
# Create a matrix of k-values with their predictive accuracy
kValueMatrix<- matrix(data= NA,
                      nrow=0,
                      ncol=2)

# Assign column names to the matrix
colnames(kValueMatrix)<- c("kvalue", "Predictive Accuracy")

# Loop through odd values of k from 1 up to the number of records
# in the training dataset
for(kValue in 1:100){
  
  # Only calculate predictive accuracy if the k value is odd
  if(kValue %% 2 != 0){
    
    # Generate the model
    vehicleInsuranceKNN <- knn(train= vehicleInsuranceTraining,
                               test=vehicleInsuranceTesting,
                               cl= vehicleInsuranceLabelTraining$Response,
                               k=kValue)
    
    # Generate the confusion matrix
    vehicleInsuranceKNNConfusionMatrix<- table(
      vehicleInsuranceTesting$Response,vehicleInsuranceKNN)
    
    # Calculate predictive accuracy
    vehicleInsuranceKNNPredictiveAccuracy <- sum(
      diag(vehicleInsuranceKNNConfusionMatrix))/ nrow(vehicleInsuranceTesting)
    
    # Add a new row to the kValueMatrix
    kValueMatrix<- rbind(kValueMatrix, c(kValue,
                                         vehicleInsuranceKNNPredictiveAccuracy))
  }
}

# Display k value matrix
print(kValueMatrix)
# The best values for K would be from 15 to 99 since the 
#predictive accuracy for these values is the highest i.e. 0.8571429