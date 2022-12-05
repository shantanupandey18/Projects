# Econometrics Project
# World Happiness Report

# Install necessary packages
# install.packages("tidyverse")
# install.packages("corrplot")
install.packages("ggplot2")
# install.packages("ggthemes")
# install.packages("fastDummies")
install.packages("plm")
install.packages("lmtest")
install.packages("dplyr")
install.packages("moonBook")
install.packages("ggeffects")
install.packages("car")
install.packages("ggplot")


# Load the package
library(car)
library(tidyverse)
library(corrplot)
library(ggplot2)
library(ggthemes)
library(fastDummies)
library(plm)
library(lmtest)
library(dplyr)
library(moonBook)
library(ggeffects)
library(tsibbledata)


# set working directory
setwd("C:/Users/ual-laptop/Documents/Econometrics/Project/Happiness Report")

# Import Datasets
WHR <- read_csv(file="WHR_2015-2022.csv",
                    col_types = "cnnnnnnnnni",
                    col_names = TRUE)

WHR_1 <- read_csv(file="WHR_1.csv",
                col_types = "nnnnnnnnni",
                col_names = TRUE)

# Display tibble
print(WHR)
print(WHR_1)

# Structure
str(WHR)

# Summary
summary(WHR)

# Filter top 20 countries year wise in a new tibble
WHR_20 <- as_tibble(WHR%>%
  filter(WHR$HappinessScore>=7.5))

# plot data

ggplot(data = WHR_20, mapping = aes(x = Year, y = HappinessScore, color = Country)) + # specify data, x-axis, y-axis and grouping variable
  geom_line() + # a line per group
  geom_hline(yintercept = 7.25) + geom_vline(xintercept = 2015)+ 
  geom_point() + # points per group
  theme_hc() +  # a ggtheme, similar to your example
  labs(title = "Happiness Score Top 5 2015-2022", # plot title
       subtitle = "", # plot subtitle
       caption = "") + # plot caption
  theme(legend.position = "bottom", # move legend to the right hand side of the plot
        axis.title.x = element_text("Year"), # remove x axis title
        axis.title.y = element_text("Happiness_Score"), # remove y axis title
        legend.title = element_text("Country:"), # remove legend title
        plot.title = element_text(size = 20, color = "gray40"), # change size and color of plot title
        plot.subtitle = element_text(color = "grey"), # change color of subtitle
        plot.caption = element_text(color = "grey", hjust = 0)) + # change color of caption and left-align
  scale_y_continuous(breaks = seq(7, 8, by = 0.25)) + # specify min, max and break distance for y axis
  scale_x_continuous(breaks = seq(2015, 2022, by = 1)) + # specify min, max and break distance for x axis
  expand_limits(y = c(7.5, 8))

# Regression on happiness score using freedom
reg1 <- lm(WHR$HappinessScore~WHR$Freedom, data = WHR)
summary(reg1)

# plot the best fit line
plot(WHR$HappinessScore,WHR$Freedom,
     pch=16,cex=1.3,col="light blue",
     xlab="Happiness Score",ylab = "Freedom")

abline(0.020031,0.1043792)
 
ggplot(WHR, aes(x = WHR$HappinessScore, y = WHR$Family)) + 
  geom_point() +
  stat_smooth(method = "lm")

# Regression on multiple factors
reg2 <- lm(HappinessScore~Freedom + Economy_GDP_per_Capita +Family+
           HealthLifeExpectancy+ TrustGovernment_Corruption+
             Generosity + Dystopia_Residual, data = WHR)
summary(reg2)

# plot the best fit line
#equation1=function(x){coef(fit1)[2]*x+coef(fit1)[1]}
#equation2=function(x){coef(fit1)[2]*x+coef(fit1)[1]+coef(fit1)[3]}

#ggplot(radial,aes(y=NTAV,x=age,color=sex))+geom_point()+
#  stat_function(fun=equation1,geom="line",color=scales::hue_pal()(2)[1])+
#  stat_function(fun=equation2,geom="line",color=scales::hue_pal()(2)[2])

ggPredict(reg2,se=TRUE,interactive=TRUE)



#produce added variable plots
avPlots(reg2)

abline(0.020031,0.1043792)

# Corelation of features
cor_Eco<-cor(WHR$HappinessScore, WHR$Economy_GDP_per_Capita)
cor_Fam<-cor(WHR$HappinessScore, WHR$Family)
cor_Life<-cor(WHR$HappinessScore, WHR$HealthLifeExpectancy)
cor_Freedom<-cor(WHR$HappinessScore, WHR$Freedom)
cor_Trust<-cor(WHR$HappinessScore, WHR$TrustGovernment_Corruption)
cor_Gen<-cor(WHR$HappinessScore, WHR$Generosity)
cor_Dyst<-cor(WHR$HappinessScore, WHR$Dystopia_Residual)

print(cor_Eco)
print(cor_Fam)
print(cor_Life)
print(cor_Freedom)
print(cor_Trust)
print(cor_Gen)
print(cor_Dyst)

#cor(reg2, y = NULL, use = "everything",
#    method = c("pearson", "kendall", "spearman"))

# Variance for all features
var(WHR$HappinessScore, WHR$Economy_GDP_per_Capita, na.rm = FALSE)
var(WHR$HappinessScore, WHR$Family, na.rm = FALSE)
var(WHR$HappinessScore, WHR$HealthLifeExpectancy, na.rm = FALSE)
var(WHR$HappinessScore, WHR$Freedom, na.rm = FALSE)
var(WHR$HappinessScore, WHR$TrustGovernment_Corruption, na.rm = FALSE)
var(WHR$HappinessScore, WHR$Generosity, na.rm = FALSE)
var(WHR$HappinessScore, WHR$Dystopia_Residual, na.rm = FALSE)

# ggplot for features
HS_log <- log(WHR$HappinessScore)
HS_log1 <- as_tibble(HS_log)

Fam_log <- log(WHR$Family)
Fam_log1 <- as_tibble(Fam_log)
ggplot(WHR,aes(x=HS_log,y=Fam_log))+
              geom_point(color="dark grey")+
              geom_smooth(method=lm,color="red")+
              ggtitle("Happiness Score - Family")

GDP_log <- log(WHR$Economy_GDP_per_Capita)

ggplot(WHR,aes(x=HS_log,y=GDP_log))+
  geom_point(color="grey")+
  geom_smooth(method=lm,color="red")+
  ggtitle("Happiness Score - GDP")

HLE_log <- log(WHR$HealthLifeExpectancy)

ggplot(WHR,aes(x=HS_log,y=HLE_log))+
  geom_point(color="dark grey")+
  geom_smooth(method=lm,color="red")+
  ggtitle("Happiness Score - Health Life Expectancy")

GDP_log<- 
  discretize(WHR$Economy_GDP_per_Capita, method="interval", breaks=2)
summary(burn$DEATH)

ggplot(WHR,aes(x=HappinessScore,y=Freedom))+
  geom_point(color="dark grey")+
  geom_smooth(method=lm,color="red")+
  ggtitle("Happiness Score - Freedom")

ggplot(WHR,aes(x=HappinessScore,y=TrustGovernment_Corruption))+
  geom_point(color="dark grey")+
  geom_smooth(method=lm,color="red")+
  ggtitle("Happiness Score - Trust_Govt")

ggplot(WHR,aes(x=HappinessScore,y=Generosity))+
  geom_point(color="dark grey")+
  geom_smooth(method=lm,color="red")+
  ggtitle("Happiness Score - Generosity")

ggplot(WHR,aes(x=HappinessScore,y=Dystopia_Residual))+
  geom_point(color="dark grey")+
  geom_smooth(method=lm,color="red")+
  ggtitle("Happiness Score - Dystopia Residual")

# Tibble without country name and rank
corr_plot1 <-   select(WHR$HappinessScore,WHR$Economy_GDP_per_Capita)

# Correlation-plot
corrplot(cor(WHR_1),
         method = "number",
         type="lower")

WHR_1 <- dummy_cols(WHR_1,select_columns = "Y")

WHR_1 <- WHR_1 %>% select(-Y)



reg_dummy <- lm(HR~ FRDM + GDP +F+
             HLE+ Corr+
             GNST + DR, data = WHR_1)
avPlots(reg_dummy)

summary(reg_dummy)


# #WHR <- WHR %>%
#   mutate(WHR$HappinessRank = replace(WHR$HappinessRank,
#                                      is.na(WHR$HappinessRank),
#                                      median(WHR$HappinessRank,
#                                             na.rm = TRUE)))

#corrplot(cor(WHR$HappinessScore),method = "number",type = "lower")

#time_series_1 <- ts(WHR,frequency = 150,start=c(2015,1))
#time_series
#plot.ts(time_series_1)

#str(time_series)

#as.factor(WHR$Year)

# Mean and standard deviation
sd(WHR$Family)

WHR_1 <- read_csv(file="WHR_1.csv",
                  col_types = "nnnnnnnnni",
                  col_names = TRUE)

WHR_1_Dummy <- dummy_cols(WHR_1, select_columns = "Y")
WHR_No_Year <- WHR_1_Dummy %>% select(-Y) #, -HappinessRank, -Generosity, -Dystopia_Residual, -Family)

WHR_1_Regression <- lm(formula = HS~.,
                       data = WHR_No_Year)
m <- cor(WHR_No_Year)

corrplot(m, method = "number", type = "lower")

# Factor

WHR_2 <- read_csv(file="WHR_2.csv",
                  col_types = "fnnnnnnnnf",
                  col_names = TRUE)

# Dummy coded Year for time series analysis
dummy_reg <- lm(HS ~ WHR_1_Dummy$F+ WHR_1_Dummy$GDP + WHR_1_Dummy$FRDM + WHR_1_Dummy$HLE
                +WHR_1_Dummy$Corr   + WHR_1_Dummy$GNST  + Y - 1, data = WHR_1_Dummy)
dummy_reg
summary(dummy_reg)

avPlots(dummy_reg)

plm_reg<- plm(formula =HappinessScore ~ Economy_GDP_per_Capita+Family+ HealthLifeExpectancy+
                Freedom + TrustGovernment_Corruption+Generosity,
                      data = whr_5,
                      index = c("Year"), 
                      model = "within", 
                      effect = "twoways")
summary(plm_reg)

coeftest(plm_reg, vcov = vcovHC, type = "HC1")


panelr
install.packages("panelr")
library(panelr)

WHR_4 <- pdata.frame(WHR_2,index=NULL)

whr_5 %>% 
  line_plot(HappinessScore, 
            overlay = FALSE,
            subset.ids = whr_5$Country,
            add.mean = TRUE,
            mean.function = "loess"
  )

whr_5 <- panel_data(WHR_20, id = Country,wave=Year)
# plot data

ggplot(data = WHR_2, mapping = aes(x = Year, y = HappinessScore, color = Country)) + # specify data, x-axis, y-axis and grouping variable
  geom_line() + # a line per group
  geom_hline(yintercept = 6.75) + geom_vline(xintercept = 2015)+ 
  geom_point() + # points per group
  theme_hc() +  # a ggtheme, similar to your example
  labs(title = "Line Graph U.S.A vs Switzerland", # plot title
       subtitle = "", # plot subtitle
       caption = "") + # plot caption
  theme(legend.position = "bottom", # move legend to the right hand side of the plot
        axis.title.x = element_text("Year"), # remove x axis title
        axis.title.y = element_text("Happiness_Score"), # remove y axis title
        legend.title = element_text("Country:"), # remove legend title
        plot.title = element_text(size = 20, color = "gray40"), # change size and color of plot title
        plot.subtitle = element_text(color = "grey"), # change color of subtitle
        plot.caption = element_text(color = "grey", hjust = 0)) + # change color of caption and left-align
  scale_y_continuous(breaks = seq(6, 8, by = 0.25)) + # specify min, max and break distance for y axis
  scale_x_continuous(breaks = seq(2015, 2022, by = 1)) + # specify min, max and break distance for x axis
  expand_limits(y = c(7.5, 8))


# Normalise the data
install.packages("normalr")
library(normalr)

WHR_3 <- read_csv(file="WHR_2.csv",
                  col_types = "fnnnnnnnnf",
                  col_names = TRUE)

dummy_reg2 <- lm(HappinessScore ~ WHR_2$Freedom+ Economy_GDP_per_Capita + WHR_2$Family + WHR_2$HealthLifeExpectancy
                +WHR_2$TrustGovernment_Corruption+ WHR_2$Generosity + Country + Year - 1, data = WHR_2)
dummy_reg2
summary(dummy_reg2)


WHR_3$Happiness_score_norm<-normalise(WHR_3$HappinessScore)
mydata$dist_norm<-normalize(mydata$dist)
