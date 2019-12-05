library(tidyverse)
library(ggplot2)
library(lubridate)
library(wbstats)
library(gridExtra)
library(caret)
library(recipes)

wb_data <- wb(indicator = c("SL.TLF.CACT.FE.ZS","NY.GDP.PCAP.CD",
                            "SE.ADT.LITR.FE.ZS","SP.DYN.TFRT.IN",
                            "SG.TIM.UWRK.FE", "SE.COM.DURS",
                            "SE.SEC.CUAT.LO.FE.ZS", 
                            "SE.TER.CUAT.BA.FE.ZS", 
                            "SE.XPD.TOTL.GB.ZS"), 
              startdate = 2005,
              enddate = 2018,
              country = "countries_only") %>% as_tibble()

wb_data <- wb_data %>% mutate(year=as.numeric(date)) %>%
  select(iso3c,year,value,indicatorID,country) %>%
  spread(key = indicatorID, value = "value")

wb_data <- wb_data %>% select(country=country,ccode=iso3c,year=year,
                              f_lfpr=SL.TLF.CACT.FE.ZS,
                              gdp_pcap=NY.GDP.PCAP.CD,
                              f_literacy_rate=SE.ADT.LITR.FE.ZS,
                              com_edu=SE.COM.DURS,
                              f_lower_sec_edu=SE.SEC.CUAT.LO.FE.ZS,
                              f_ba_edu=SE.TER.CUAT.BA.FE.ZS,
                              gov_exp_edu=SE.XPD.TOTL.GB.ZS,
                              f_ptime_chores=SG.TIM.UWRK.FE,
                              fertility_rate=SP.DYN.TFRT.IN)

wb_data <- wb_data %>% filter(!is.na(f_lfpr))

#split data into training and test data set 
train_data <- wb_data %>% filter(year < 2016)
test_data <- wb_data %>% filter(year > 2015)

p1 <- ggplot(data=train_data) +
  geom_point(aes(x=gdp_pcap, y=f_lfpr,alpha=0.3))+
  theme_minimal()+
  theme(legend.position = "none",
        plot.title = element_text(face="bold",hjust=0.5))+
  labs(title = "Female LFPR vs GDP Per Capita",
       x="GDP Per Capita",
       y="Female LFPR")


p2 <- ggplot(data=train_data)+
  geom_point(aes(x=fertility_rate,y=f_lfpr,alpha=0.8,color="red"))+
  theme_minimal()+
  theme(legend.position = "none",
        plot.title = element_text(face="bold",hjust=0.5))+
  labs(title = "Female LFPR vs Fertility Rate",
       x="Fertility Rate",
       y="Female LFPR")

p3 <- ggplot(data=train_data,aes(x=f_lower_sec_edu,y=f_lfpr))+
  geom_point(color="darkgreen",alpha=0.8)+
  geom_smooth(method='lm',se=FALSE)+
  theme_minimal()+
  theme(legend.position = "none",
        plot.title = element_text(face="bold",hjust=0.5))+
  labs(title = "Female LFPR vs Secondary Education",
       x="% female aged 25+ completed at least lower secondary education",
       y="Female LFPR")

p4 <- ggplot(data=train_data,aes(x=com_edu,y=f_lfpr))+
  geom_point(color="steelblue",alpha=0.8)+
  theme_minimal()+
  theme(legend.position = "none",
        plot.title = element_text(face="bold",hjust=0.5))+
  labs(title = "Female LFPR vs Compulsory Education",
       x="Years of Compulsory Education",
       y="Female LFPR")

#Reprocess the data
train_data1 <- train_data %>% select(-ccode,-country)

my_recipe <- recipe(f_lfpr~., data=train_data1) %>%
  step_knnimpute(all_predictors()) %>%
  step_range(all_numeric()) %>%
  prep()

train_data2 <- bake(my_recipe, train_data)
test_data2 <- bake(my_recipe, test_data)

#Set cross-validation
set.seed(5678)
folds <- createFolds(train_data2$f_lfpr,k=5)
sapply(folds,length)
cross_val <- trainControl(method = "cv", index = folds)

#K-nearest Neighbors(KNN)
knn_tune = expand.grid(k=c(1,5,10,50))

model_knn <- train(f_lfpr~.,data = train_data2, method="knn",
                   metric="RMSE",trControl=cross_val, 
                   tuneGrid=knn_tune)

#Random Forest(ranger)
model_rf <- train(f_lfpr~., data=train_data2,method="ranger",
                  metric="RMSE", trControl=cross_val)

model_all <- list(knn=model_knn, rf=model_rf)
resamples(model_all)
p5 <- dotplot(resamples(model_all), metric= "RMSE")

#Test the out-of-sample predictive performance
prediction <-predict(model_knn, newdata = test_data2)
rmse = sqrt(sum((test_data2$f_lfpr - prediction)^2/nrow(test_data2)))
rmse