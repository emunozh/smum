library('GREGWT')

#setwd('~/workspace/R/GREGWT/src/')
#source("GREGWT.R")
#source("PrepareData.R")

setwd('~/workspace/python/urbanmetabolism/population/')

census <- read.csv('temp/toR_census.csv')
#View(census)
survey <- read.csv('temp/toR_survey.csv')
#View(survey)

simulation_data <- prepareData(
  census, survey,
  align=data.frame(pop=c(1,11)),
  breaks=c(2,4,6,8,10,20,22,24,33),
  #verbose=TRUE,
  survey_weights='w',
  pop_total_col='pop',
  census_categories=seq(2, dim(census)[2]-1),
  survey_categories=seq(2, dim(survey)[2]-1)
  )

Weights <- GREGWT(data_in=simulation_data, use_ginv=TRUE)
#print(Weights)
#plot(Weights)

write.csv(Weights$final_weights, file = "temp/new_weights.csv")
