library('GREGWT')

#setwd('~/workspace/R/GREGWT/src/')
#source("GREGWT.R")
#source("PrepareData.R")
#setwd('~/workspace/python/urbanmetabolism_doc/_examples/')

census <- read.csv('temp/toR_census.csv')
#View(census)
survey <- read.csv('temp/toR_survey.csv')
#View(survey)

simulation_data <- prepareData(
  census, survey,
  align=data.frame(pop=c(1,7)),
  breaks=c(2,4,6,19,22),
  #verbose=TRUE,
  survey_weights='w',
  pop_total_col='pop',
  census_categories=seq(3, 31),
  survey_categories=c(2,3,4,5)
  )

Weights <- GREGWT(
    data_in=simulation_data,
    max_iter = 1000,
    use_ginv=TRUE,
    #area_code='internal'
    )
#print(Weights)
#plot(Weights)

write.csv(Weights$final_weights, file = "temp/new_weights.csv")
