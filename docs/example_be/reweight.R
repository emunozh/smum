library('GREGWT')

census <- read.csv('temp/toR_census.csv')
#View(census)
survey <- read.csv('temp/toR_survey.csv')
#View(survey)

simulation_data <- prepareData(
  census, survey,
  verbose=TRUE,
  align=data.frame(pop.1=c(1,3), pop.2=c(4,6)),
  breaks=c(8,21,22,24,31,32,33),
  survey_weights='w',
  convert = TRUE,
  pop_total_col='pop',
  census_categories=seq(3, 36),
  survey_categories=c(3,4,5,6,7)
  )

Weights <- GREGWT(
    data_in=simulation_data,
    max_iter = 1000,
    use_ginv=TRUE,
    #output_log = TRUE,
    #area_code='internal'
    )
#print(Weights)
#plot(Weights)

write.csv(Weights$final_weights, file = "temp/new_weights.csv")
