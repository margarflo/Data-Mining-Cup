library(dplyr)
library(purrr)
library(magrittr)
library(tidyr)
library(mlr3verse)
library(future)
library(mlr3pipelines)
library(caret)
library(mlr3measures)
library(mlr3tuning)

# Read and prepare training data -----------------------------------------------

train_data <- read.csv("data/train_values.csv")

train_labels <- read.csv("data/train_labels.csv")

train_data_label <-
  inner_join(
    train_data,
    train_labels,
    by = "building_id"
  )


train_factors <-
  map_dfc(
    1:40,
    function(col){
      if(col > 8 | col == 5){
        out <- list(as.factor(train_data_label[, col]))
        names(out) <- names(train_data_label[col])
        return(out)
      } else{
        out <- list(train_data_label[, col])
        names(out) <- names(train_data_label[col])
        return(out)
      }
    }
  ) %>%
  as.data.frame() %>%
  select(-building_id)

# Load and prepare test data ---------------------------------------------------

test_data <-
  read.csv("data/test_values.csv") %>%
  select(-building_id)

test_factors <-
  map_dfc(
    1:38,
    function(col){
      if(col > 7 | col == 4){
        out <- list(factor(test_data[, col], levels = levels(train_factors[, col])))
        names(out) <- names(test_data[col])
        return(out)
      } else{
        out <- list(test_data[, col])
        names(out) <- names(test_data[col])
        return(out)
      }
    }
  ) %>%
  as.data.frame()


# Define F1 score for multiclass -----------------------------------------------

f1_multiclass = R6::R6Class(
  "f1_multiclass",
  inherit = mlr3::MeasureClassif,
  public = list(
    initialize = function() {
      super$initialize(
        # custom id for the measure
        id = "classif.f1_multiclass",

        # additional packages required to calculate this measure
        packages = c('caret'),

        # properties, see below
        properties = character(),

        # required predict type of the learner
        predict_type = "response",

        # feasible range of values
        range = c(0, 1),

        # minimize during tuning?
        minimize = FALSE
      )
    }
  ),

  private = list(
    # custom scoring function operating on the prediction object
    .score = function(prediction, ...) {

      f1_multiclass <- function(truth, response){

        # Actual: rows; Predictions: columns
        cm <- confusionMatrix(response, reference = truth, dnn = c("Reference", "Prediction"))
        cm <- cm$table
        rowsums = apply(cm, 1, sum) # number of instances per class
        colsums = apply(cm, 2, sum) # number of predictions per class
        n = sum(cm) # number of instances
        nc = nrow(cm) # number of classes

        # one-vs-all confusion matrix for each class (3 matrices in this case)
        # 3 binary classification tasks where one class is considered the positive class
        # while the combination of all the other classes make up the negative class
        oneVsAll = lapply(
          1 : nc,
          function(i){
            v = c(cm[i,i],
                  rowsums[i] - cm[i,i],
                  colsums[i] - cm[i,i],
                  n-rowsums[i] - colsums[i] + cm[i,i]);
            return(matrix(v, nrow = 2, byrow = T))
          }
        )
        # Summing up the values of these 3 matrices results in one confusion matrix and
        # allows us to compute weighted metrics
        s = matrix(0, nrow = 2, ncol = 2)
        for(i in 1 : nc){
          s = s + oneVsAll[[i]]
        }

        # Because the sum of the one-vs-all matrices is a symmetric matrix, the micro-averaged
        # precision, recall, and F-1 wil be the same
        micro_f1 = (diag(s) / apply(s,1, sum))[1]

        return(micro_f1)

      }

      f1_multiclass(truth = prediction$truth, response = prediction$response)

    }
  )
)

mlr3::mlr_measures$add("classif.f1_multiclass", f1_multiclass)


# Define the pipeline ----------------------------------------------------------

po = R6::R6Class(
  "PipeOpOrderGeoLevel",
  inherit = mlr3pipelines::PipeOpTaskPreproc,
  public = list(
    initialize = function(id = "orderGeoLevel") {
      super$initialize(
        id = id,
        packages = c('dplyr', 'magrittr')
      )
    }
  ),

  private = list(
    .train_dt = function(dt, levels, target) {

      library(magrittr)
      library(dplyr)
      library(tidyr)
      library(stringr)

      # Add target variable
      dt$target = target

      # Determine new geo levels

      new.geo.level.1 <-
        dt %>%
        select(geo_level_1_id, target) %>%
        group_by(geo_level_1_id, target) %>%
        mutate(number_per_damage_and_geo_level = n()) %>%
        ungroup() %>%
        distinct_all() %>%
        group_by(geo_level_1_id) %>%
        mutate(number_per_geo_level = sum(number_per_damage_and_geo_level)) %>%
        ungroup() %>%
        distinct_all() %>%
        mutate(percentage = number_per_damage_and_geo_level / number_per_geo_level) %>%
        group_by(geo_level_1_id) %>%
        mutate(expected_damage_grade = sum(as.numeric(target) * percentage)) %>%
        ungroup() %>%
        select(geo_level_1_id, expected_damage_grade) %>%
        distinct_all() %>%
        mutate(reordered_geo_level_1_id = dense_rank(desc(expected_damage_grade))) %>%
        select(geo_level_1_id, reordered_geo_level_1_id) %>%
        as.data.table()

      new.geo.level.2 <-
        dt %>%
        select(geo_level_2_id, target) %>%
        group_by(geo_level_2_id, target) %>%
        mutate(number_per_damage_and_geo_level = n()) %>%
        ungroup() %>%
        distinct_all() %>%
        group_by(geo_level_2_id) %>%
        mutate(number_per_geo_level = sum(number_per_damage_and_geo_level)) %>%
        ungroup() %>%
        distinct_all() %>%
        mutate(percentage = number_per_damage_and_geo_level / number_per_geo_level) %>%
        group_by(geo_level_2_id) %>%
        mutate(expected_damage_grade = sum(as.numeric(target) * percentage)) %>%
        ungroup() %>%
        select(geo_level_2_id, expected_damage_grade) %>%
        distinct_all() %>%
        mutate(reordered_geo_level_2_id = dense_rank(desc(expected_damage_grade))) %>%
        select(geo_level_2_id, reordered_geo_level_2_id) %>%
        as.data.table()

      new.geo.level.3 <-
        dt %>%
        select(geo_level_3_id, target) %>%
        group_by(geo_level_3_id, target) %>%
        mutate(number_per_damage_and_geo_level = n()) %>%
        ungroup() %>%
        distinct_all() %>%
        group_by(geo_level_3_id) %>%
        mutate(number_per_geo_level = sum(number_per_damage_and_geo_level)) %>%
        ungroup() %>%
        distinct_all() %>%
        mutate(percentage = number_per_damage_and_geo_level / number_per_geo_level) %>%
        group_by(geo_level_3_id) %>%
        mutate(expected_damage_grade = sum(as.numeric(target) * percentage)) %>%
        ungroup() %>%
        select(geo_level_3_id, expected_damage_grade) %>%
        distinct_all() %>%
        mutate(reordered_geo_level_3_id = dense_rank(desc(expected_damage_grade))) %>%
        select(geo_level_3_id, reordered_geo_level_3_id) %>%
        as.data.table()

      dummy.high.dam.3.2 <-
        dt %>%
        select(geo_level_2_id, target) %>%
        group_by(geo_level_2_id, target) %>%
        mutate(number_per_damage_and_geo_level = n()) %>%
        ungroup() %>%
        distinct_all() %>%
        group_by(geo_level_2_id) %>%
        mutate(number_per_geo_level = sum(number_per_damage_and_geo_level)) %>%
        ungroup() %>%
        distinct_all() %>%
        mutate(percentage = number_per_damage_and_geo_level / number_per_geo_level) %>%
        filter(target == 3) %>%
        mutate(indicator_dam_grade_3 = case_when(
          percentage >= 0.55 ~ 2,
          between(percentage, 0.35, 0.55) ~ 1,
          TRUE ~ 0
        )) %>%
        select(geo_level_2_id, indicator_dam_grade_3)

      dummy.low.dam.1.3 <-
        dt %>%
        select(geo_level_3_id, target) %>%
        group_by(geo_level_3_id, target) %>%
        mutate(number_per_damage_and_geo_level = n()) %>%
        ungroup() %>%
        distinct_all() %>%
        group_by(geo_level_3_id) %>%
        mutate(number_per_geo_level = sum(number_per_damage_and_geo_level)) %>%
        ungroup() %>%
        distinct_all() %>%
        mutate(percentage = number_per_damage_and_geo_level / number_per_geo_level) %>%
        filter(target == 1) %>%
        mutate(indicator_dam_grade_1 = case_when(
          percentage >= 0.6 ~ 2,
          between(percentage, 0.12, 0.6) ~ 1,
          TRUE ~ 0
        )) %>%
        select(geo_level_3_id, indicator_dam_grade_1)

      # Remove Target variable
      dt$target = NULL

      # Save values in state to make them available in .predict part
      self$state = list(
        new.geo.level.1 = new.geo.level.1,
        new.geo.level.2 = new.geo.level.2,
        new.geo.level.3 = new.geo.level.3,
        dummy.high.dam.3.2 = dummy.high.dam.3.2,
        dummy.low.dam.1.3 = dummy.low.dam.1.3
      )

      # Add order variable to retain ordering after merge (!)
      dt$order.id <- 1:nrow(dt)

      # Add new variables to data.table
      merge(dt, self$state$new.geo.level.1, by = "geo_level_1_id") %>%
        merge(self$state$new.geo.level.2, by = "geo_level_2_id") %>%
        merge(self$state$new.geo.level.3, by = "geo_level_3_id") %>%
        merge(self$state$dummy.high.dam.3.2, by = "geo_level_2_id", all.x = TRUE) %>%
        merge(self$state$dummy.low.dam.1.3, by = "geo_level_3_id", all.x = TRUE) %>%
        mutate(
          indicator_dam_grade_3 = if_else(
            is.na(indicator_dam_grade_3), 0, indicator_dam_grade_3
          ) %>% as.factor(),
          indicator_dam_grade_1 = if_else(
            is.na(indicator_dam_grade_1), 0, indicator_dam_grade_1
          ) %>% as.factor(),
          mud_mortar_stone_adobe_mud = case_when(
            has_superstructure_mud_mortar_stone == 1 ~ 1,
            has_superstructure_adobe_mud == 1 ~ 1,
            TRUE ~ 0
          ) %>% as.factor(),
          bamboo_timber = case_when(
            has_superstructure_bamboo == 1 ~ 1,
            has_superstructure_timber == 1 ~ 1,
            TRUE ~ 0
          ) %>% as.factor()
        ) %>%
        arrange(order.id) %>%
        select(-order.id) %>%
        as.data.table()

    },

    .predict_dt = function(dt, levels) {

      library(magrittr)
      library(dplyr)
      library(tidyr)
      library(stringr)

      # Add order variable to retain ordering after merge (!)
      dt$order.id <- 1:nrow(dt)

      # Merge variables from state to data.table
      merge(dt, self$state$new.geo.level.1, by = "geo_level_1_id", all.x = TRUE) %>%
        merge(self$state$new.geo.level.2, by = "geo_level_2_id", all.x = TRUE) %>%
        merge(self$state$new.geo.level.3, by = "geo_level_3_id", all.x = TRUE) %>%
        merge(self$state$dummy.high.dam.3.2, by = "geo_level_2_id", all.x = TRUE) %>%
        merge(self$state$dummy.low.dam.1.3, by = "geo_level_3_id", all.x = TRUE) %>%
        mutate(
          indicator_dam_grade_3 = if_else(
            is.na(indicator_dam_grade_3), 0, indicator_dam_grade_3
          ) %>% as.factor(),
          indicator_dam_grade_1 = if_else(
            is.na(indicator_dam_grade_1), 0, indicator_dam_grade_1
          ) %>% as.factor(),
          reordered_geo_level_1_id = if_else(
            is.na(reordered_geo_level_1_id),
            9999L,
            reordered_geo_level_1_id
          ),
          reordered_geo_level_2_id = if_else(
            is.na(reordered_geo_level_2_id),
            9999L,
            reordered_geo_level_2_id
          ),
          reordered_geo_level_3_id = if_else(
            is.na(reordered_geo_level_3_id),
            9999L,
            reordered_geo_level_3_id
          )
        ) %>%
        mutate(
          mud_mortar_stone_adobe_mud = case_when(
            has_superstructure_mud_mortar_stone == 1 ~ 1,
            has_superstructure_adobe_mud == 1 ~ 1,
            TRUE ~ 0
          ) %>% as.factor(),
          bamboo_timber = case_when(
            has_superstructure_bamboo == 1 ~ 1,
            has_superstructure_timber == 1 ~ 1,
            TRUE ~ 0
          ) %>% as.factor()
        ) %>%
        arrange(order.id) %>%
        dplyr::select(-order.id) %>%
        as.data.table()

    }
  )
)


# Define the task --------------------------------------------------------------

task_pipe <-
  as_task_classif(
    train_factors,
    target = "damage_grade",
    id = "pipe"
  )

# Set stratum variable
task_pipe$col_roles$stratum <- "damage_grade"


# Random Forest ----------------------------------------------------------------

## With feature pipeline -------------------------------------------------------

# Define the graph (= pipeline)
graph_rf <-
  po$new() %>>%
  lrn(
    "classif.ranger",
    predict_type = "response",
    importance = "impurity",
    num.threads = 16,
    mtry.ratio = 0.1071,
    num.trees = 1000
  )

# Transform pipeline to learner
randomforest <- as_learner(graph_rf)

# Train the learner
set.seed(123)
randomforest$train(task_pipe)
randomforest$model

with_new_var <-
  round(
    randomforest$model$classif.ranger$model$confusion.matrix / nrow(train_factors),
    3
  )

preds <- randomforest$predict_newdata(test_factors)

# Create .csv file for upload
preds_out <-
  read.csv("data/test_values.csv") %>%
  select(building_id) %>%
  mutate(damage_grade = preds$response) %>%
  as_tibble()

write.csv(preds_out, "data/predictions_pipe_rf.csv", row.names = FALSE)


## Tuning ----------------------------------------------------------------------

# Use Multisession for parallelization
plan(multisession)

# Set up pipeline for tuning including the tuning grid
graph_tuning_rf <-
  po$new() %>>%
  lrn(
    "classif.ranger",
    predict_type = "response",
    mtry.ratio = to_tune(0.07, 0.2)#,
    # sample.fraction = to_tune(0.5, 1),
    # max.depth = to_tune(c(25, 35, 45, 55, NULL))
  )

# Transform pipeline to learner
learner_tuning_rf <- as_learner(graph_tuning_rf)


# Setup auto_tuner
at_rf = auto_tuner(
  learner = learner_tuning_rf,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.f1_multiclass"),
  tuner = mlr3tuning::tnr("grid_search", resolution = 15)
)

set.seed(123)
# Run tuning !! This might take some time !!
at_rf$train(task_pipe)

at_rf$tuning_instance$archive$data %>%
  .[, 1:4] %>%
  arrange(desc(classif.f1_multiclass)) %>%
  View()

# Prediction
preds_after_tuning <- at_rf$predict_newdata(test_factors)

preds_tuning_out <-
  read.csv("data/test_values.csv") %>%
  select(building_id) %>%
  mutate(damage_grade = preds_after_tuning$response) %>%
  as_tibble()

write.csv(preds_tuning_out, "data/preds_after_tuning_rf.csv", row.names = FALSE)


# Only using the original variables --------------------------------------------

# Define the graph (= pipeline)
graph_rf <-
  lrn(
    "classif.ranger",
    predict_type = "response",
    num.threads = 16,
    mtry.ratio = 0.1071,
    num.trees = 1000
  )

# Transform pipeline to learner
randomforest <- as_learner(graph_rf)

# Train the learner
set.seed(123)
randomforest$train(task_pipe)
randomforest$model

preds <- randomforest$predict_newdata(test_factors)

# Create .csv file for upload
preds_out <-
  read.csv("data/test_values.csv") %>%
  select(building_id) %>%
  mutate(damage_grade = preds$response) %>%
  as_tibble()

write.csv(preds_out, "data/predictions_rf.csv", row.names = FALSE)


# Featureless model ------------------------------------------------------------

graph_featureless <-
  lrn(
    "classif.featureless",
    method = "mode"
  )

featureless <- as_learner(graph_featureless)
featureless$train(task_pipe)

preds <- featureless$predict_newdata(test_factors)

# Create .csv file for upload
preds_out <-
  read.csv("data/test_values.csv") %>%
  select(building_id) %>%
  mutate(damage_grade = preds$response) %>%
  as_tibble()


write.csv(preds_out, "data/predictions_featureless_mode.csv", row.names = FALSE)


# Decision Tree ----------------------------------------------------------------

## Tuning ----------------------------------------------------------------------

# Use Multisession for parallelization
plan(multisession)

graph_decision_tree <-
  po$new() %>>%
  lrn(
    "classif.rpart",
    maxdepth = to_tune(5, 30),
    cp = to_tune(0.0001, 0.03),
    minsplit = to_tune(5, 30)
  )


# Transform pipeline to learner
decision_tree_tuning <- as_learner(graph_decision_tree)


# Setup auto_tuner
at_decision_tree = auto_tuner(
  learner = decision_tree_tuning,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.f1_multiclass"),
  tuner = mlr3tuning::tnr("grid_search", resolution = 7)
)

set.seed(123)
# Run tuning !! This might take some time !!
at_decision_tree$train(task_pipe)


preds <- at_decision_tree$predict_newdata(test_factors)

# Create .csv file for upload
preds_out <-
  read.csv("data/test_values.csv") %>%
  select(building_id) %>%
  mutate(damage_grade = preds$response) %>%
  as_tibble()


write.csv(preds_out, "data/predictions_tuned_decision_tree.csv", row.names = FALSE)


## Only original features ------------------------------------------------------

graph_decision_tree <-
  lrn(
    "classif.rpart",
    maxdepth = 22,
    cp = 0.0001,
    minsplit = 30
  )


# Transform pipeline to learner
learner_decision_tree <- as_learner(graph_decision_tree)

learner_decision_tree$train(task_pipe)

preds <- learner_decision_tree$predict_newdata(test_factors)

# Create .csv file for upload
preds_out <-
  read.csv("data/test_values.csv") %>%
  select(building_id) %>%
  mutate(damage_grade = preds$response) %>%
  as_tibble()


write.csv(preds_out, "data/predictions_decision_tree.csv", row.names = FALSE)



# XGBoost ----------------------------------------------------------------------


## Tuning ----------------------------------------------------------------------

graph_xgboost_tuning <-
  po$new() %>>%
  pipeline_robustify(
    task = task_pipe,
    learner = lrn(
      "classif.xgboost"
    )
  ) %>>%
  lrn(
    "classif.xgboost",
    predict_type = "response",
    nrounds = 1000,
    early_stopping_rounds = 20,
    early_stopping_set = "train",
    eta = to_tune(0.01, 0.3),
    colsample_bytree = to_tune(0.7, 1),
    gamma = to_tune(0.01, 0.3),
    alpha = to_tune(0.01, 0.3),
    lambda = to_tune(0.01, 0.3),
    max_depth = to_tune(2, 7),
    objective = "multi:softmax"
  )

# Transform pipeline to learner
xgboost_tuner <- as_learner(graph_xgboost_tuning)

# Setup auto_tuner
at_xgboost = auto_tuner(
  learner = xgboost_tuner,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.f1_multiclass"),
  tuner = mlr3tuning::tnr("random_search"),
  terminator = trm("run_time", secs = 8 * 60 *60)
)

plan(multisession)

# Train the learner
set.seed(123)
at_xgboost$train(task_pipe)

preds <- at_xgboost$predict_newdata(test_factors)

# Create .csv file for upload
preds_out <-
  read.csv("data/test_values.csv") %>%
  select(building_id) %>%
  mutate(damage_grade = preds$response) %>%
  as_tibble()


write.csv(preds_out, "data/predictions_tuned_xgb.csv", row.names = FALSE)


## Only original features ------------------------------------------------------

graph_xgboost <-
  pipeline_robustify(
    task = task_pipe,
    learner = lrn(
      "classif.xgboost"
    )
  ) %>>%
  lrn(
    "classif.xgboost",
    predict_type = "response",
    nrounds = 1000,
    early_stopping_rounds = 20,
    early_stopping_set = "train",
    eta = 0.1116,
    colsample_bytree = 0.7064,
    gamma = 0.1529,
    alpha = 0.2719,
    lambda = 0.0470,
    max_depth = 7,
    objective = "multi:softmax",
    nthread = 16
  )

# Transform pipeline to learner
learner_xgboost <- as_learner(graph_xgboost)

set.seed(123)
learner_xgboost$train(task_pipe)

preds <- learner_xgboost$predict_newdata(test_factors)

# Create .csv file for upload
preds_out <-
  read.csv("data/test_values.csv") %>%
  select(building_id) %>%
  mutate(damage_grade = preds$response) %>%
  as_tibble()


write.csv(preds_out, "data/predictions_xgboost.csv", row.names = FALSE)
