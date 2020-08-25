library(workflows)
library(parsnip)
library(recipes)
library(yardstick)
library(glmnet)
library(tidyverse)
library(tidyquant)
library(timetk)

bikes <- read_csv('day.csv')

head(bikes)

bikes_tbl <- bikes %>% select(dteday, cnt) %>% rename(date = dteday, value = cnt)

head(bikes_tbl)

bikes_tbl %>% 
    ggplot(aes(x=date, y=value)) + 
    geom_rect(xmin = as.numeric(ymd('2012-07-01')),
              xmax = as.numeric(ymd('2013-01-01')),
              ymin = 0, ymax = 10000,
              fill = palette_light()[[4]], alpha = 0.01) +
    annotate("text", x = ymd('2011-10-01'), y = 7800,
            color = palette_light()[[1]], label = 'Train Region') +
    annotate("text", x = ymd('2012-10-01'), y = 1550,
            color = palette_light()[[1]], label = 'Train Region') +
    geom_point(alpha = 0.5, color = palette_light()[[1]]) +
    labs(title = 'Bike Sharing Dataset: Daily Scale', x="") +
    theme_tq()

train_tbl <- bikes_tbl %>% filter(date < ymd("2012-07-01"))
test_tbl <- bikes_tbl %>% filter(date >= ymd("2012-07-01")) 

recipe_spec_timeseries <- recipe(value ~., data =train_tbl) %>%
    step_timeseries_signature(date)

bake(prep(recipe_spec_timeseries), new_data = train_tbl)

recipe_spec_final <- recipe_spec_timeseries %>%
    step_rm(date) %>%
    step_rm(contains("iso"), 
            contains("second"), contains("minute"), contains("hour"),
            contains("am.pm"), contains("xts")) %>%
    step_normalize(contains("index.num"), date_year) %>%
    step_interact(~ date_month.lbl * date_day) %>%
    step_interact(~ date_month.lbl * date_mweek) %>%
    step_interact(~ date_month.lbl * date_wday.lbl * date_yday) %>%
    step_dummy(contains("lbl"), one_hot = TRUE) 

bake(prep(recipe_spec_final), new_data = train_tbl)

model_spec_glmnet <- linear_reg(mode = "regression", penalty = 10, mixture = 0.7) %>%
    set_engine("glmnet")

workflow_glmnet <- workflow() %>%
    add_recipe(recipe_spec_final) %>%
    add_model(model_spec_glmnet)

workflow_glmnet

workflow_trained <- workflow_glmnet %>% fit(data = train_tbl)

prediction_tbl <- workflow_trained %>% 
    predict(test_tbl) %>%
    bind_cols(test_tbl) 

head(prediction_tbl)

ggplot(aes(x = date), data = bikes_tbl) +
    geom_rect(xmin = as.numeric(ymd("2012-07-01")),
              xmax = as.numeric(ymd("2013-01-01")),
              ymin = 0, ymax = 10000,
              fill = palette_light()[[4]], alpha = 0.01) +
    annotate("text", x = ymd("2011-10-01"), y = 7800,
             color = palette_light()[[1]], label = "Train Region") +
    annotate("text", x = ymd("2012-10-01"), y = 1550,
             color = palette_light()[[1]], label = "Test Region") + 
    geom_point(aes(x = date, y = value),  
               alpha = 0.5, color = palette_light()[[1]]) +
    geom_point(aes(x = date, y = .pred), data = prediction_tbl, 
               alpha = 0.5, color = palette_light()[[2]]) +
    theme_tq() +
    labs(title = "GLM: Out-Of-Sample Forecast")

prediction_tbl %>% metrics(value, .pred)

prediction_tbl %>%
    ggplot(aes(x = date, y = value - .pred)) +
    geom_hline(yintercept = 0, color = "black") +
    geom_point(color = palette_light()[[1]], alpha = 0.5) +
    geom_smooth(span = 0.05, color = "red") +
    geom_smooth(span = 1.00, se = FALSE) +
    theme_tq() +
    labs(title = "GLM Model Residuals, Out-of-Sample", x = "") +
    scale_y_continuous(limits = c(-5000, 5000))

idx <- bikes_tbl %>% tk_index()
bikes_summary <- idx %>% tk_get_timeseries_summary()

bikes_summary[1:6]

bikes_summary[7:12]

idx_future <- idx %>% tk_make_future_timeseries(n_future = 180)

future_tbl <- tibble(date = idx_future) 

head(future_tbl)

future_predictions_tbl <- workflow_glmnet %>% 
    fit(data = bikes_tbl) %>%
    predict(future_tbl) %>%
    bind_cols(future_tbl)

bikes_tbl %>%
    ggplot(aes(x = date, y = value)) +
    geom_rect(xmin = as.numeric(ymd("2012-07-01")),
              xmax = as.numeric(ymd("2013-01-01")),
              ymin = 0, ymax = 10000,
              fill = palette_light()[[4]], alpha = 0.01) +
    geom_rect(xmin = as.numeric(ymd("2013-01-01")),
              xmax = as.numeric(ymd("2013-07-01")),
              ymin = 0, ymax = 10000,
              fill = palette_light()[[3]], alpha = 0.01) +
    annotate("text", x = ymd("2011-10-01"), y = 7800,
             color = palette_light()[[1]], label = "Train Region") +
    annotate("text", x = ymd("2012-10-01"), y = 1550,
             color = palette_light()[[1]], label = "Test Region") +
    annotate("text", x = ymd("2013-4-01"), y = 1550,
             color = palette_light()[[1]], label = "Forecast Region") +
    geom_point(alpha = 0.5, color = palette_light()[[1]]) +
    geom_point(aes(x = date, y = .pred), data = future_predictions_tbl,
               alpha = 0.5, color = palette_light()[[2]]) +
    geom_smooth(aes(x = date, y = .pred), data = future_predictions_tbl,
                method = 'loess') + 
    labs(title = "Bikes Sharing Dataset: 6-Month Forecast", x = "") +
    theme_tq()

test_resid_sd <- prediction_tbl %>%
    summarize(stdev = sd(value - .pred))

future_predictions_tbl <- future_predictions_tbl %>%
    mutate(
        lo.95 = .pred - 1.96 * test_resid_sd$stdev,
        lo.80 = .pred - 1.28 * test_resid_sd$stdev,
        hi.80 = .pred + 1.28 * test_resid_sd$stdev,
        hi.95 = .pred + 1.96 * test_resid_sd$stdev
    )

bikes_tbl %>%
    ggplot(aes(x = date, y = value)) +
    geom_point(alpha = 0.5, color = palette_light()[[1]]) +
    geom_ribbon(aes(y = .pred, ymin = lo.95, ymax = hi.95), 
                data = future_predictions_tbl, 
                fill = "#D5DBFF", color = NA, size = 0) +
    geom_ribbon(aes(y = .pred, ymin = lo.80, ymax = hi.80, fill = key), 
                data = future_predictions_tbl,
                fill = "#596DD5", color = NA, size = 0, alpha = 0.8) +
    geom_point(aes(x = date, y = .pred), data = future_predictions_tbl,
               alpha = 0.5, color = palette_light()[[2]]) +
    geom_smooth(aes(x = date, y = .pred), data = future_predictions_tbl,
                method = 'loess', color = "white") + 
    labs(title = "Bikes Sharing Dataset: 6-Month Forecast with Prediction Intervals", x = "") +
    theme_tq()


