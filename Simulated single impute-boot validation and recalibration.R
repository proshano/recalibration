######## PART 1: calculate the optimism you need to correct for ##########


# Load required libraries
rm(list=ls(all=TRUE))  # Clear workspace
library(rms)
library(Hmisc)
library(ggplot2)
library(pROC)

# Set seed for reproducibility
set.seed(42)

# Simulate data with missing values and noise predictors
n <- 1700
d <- data.frame(
  x1 = rnorm(n),
  x2 = rnorm(n),
  x3 = rnorm(n),
  noise1 = rnorm(n),
  noise2 = rnorm(n),
  noise3 = rnorm(n),
  noise4 = rnorm(n),
  noise5 = rnorm(n)
)

# Generate outcome with very low event rate
intercept <- -4  # Adjusted to get very low event rate
d$y <- rbinom(n, 1, plogis(intercept + 0.5*d$x1 + 0.7*d$x2 - 0.3*d$x3))

# Check the number of events
num_events <- sum(d$y)
event_rate <- num_events / n
cat("Number of events:", num_events, "\n")
cat("Event rate:", event_rate, "\n")

# Introduce missing data
set.seed(123)
d$x1[sample(n, 20)] <- NA
d$x2[sample(n, 20)] <- NA
d$x3[sample(n, 20)] <- NA
d$noise1[sample(n, 20)] <- NA
d$noise2[sample(n, 20)] <- NA
d$noise3[sample(n, 20)] <- NA
d$noise4[sample(n, 20)] <- NA
d$noise5[sample(n, 20)] <- NA

# Perform single imputation
imp <- aregImpute(~ y + x1 + x2 + x3 + noise1 + noise2 + noise3 + noise4 + noise5, data = d, n.impute = 1)
imputed_values <- impute.transcan(imp, imputation = 1, data = d, list.out = TRUE)
d_imputed <- d  # Start with the original dataset
for(var in names(imputed_values)) {
  d_imputed[[var]] <- imputed_values[[var]]
}
# Fit the model
dd <- datadist(d_imputed)
options(datadist = "dd")
require(rms)

B <- 100  # Number of bootstrap samples for each validation
reps <- 100  # Number of repetitions for confidence interval estimation
n <- nrow(d_imputed)

# Initialize vectors to store results
dxy <- numeric(reps)
slopes <- numeric(reps)
intercepts <- numeric(reps)
dxy_optimism <- numeric(reps)
slope_optimism <- numeric(reps)
intercept_optimism <- numeric(reps)

# Original model fit
f <- lrm(y ~ x1 + x2 + x3 + noise1 + noise2 + noise3 + noise4 + noise5, data=d_imputed, x=TRUE, y=TRUE)
print(f)  # Show original model fit

# Repetitions for confidence interval estimation
for(i in 1:reps) {
  boot_sample <- d_imputed[sample(1:n, n, replace=TRUE), ]
  g <- update(f, data=boot_sample)
  v <- validate(g, B=B)
  
  dxy[i] <- v['Dxy', 'index.corrected']
  slopes[i] <- v['Slope', 'index.corrected']
  intercepts[i] <- v['Intercept', 'index.corrected']
  
  dxy_optimism[i] <- v['Dxy', 'optimism']
  slope_optimism[i] <- v['Slope', 'optimism']
  intercept_optimism[i] <- v['Intercept', 'optimism']
}

# Function to calculate and format results
format_results <- function(values, optimism_values, metric_type) {
  if (metric_type == "Dxy") {
    c_values <- 0.5 * (values + 1)  # Convert Dxy to C-index
    c_optimism <- 0.5 * optimism_values  # Convert Dxy optimism to C-index optimism
  } else {
    c_values <- values
    c_optimism <- optimism_values
  }
  
  list(
    median = median(c_values),
    mean = mean(c_values),
    ci = quantile(c_values, c(0.025, 0.975)),
    median_optimism = median(c_optimism),
    mean_optimism = mean(c_optimism),
    ci_optimism = quantile(c_optimism, c(0.025, 0.975))
  )
}

# Calculate results for each metric
c_index_results <- format_results(dxy, dxy_optimism, "Dxy")
slope_results <- format_results(slopes, slope_optimism, "Slope")
intercept_results <- format_results(intercepts, intercept_optimism, "Intercept")

# Function to print results with rounding to 2 decimal places
print_results <- function(name, results) {
  cat(name, ":\n")
  cat("  Optimism-corrected Median:", round(results$median, 2), "\n")
  cat("  Optimism-corrected Mean:", round(results$mean, 2), "\n")
  cat("  95% CI:", paste(round(results$ci, 2), collapse=" to "), "\n")
  cat("  Optimism Median:", round(results$median_optimism, 2), "\n")
  cat("  Optimism Mean:", round(results$mean_optimism, 2), "\n")
  cat("  Optimism 95% CI:", paste(round(results$ci_optimism, 2), collapse=" to "), "\n\n")
}

# Print results
print_results("C-index", c_index_results)
print_results("Calibration Slope", slope_results)
print_results("Calibration Intercept", intercept_results)

# Generate calibration curve
cal <- calibrate(f, method="boot", B=200)  # Using 200 bootstrap samples

# Plot the calibration curve
plot(cal, xlim=c(0,1), ylim=c(0,1),
     subtitles = FALSE,
     legend=TRUE,
     xlab="Predicted Probability",
     ylab="Observed Probability",
     main="Calibration Curve")






######### PART 2: Demonstrate the effect of recalibration ################

library(pROC)
library(ggplot2)


# bring in the optimism-corrected slope and intercept calculated in PART 1.
corrected_slope <-  mean(slopes)
corrected_intercept <- mean(intercepts)



custom_calibrate <- function(fit, corrected_slope, corrected_intercept, B = 200) {
  # Combine x and y into a single data frame
  data <- data.frame(y = fit$y, fit$x)
  n <- nrow(data)
  
  # Step 1: Fit the model on the original data and record C-statistic and predicted probabilities
  original_lp <- fit$linear.predictors
  predicted_original <- plogis(original_lp)
  original_c_stat <- roc(fit$y, predicted_original)$auc
  
  # Initialize storage for bootstrap results
  optimism_diffs <- matrix(NA, nrow = B, ncol = length(predicted_original))
  optimism_diffs_recalibrated <- matrix(NA, nrow = B, ncol = length(predicted_original))
  recalibrated_pred_probs <- vector("list", B)
  recalibrated_c_stats <- numeric(B)
  
  # Bootstrap loop
  for (i in 1:B) {
    # Step 2: Draw a bootstrap sample
    boot_data <- data[sample(1:n, n, replace = TRUE), ]
    
    # Step 3: Fit the model on the bootstrap sample
    boot_fit <- lrm(fit$terms, x = TRUE, y = TRUE, data = boot_data)
    
    # Step 4: Predict on the bootstrap sample and original sample using the model fit on the bootstrap sample
    predicted_boot <- plogis(boot_fit$linear.predictors)
    predicted_orig_on_boot <- predict(boot_fit, newdata = as.data.frame(fit$x), type = "fitted")
    
    # Fit calibration curves for both bootstrap and original samples
    boot_cal_fit <- lowess(predicted_boot, boot_data$y, iter = 0)
    orig_cal_fit_on_boot <- lowess(predicted_orig_on_boot, fit$y, iter = 0)
    
    # Use approxfun to create functions for the calibration curves
    boot_cal_fun <- approxfun(boot_cal_fit$x, boot_cal_fit$y)
    orig_cal_fun_on_boot <- approxfun(orig_cal_fit_on_boot$x, orig_cal_fit_on_boot$y)
    
    # Calculate the optimism as the difference between the two calibration curves over the range of predictions
    srange <- sort(predicted_original)
    optimism_diffs[i, ] <- boot_cal_fun(srange) - orig_cal_fun_on_boot(srange)
    
    # Step 5: Recalibrate the model using the slope and intercept correction factors
    recalibrated_lp <- corrected_slope * predict(boot_fit, newdata = as.data.frame(fit$x), type = "lp") + corrected_intercept
    recalibrated_pred <- plogis(recalibrated_lp)
    recalibrated_pred_probs[[i]] <- recalibrated_pred
    
    # Record the C-statistic after recalibration
    recalibrated_c_stats[i] <- roc(fit$y, recalibrated_pred)$auc
    
    # Fit calibration curves for recalibrated predictions on original sample
    recal_cal_fit_on_boot <- lowess(recalibrated_pred, fit$y, iter = 0)
    recal_cal_fun_on_boot <- approxfun(recal_cal_fit_on_boot$x, recal_cal_fit_on_boot$y)
    
    # Calculate the optimism for recalibrated model as the difference between the recalibrated calibration curve and original calibration curve
    optimism_diffs_recalibrated[i, ] <- boot_cal_fun(srange) - recal_cal_fun_on_boot(srange)
  }
  
  # Average optimism over all bootstrap samples
  avg_optimism_diff <- colMeans(optimism_diffs, na.rm = TRUE)
  avg_optimism_diff_recalibrated <- colMeans(optimism_diffs_recalibrated, na.rm = TRUE)
  
  # Calculate optimism-corrected predictions
  avg_recalibrated_pred <- rowMeans(do.call(cbind, recalibrated_pred_probs)) # on original data after recal
  
  # Calibration function
  cal.func <- function(pred_probs, y, smoother = "lowess") {
    smo <- if(is.function(smoother)) smoother(pred_probs, y) else lowess(pred_probs, y, iter = 0)
    return(approx(smo$x, smo$y, xout = sort(pred_probs))$y)
  }
  
  # Step 6: Create calibration plot
  apparent_cal <- cal.func(predicted_original, fit$y)
  optimism_corrected_cal <- apparent_cal - avg_optimism_diff
  optimism_corrected_recalibrated_cal <- (apparent_cal - avg_optimism_diff_recalibrated)
  
  # Create result data frame for plotting
  result <- data.frame(
    predy = sort(predicted_original),
    apparent = apparent_cal,
    optimism.corrected = optimism_corrected_cal,
    optimism.corrected.recalibrated = optimism_corrected_recalibrated_cal
  )
  
  # Plotting the calibration curves
  p <- ggplot(result, aes(x = predy)) +
    geom_abline(intercept = 0, slope = 1, linetype = "solid", color = "black") +
    geom_line(aes(y = apparent, color = "Apparent"), linetype = "dotted") +
    geom_line(aes(y = optimism.corrected, color = "Optimism-corrected"), linetype = "dashed") +
    geom_line(aes(y = optimism.corrected.recalibrated, color = "Optimism-corrected and recalibrated"), linetype = "solid") +
    scale_color_manual(values = c("Apparent" = "blue", 
                                  "Optimism-corrected" = "orange", 
                                  "Optimism-corrected and recalibrated" = "purple")) +
    labs(x = "Predicted Probability", 
         y = "Observed Probability",
         title = "Calibration Plot",
         color = "Calibration Type") +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "bottom",
      panel.grid.minor = element_line(color = "gray90"),
      panel.grid.major = element_line(color = "gray80")
    )
  
  return(list(plot = p, original_c_stat = original_c_stat, recalibrated_c_stats = recalibrated_c_stats))
}

# Usage example
cal_result <- custom_calibrate(f, corrected_slope = corrected_slope, corrected_intercept = corrected_intercept, B = 200)
print(cal_result$plot)

mean(cal_result$original_c_stat)
mean(cal_result$recalibrated_c_stats)

# Assuming you've already run the custom_calibrate function
# Let's see the final equation
# Apply recalibration to the original coefficients
recalibrated_coefficients <- coef(f)[-1] * corrected_slope
recalibrated_intercept <- coef(f)[1] * corrected_slope + corrected_intercept

# Print the recalibrated model
cat("Recalibrated Model Equation:\n")
cat("logit(p) =", recalibrated_intercept, "+", paste(recalibrated_coefficients, names(recalibrated_coefficients), sep = " * ", collapse = " + "), "\n")

#the end?

