Features selection using LASSo
================
Joshua Edefo
2024-08-12

Install and load the glmnet package if not already installed

``` r
library(glmnet)
```

    ## Warning: package 'glmnet' was built under R version 4.3.2

``` r
library(usethis)
```

    ## Warning: package 'usethis' was built under R version 4.3.2

Generate a synthetic dataset for demonstration, and Split data

``` r
set.seed(123)
n <- 100  # Number of observations
p <- 10   # Number of predictors
X <- matrix(rnorm(n * p), n, p)  # Predictor matrix
beta <- c(3, 0, 0, -2, 0, 0, 1.5, 0, 0, 0)  # True coefficients
y <- X %*% beta + rnorm(n)  # Response vector with noise

# Split data into training and testing sets
train_index <- sample(seq_len(n), size = 0.7 * n)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]
```

Fit LASSO model using cross-validation, plot, extract the best lamda
value, and make predictions

``` r
# Fit LASSO model using cross-validation
lasso_cv <- cv.glmnet(X_train, y_train, alpha = 1, family = "gaussian")

# Plot cross-validation results
plot(lasso_cv)
```

![](Features-selection-using-LASSO_2_files/figure-gfm/c-1.png)<!-- -->

``` r
# Extract the best lambda value
best_lambda <- lasso_cv$lambda.min
best_lambda
```

    ## [1] 0.05510448

``` r
cat("Best Lambda: ", best_lambda, "\n")
```

    ## Best Lambda:  0.05510448

``` r
# Coefficients of the best model
lasso_coef <- coef(lasso_cv, s = "lambda.min")
lasso_coef
```

    ## 11 x 1 sparse Matrix of class "dgCMatrix"
    ##                       s1
    ## (Intercept)  0.191366787
    ## V1           3.031315529
    ## V2           0.116577336
    ## V3          -0.009961886
    ## V4          -1.883218333
    ## V5           .          
    ## V6           .          
    ## V7           1.580358371
    ## V8           0.099405959
    ## V9          -0.151104438
    ## V10          0.145114301

``` r
print(lasso_coef)
```

    ## 11 x 1 sparse Matrix of class "dgCMatrix"
    ##                       s1
    ## (Intercept)  0.191366787
    ## V1           3.031315529
    ## V2           0.116577336
    ## V3          -0.009961886
    ## V4          -1.883218333
    ## V5           .          
    ## V6           .          
    ## V7           1.580358371
    ## V8           0.099405959
    ## V9          -0.151104438
    ## V10          0.145114301

``` r
# V1 to V10: These are the coefficients for each feature in the model.
# Represents zero coefficients for the features V5 and V6. In Lasso regression, some coefficients are shrunk to #exactly zero, which helps in feature selection by effectively excluding these variables from the model.

# Make predictions on the test set
predictions <- predict(lasso_cv, s = "lambda.min", newx = X_test)

# Convert the sparse matrix to a regular matrix for easier handling
lasso_coef <- as.matrix(lasso_coef)
lasso_coef
```

    ##                       s1
    ## (Intercept)  0.191366787
    ## V1           3.031315529
    ## V2           0.116577336
    ## V3          -0.009961886
    ## V4          -1.883218333
    ## V5           0.000000000
    ## V6           0.000000000
    ## V7           1.580358371
    ## V8           0.099405959
    ## V9          -0.151104438
    ## V10          0.145114301

``` r
# Identify variables with zero coefficients
zero_coef_vars <- rownames(lasso_coef)[lasso_coef == 0]
zero_coef_vars
```

    ## [1] "V5" "V6"

``` r
#Remove intercept from zero coefficient variables
zero_coef_vars <- zero_coef_vars[zero_coef_vars != "(Intercept)"]
zero_coef_vars
```

    ## [1] "V5" "V6"

``` r
# Remove zero coefficient variables from the dataset
X_reduced <- X[, !(colnames(X) %in% zero_coef_vars)]
```

Evaluate model performance

``` r
mse <- mean((y_test - predictions)^2)
cat("Mean Squared Error: ", mse, "\n")
```

    ## Mean Squared Error:  1.435743

session information

``` r
sessionInfo()
```

    ## R version 4.3.1 (2023-06-16 ucrt)
    ## Platform: x86_64-w64-mingw32/x64 (64-bit)
    ## Running under: Windows 11 x64 (build 22631)
    ## 
    ## Matrix products: default
    ## 
    ## 
    ## locale:
    ## [1] LC_COLLATE=English_United Kingdom.utf8 
    ## [2] LC_CTYPE=English_United Kingdom.utf8   
    ## [3] LC_MONETARY=English_United Kingdom.utf8
    ## [4] LC_NUMERIC=C                           
    ## [5] LC_TIME=English_United Kingdom.utf8    
    ## 
    ## time zone: Europe/London
    ## tzcode source: internal
    ## 
    ## attached base packages:
    ## [1] stats     graphics  grDevices utils     datasets  methods   base     
    ## 
    ## other attached packages:
    ## [1] usethis_2.2.2  glmnet_4.1-8   Matrix_1.6-1.1
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] vctrs_0.6.3       cli_3.6.1         knitr_1.44        rlang_1.1.1      
    ##  [5] xfun_0.40         purrr_1.0.2       glue_1.6.2        htmltools_0.5.6  
    ##  [9] rmarkdown_2.25    grid_4.3.1        evaluate_0.21     fastmap_1.1.1    
    ## [13] yaml_2.3.7        foreach_1.5.2     lifecycle_1.0.3   compiler_4.3.1   
    ## [17] codetools_0.2-19  fs_1.6.3          Rcpp_1.0.11       rstudioapi_0.15.0
    ## [21] lattice_0.21-8    digest_0.6.33     splines_4.3.1     shape_1.4.6      
    ## [25] magrittr_2.0.3    tools_4.3.1       iterators_1.0.14  survival_3.7-0
