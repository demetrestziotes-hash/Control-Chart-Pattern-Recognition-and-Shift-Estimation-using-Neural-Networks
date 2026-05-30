install.packages("neuralnet")
library(neuralnet)

set.seed(123)
n <- 8 # size of sliding window
samples <- 100

# Simulate Data

# Pattern: Normal
data.normal <- t(replicate(samples, rnorm(n, mean=0, sd=0.5)))
labels.norm <- data.frame(normal=rep(1, samples), trend=0, shift=0, cyclic=0)

# Pattern: Trend (up)
data.trend <- t(replicate(samples, rnorm(n, mean=0, sd=0.5) + seq(0, 3, length.out=n)))
labels.trend <- data.frame(normal=0, trend=rep(1, samples), shift=0, cyclic=0)

# Pattern: Shift (down)
data.shift <- t(replicate(samples, rnorm(n, mean=0, sd=0.5) + c(rep(0, n/2), rep(-3, n/2))))
labels.shift <- data.frame(normal=0, trend=0, shift=rep(1, samples), cyclic=0)

# Pattern: Cycle
data.cycle <- t(replicate(samples, rnorm(n, mean=0, sd=0.5) + 2*sin(seq(0, 2*pi, length.out=n))))
labels.cycle <- data.frame(normal=0, trend=0, shift=0, cyclic=rep(1, samples))

# Combine the different sets
data <- rbind(data.normal, data.trend, data.shift, data.cycle)
labels <- rbind(labels.norm, labels.trend, labels.shift, labels.cycle)
train.data <- data.frame(data, labels)
colnames(train.data)[1:8] <- c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8")

# Train BPPR
hidden.values <- seq(2, 20, by=3)
shift.values <- c(-1, -3)
cutoff.values <- c(0.85, 0.90, 0.95)
results <- data.frame(hidden.neurons=numeric(), shift=numeric(), cutoff=numeric(), 
                      ROT=numeric(), ATPRL=numeric(), ARLIDX=numeric())

pattern.names <- c("normal", "trend", "shift", "cycle")
target.pattern <- "shift"

# Now repeat the process many times with monte-carlo to estimate performance measures
for (h in hidden.values) {
  # Train network with h hidden layers
  m1 <- neuralnet(normal + trend + shift + cyclic ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8,
                  data = train.data, hidden = h, act.fct = "logistic",
                  linear.output = FALSE, stepmax = 1e6)
                  
  for (s in shift.values) {
    for (c in cutoff.values) {
      detected.patterns <- character(50)
      detection.times <- numeric(50)
      
      for (i in 1:50) {
        # Simulate data from a process
        process <- rnorm(400, mean=500, sd=0.5)
        # After obs. 85 the machine malfunctions and we have shift of 3 units
        process[231:400] <- process[231:400] + s
        
        alarm <- FALSE
        first.pattern <- NA
        first.time <- NA
        
        for (t in n:400) {
          # Keep the 8 last values
          curr <- process[(t - n + 1):t]
          curr.st <- curr - 500
          # Convert it into data frame for the model to read
          input <- data.frame(matrix(curr.st, nrow=1))
          colnames(input) <- c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8")
          
          # Prediction from network
          pred <- compute(m1, input)$net.result
          p <- as.numeric(pred) # prob. in [0,1] for trend
          
          # Find the neuron with max prob.
          max.prob <- max(p)
          ind <- which.max(p)
          
          if (max.prob >= c && ind != 1) {
            alarm <- TRUE
            first.pattern <- pattern.names[ind]
            first.time <- t
            break
          }
        }
        
        if (alarm) {
          detected.patterns[i] <- first.pattern
          detection.times[i] <- first.time
        } else {
          detected.patterns[i] <- "none"
          detection.times[i] <- NA
        }
      }
      
      # Rate of Target (ROT)
      ROT <- sum(detected.patterns == target.pattern) / 50
      
      # Rate of Non-Target (RONT)
      RONT <- sum(detected.patterns != target.pattern & detected.patterns != "none") / 50
      
      # Average Target Pattern Run Length (ATPRL)
      target.times <- detection.times[detected.patterns == target.pattern]
      run.lengths <- target.times - 231 + 1
      
      # If found before 231 then its wrong
      valid <- run.lengths[run.lengths > 0]
      ATPRL <- mean(valid)
      
      # Average Run Length Index (ARLIDX)
      ARLIDX <- ATPRL / ROT
      
      results <- rbind(results, data.frame(hidden.neurons=h, shift=s, cutoff=c, 
                                           ROT=round(ROT, 3), ATPRL=round(ATPRL, 3), 
                                           ARLIDX=round(ARLIDX, 3)))
    }
  }
}

results <- na.omit(results)

# To detect small shifts (-1)
# Set a ROT threshold at 0.9
final.models.sm <- results[results$ROT >= 0.90 & results$shift == -1, ]
round(final.models.sm, 2)

# Find the minimum ARLIDX
min.arlidx.sm <- min(final.models.sm$ARLIDX)
min.models.sm <- final.models.sm[final.models.sm$ARLIDX == min.arlidx.sm, ]

# To detect large shifts (-3)
# Set a ROT threshold at 0.9
final.models.la <- results[results$ROT >= 0.90 & results$shift == -3, ]
round(final.models.la, 2)

# Find the minimum ARLIDX
min.arlidx.la <- min(final.models.la$ARLIDX)
min.models.la <- final.models.la[final.models.la$ARLIDX == min.arlidx.la, ]

# 1 + N2 modules train data
# define a sequence of shifts
seq.shifts <- seq(-2, +2, by=0.2)
samples.2 <- 50
data.b <- list()
labels.b <- numeric()

for (s in seq.shifts) {
  data.2 <- t(replicate(samples.2, rnorm(n, 0, 0.5) + c(rep(0, n/2), rep(s, n/2))))
  data.b[[length(data.b) + 1]] <- data.2
  labels.b <- c(labels.b, rep(s, samples.2))
}

data.bb <- do.call(rbind, data.b)
train.data.b <- data.frame(data.bb, shift=labels.b)
colnames(train.data.b)[1:8] <- c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8")

# This network is representing the N2 module
N2 <- neuralnet(shift ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8,
                data = train.data.b, hidden = 15, act.fct = "logistic",
                linear.output = TRUE, stepmax = 1e6)

# Choose the best BPPR model
m <- neuralnet(normal + trend + shift + cyclic ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8,
               data = train.data, hidden = 5, act.fct = "logistic",
               linear.output = FALSE, stepmax = 1e6)
cutoff <- 0.95

# Set a random shift
shift1 <- -1.51

process <- rnorm(400, 0, 0.5)
process[231:400] <- process[231:400] + shift1

for (t in n:400) {
  curr <- process[(t - n + 1):t]
  input <- data.frame(matrix(curr, nrow=1))
  colnames(input) <- c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8")
  
  # N1 classifies the pattern
  predN1 <- compute(m, input)$net.result
  p <- as.numeric(predN1)
  max.prob <- max(p)
  ind <- which.max(p)
  
  if (max.prob >= cutoff && ind != 1) {
    patt <- pattern.names[ind]
    if (patt == "shift") {
      sh <- compute(N2, input)$net.result
    }
    break
  }
}

sh

# Create samples from -2 to 2
set.seed(123)
test.samples <- 200
true.shifts <- runif(test.samples, -2, 2)

test.data.list <- list()
for (s in true.shifts) {
  win <- rnorm(n, mean=0, sd=0.5) + c(rep(0, n/2), rep(s, n/2))
  test.data.list[[length(test.data.list) + 1]] <- win
}

test.df <- data.frame(do.call(rbind, test.data.list))
colnames(test.df) <- c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8")

# Use N2 for predictions in unknown data
predictions <- compute(N2, test.df)$net.result
estimated.shifts <- as.numeric(predictions)

# Estimate residuals
errors <- estimated.shifts - true.shifts

# Calculate RMSE and SE
RMSE <- sqrt(mean(errors^2))
SE <- sd(errors)

# 95% CI for the estimate
sh[1, 1] + 1.96 * c(-1, 1) * SE
