# Load required libraries
library(jsonlite)
library(ggplot2)
library(plotly)

# Read data
JsonData <- jsonlite::fromJSON(txt = "C:\\Users\\pedro\\Downloads\\laser_tracking_data.json")
JsonData <- as.data.frame(JsonData)  # Ensure it's a data frame

# Check the structure and column names
str(JsonData)
print(names(JsonData))

# Create the ggplot object without faceting
gg <- ggplot(data = JsonData, mapping = aes(x = x, y = y)) + 
  scale_y_reverse() +    # Reverse the y-axis
  geom_path() +         # Connect the points with a path
  theme_minimal()       # Apply a minimal theme

# Print ggplot object to check
print(gg)

# Convert to plotly and save
l <- plotly::ggplotly(gg)
htmlwidgets::saveWidget(l, "C:\\Users\\pedro\\Downloads\\sample.html")
