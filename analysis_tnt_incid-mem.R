# Clear all before data
rm(list = ls(all.names = TRUE))

library(tidyverse)

# Get current directory
data_dir = "/Users/joecool890/Dropbox/UC-Davis/projects/tnt_visual-search/raw-data/"
root_dir = getwd()
setwd(data_dir)

# Useful Colors
ucd_blue = "#002855"
ucd_yellow = "#DAAA00"
black = "black"

# find data
data_files = list.files(pattern = "*.csv")

# load all datafile into one dataframe
data_frame = lapply(data_files, function(i){
  read_csv(i, skip=2, show_col_types = FALSE, col_names=TRUE)
}) %>% bind_rows()


filter(.data = diamonds, cut == "Ideal")
