# handling geographic data and geosom parameters
library(kohonen)
# call geosom by specifying data, coordinate labels, and spatial weight
# data as data.frame
# coords as a list of coordinate column labels
# k as a decimal number from 0 to 1
# norm as Boolean

# geosom function
"geosom" <- function(data, coords, k, norm, ...) {
  # subset spatial data
  geodata = data.matrix(data[,(names(data) %in% coords)])
  # subset non-spatial data
  data = data.matrix(data[,!(names(data) %in% coords)])
  spatial.weight = c(1, k)
  # if data to be normalised, we do it here
  # xyz cannot be scaled disproportionately
  if (isTRUE(norm)) {
    data = scale(data)
  }
  # disregard geography if spatial weight is zero
  if (k==0) {
    supersom(list(data), normalizeDataLayers=FALSE, ...)
  } else {
    supersom(list(data, geodata), user.weights=spatial.weight, normalizeDataLayers=FALSE, ...)
  }
}

"geosomgrid" <- function(x,y) {
  somgrid(x,y,"hexagonal", "gaussian")
}

# plot function 3 x 2
"show_results" = function(kohobj) {
  par(mfrow = c(2, 3))
  codes = getCodes(kohobj)
  # in the case of >1 attribute, only the first is plotted in "property" plot
  if (is.list(codes)) {
    plot(kohobj, type = "codes", whatmap = 1, shape="straight")
    plot(kohobj, type = "property", property = codes[[1]][, 1], shape="straight")
  } else {
    plot(kohobj, type = "codes", shape="straight")
    plot(kohobj, type = "property", property = codes[, 1], shape="straight")
  }
  plot_types = c("mapping", "counts", "dist.neighbours", "changes")
  for (plot_type in plot_types) {
    plot(kohobj, type = plot_type, shape="straight")
  }
}
