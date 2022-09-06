## Geographically-Aware Self-Organizing Maps in kohnonen 3.0

library("geosom")
library("Rcpp")

## 4 Corners example
## The points follow a uniform distribution in the geographical coordinate,
## within the rectangle limited by [(0,0),(20,5)]. In the non-geographical
## dimension there are three zones of high spatial autocorrelation, where the
## values of z are very similar among neighbouring points, with a uniform in
## [90,91] in two zones and [10,11] in another. There is also one area of
## "negative autocorrelation", where half the data points have z==0 and the
## other half have z==90. In the rest of the input space z has a uniform
## distribution in [0,100].
