## Geographically-Aware Self-Organizing Maps in kohnonen 3.0

library("geosom")

##########################################################
## The squareville example (Bacao, 2004)

## An artificial dataset to to evaluate the GeoSOM
## Squareville is a small town with square boundaries and an area of 10000 m2.
## Squareville has 100 houses evenly spaced with coordinates x in [5, 95] and
## y in [5, 95]. For each house we know the average salary, which is
## s in  [900 1000] for 35<=x<=65 and s in [0 100]

library("Rcpp")

## read the squareville dataset
## get the data
data = read.csv("data/squareville.csv", sep=";")
points = st_as_sf(data, coords = c("Y", "X"), crs = 4326)

## To the kohonen routine, we add a spatial discriminant


BCcode <-
  '#include <Rcpp.h>
typedef double (*DistanceFunctionPtr)(double *, double *, int, int);

  double brayCurtisDissim(double *data, double *codes, int n, int nNA) {
    if (nNA > 0) return NA_REAL;

    double num = 0.0, denom = 0.0;
    for (int i = 0; i < n; i++) {
        num += std::abs(data[i] - codes[i]);
        denom += data[i] + codes[i];
    }

    return num/denom;
  }

  // [[Rcpp::export]]
  Rcpp::XPtr<DistanceFunctionPtr> BrayCurtis() {
    return Rcpp::XPtr<DistanceFunctionPtr>(new DistanceFunctionPtr(&brayCurtisDissim));
  }'

sourceCpp(code = BCcode)

## alternatively, use the BC.cpp file in the inst/Distances
## subdirectory of the package:
## sourceCpp(file = paste(path.package("kohonen"),
##                        "Distances/BC.cpp", sep = "/"))
