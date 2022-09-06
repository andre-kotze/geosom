\name{expandMap}
\alias{expandMap}
\title{Expand a self-organising map}
\description{
Double the size of a map, imputing the codebookvectors of the new units
by averiging their immediate neighbours.
}
\usage{
expandMap(kohobj)
}
\arguments{
  \item{kohobj}{Object of class \code{"geosom"})}
}
\value{A new kohonen object, with a double size.}
\author{Ron Wehrens}

\examples{
}
\keyword{classif}
