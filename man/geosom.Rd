\name{geosom}
\alias{geosom}
\alias{geosomgrid}
\alias{show_results}
\title{Geographic Self-Organising Maps}
\usage{
geosom(data, coords, k ,norm, ...)
geosomgrid(x, y)
show_results(kohobj)
}
\description{
    A GeoSOM is a spatially aware extension of a conventional SOM
    Function \code{geosom} is a wrapper for the kohonen \code{supersom} function,
    and parses input data and geographic parameters before calling the \code{supersom} 
    function from package \code{kohonen}.
    Function \code{geosomgrid} is a utility function for quickly generating non-toroidal 
    hexagonal neural networks for use in GeoSOM analysis
    Function \code{show_results} is a utility function to view SOM results at a glance, 
    with 6 plot types displayed in a single figure.
}
\arguments{
    \item{data}{Input data as data.frame}
    \item{coords}{A list of spatial coordinate column labels in the dataframe}
    \item{k}{The weight of the spatial influence (as a multiple of the weight of 
        non-spatial attributes)}
    \item{norm}{Whether to normalise non-spatial data. Coordinate data is never normalised}
    \item{...}{Further arguments for the \code{supersom} function
    presented to the \code{geosom} wrapper.}
    \item{x}{Horizontal size of SOM grid}
    \item{y}{Vertical size of SOM grid}
    \item{kohobj}{an object of class \code{kohonen}.}
}
\examples{
    data(squareville)
    geogrid = geosomgrid(6, 10)
    geosom.model = geosom(data=squareville, 
                        coords = c("X", "Y"), 
                        k=3, 
                        norm=norm,
                        grid = geogrid)
    show_results(geosom.model)
}
