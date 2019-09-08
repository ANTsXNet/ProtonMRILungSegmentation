library( ANTsR )
library( ANTsRNet )
library( keras )

args <- commandArgs( trailingOnly = TRUE )

if( length( args ) != 3 )
  {
  helpMessage <- paste0( "Usage:  Rscript doBrainExtraction.R",
    " inputFile outputFilePrefix reorientationTemplate\n" )
  stop( helpMessage )
  } else {
  inputFileName <- args[1]
  outputFileName <- args [2]
  reorientTemplateFileName <- args[3]
  }

classes <- c( "background", "leftLung", "rightLung" )
numberOfClassificationLabels <- length( classes )

imageMods <- c( "Proton" )
channelSize <- length( imageMods )

cat( "Reading reorientation template", reorientTemplateFileName )
startTime <- Sys.time()
reorientTemplate <- antsImageRead( reorientTemplateFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

resampledImageSize <- dim( reorientTemplate )

unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ),
  numberOfOutputs = numberOfClassificationLabels,
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = 16, dropoutRate = 0.0,
  convolutionKernelSize = c( 7, 7, 5 ), deconvolutionKernelSize = c( 7, 7, 5 ) )

cat( "Loading weights file" )
startTime <- Sys.time()
weightsFileName <- paste0( getwd(), "/protonLungMriWeights.h5" )
if( ! file.exists( weightsFileName ) )
  {
  weightsFileName <- getPretrainedNetwork( "protonLungMri", weightsFileName )
  }
load_model_weights_hdf5( unetModel, filepath = weightsFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

# Process input

startTimeTotal <- Sys.time()

cat( "Reading ", inputFileName )
startTime <- Sys.time()
image <- antsImageRead( inputFileName, dimension = 3 )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Normalizing to template" )
startTime <- Sys.time()
centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
centerOfMassImage <- getCenterOfMass( image )
xfrm <- createAntsrTransform( type = "Euler3DTransform",
  center = centerOfMassTemplate,
  translation = centerOfMassImage - centerOfMassTemplate )
warpedImage <- applyAntsrTransformToImage( xfrm, image, reorientTemplate )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

batchX <- array( data = as.array( warpedImage ),
  dim = c( 1, resampledImageSize, channelSize ) )
batchX <- ( batchX - mean( batchX ) ) / sd( batchX )

cat( "Prediction and decoding" )
startTime <- Sys.time()
predictedData <- unetModel %>% predict( batchX, verbose = 0 )
probabilityImagesArray <- decodeUnet( predictedData, reorientTemplate )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( " (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Renormalize to native space and write image" )
startTime <- Sys.time()
probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
  probabilityImagesArray[[1]][[2]], image )
antsImageWrite( probabilityImage, paste0( outputFileName, "Probability1.nii.gz" ) )
probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
  probabilityImagesArray[[1]][[3]], image )
antsImageWrite( probabilityImage, paste0( outputFileName, "Probability2.nii.gz" ) )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

endTimeTotal <- Sys.time()
elapsedTimeTotal <- endTimeTotal - startTimeTotal
cat( "\nTotal elapsed time:", elapsedTimeTotal, "seconds\n\n" )
