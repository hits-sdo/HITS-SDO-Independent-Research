^  *-*      
# Steps from reading file to outputting and saving tiles in a directory
- [ ] File source (location)
    - local directory
    - server (define db)
- check what kind of file it is jpeg, fits and make sure the images are only those two 
- check if the file path exists depending on what machine egnostic their on
- Make sure the python package we install can run on multiple operating systems
- Make test to check for file corruption
- [ ] Define File Class
  - [ ] Source
  - [ ] File/image type
  - [ ] isValid bool to check for errors
  - [ ] file_footprint{isValid: , file_type: , source: }
  - [ ] Member Functions
	- [ ] Getter functions
      - Get file name
      - Get image type
      - Get isValid
      - Get numpy array
      - get timestamp 
      -get numpy
      -fetch remote
      -fetch local
      -
    - [ ] Setter functions

      - Set file name
      - Set image type
      - Set isValid (not publicly exposed)
      - Set numpy array
      - handle both png or fits, converting into numbpy <== revisit
    - [ ] __string__ (observer function)
      - Print fucntion to check to see if the variables were assigned 

- [ ] Define Preprocessing Class
  - [ ] Data Attributes
   - imageSource
   - imageShape
   - preprocessing_footprint{brightness: , scale: , contrast: , paddingType: }
      tilingFootprint
        tileWidth: tuple/dictionary
        tileHeight: tuple/dictionary
        timeInterval: tuple/dictionary
  - [ ] get image shape
  - [ ] get image size
  - [ ] get min/max pixels
  - [ ] create standard (& define) brightness
  - [ ] padding algorithm
  - [ ] standardize pixel dimensions
    - scale pixels
  - [ ] getFrequencies
**********************  
- [ ] Tiling Class
  - [ ] tf.images.extract_patches to divide into patches
    - [ ] Should we use the patchify library?
    - [ ] (Strides, rates...?)
  - [ ] tilingFootprint - dictionary
    - [ ] json of changes
    - [ ] tile dimensions
    - [ ] strides
    - [ ] etc...
  Member Functions 
    - [ ] Retrieve file width/height, desired tile width/height -> check if divides evenly else calc padding

- [ ] Define Dataloader

      
	  
  - Does TensorFlow work for PNG and FITS? => Both images are converted to 'tensors' (vector array STL container: https://www.tensorflow.org/guide/tensor)

- [] Open a file
  - Check if file exists and in specified directory
- [] Define how much the file is being read

- [ ] Save tiles into an output directory
  - Check if directory exists...

- how we're going to use this & if it's going to be a helper function that helps add padding to the parent image

    # team yellow - parent: 1024 x 1024
    # - tiles: 64 x 64

    # team red - We don't actually need to save the tiles as long as we have the (width x height)
    # and (starting row x starting column)

    #=======Intention for Fri 3/31/23=======
    # using the amount of padding we need (now global variables) to calculate
    # the number of pixels we need to add for padding
    # also add in the width and height padding methods a statement to redefine
    # the value of the padding variables

    Continue cleaning up based on Linter feedback, 
    double check the object that we are creating, 
    and double check over our classes



