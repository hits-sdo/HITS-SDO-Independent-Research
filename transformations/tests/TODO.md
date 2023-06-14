
# Test our tiles so that we don't unevenly divide images (specification test)
^ should this go somewhere else? like in a bullet point?

## Potential Tests:
* To make sure we didn't slice our images: if final results are smaller than whatever the original imput was.
* Reverse is true; make sure we didn't make our slices bigger than original image.
    * We didn't slice outside of pizza (sliced the new wood table instead of pizza).
* Don't make slice so small such that it's 0.

## Questions:
* How do we check to see that our tiles don't unevenly divide images?
    * Will probably require more that one unit test.
* Which images and what we dividing?
    * How do we divide an image?
    * What does dividing mean? Will tiles overlap?
    * How do we divide the images? 

Notes:
* Kinds of developer testing that we do:
    1. Specification testing.
    2. Boundary testing.
    3. Structural testing.


/*      this block should go elsewhere
Red- writing failing test case.
Green- writing passing test case.
Refactor- improve passing test cases.

what we need: size of original image.
*/


