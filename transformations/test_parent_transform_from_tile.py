"""
Unittest to test functionality of our functions.
Importing parent_transform_from_tile so that we can access data members
Pyprojroot so we can work with relative paths instead of absolute paths
"""
import unittest
# import pyprojroot # in case we do need it
import parent_transform_from_tile

tile_pixel_width= 3
tile_pixel_height= 2

class TestParentTransformationsFromTile(unittest.TestCase):
    """
    Class to Test Parent Transformation From Tile
    This class contains the unit tests that the tile object is valid.
    """
    def setUp(self) -> None:
        """This function declares our variables"""
        self.parent_transform = parent_transform_from_tile.ParentTransformationsFromTile(
            parent_img_width=27,
            parent_img_height=4,
            parent_img_width_after_padding=0,
            parent_img_height_after_padding=0,
            parent_file_is_valid=True,
            parent_file_file_type="blah",
            parent_file_source="blah")

    def test_tile_division_width(self, tile_pixel_width):
        """This test will check if the tile's width evenly divides into the parent image(s) width"""
        self.assertTrue(self.parent_transform.parent_img_width % \
                    tile_pixel_width== 0)

    def test_export_padded_parent_to_file(self, tile_pixel_height, tile_pixel_width, filepath_output):
        # first call the export_padded_parent_to_file
        # 


                        #    def test_tile_division_height(self, tile_pixel_height):
#        """This test will check if the tile's height 
#        evenly divides into the parent image(s) height"""
#        self.assertTrue(self.parent_transform.parent_img_height % \
#                        tile_pixel_height == 0)
#
#    def test_tile_smaller_than_parent_width(self, tile_pixel_width):
#        """check to see if tile dimensions are less than parents dimensions"""
#        self.assertTrue(tile_pixel_width < \
#                        self.parent_transform.parent_img_width)
#
#    def test_tile_smaller_than_parent_height(self, tile_pixel_height):
#        """check to see if tile dimensions are less than parents dimensions"""
#        self.assertTrue(tile_pixel_height < \
#                        self.parent_transform.parent_img_height)
#
#    def test_correct_padding_width(self, tile_pixel_width):
#        """function to test member function of TileItem which calculates padding width 
#        to add null pixels to both columns ensuring the sun is centered."""
#        (test_left_padding, test_right_padding) = self.parent_transform.calc_padding_width(tile_pixel_width)
#        temp = (self.parent_transform.parent_img_width + test_left_padding + test_right_padding) % \
#            tile_pixel_width
#        self.assertEqual(0, temp)
#
#    def test_correct_padding_height(self, tile_pixel_height):
#        """function to test member function of TileItem which calculates padding height 
#        to add null pixels to both columns ensuring the sun is centered."""
#        (test_top_padding, test_bottom_padding) = self.parent_transform.calc_padding_height(tile_pixel_height)
#        temp = (self.parent_transform.parent_img_height + test_top_padding + test_bottom_padding) \
#            % tile_pixel_height
#        self.assertEqual(0, temp)
#
#        tile_pixel_height = 5
#        self.parent_transform.parent_img_height = 45
#        self.assertEqual(self.parent_transform.calc_padding_height(tile_pixel_height), (0,0))
#
#    def test_correct_padding(self, tile_pixel_height, tile_pixel_width):
#        """function to test the calculated height and width is correct"""
#        self.parent_transform.parent_img_height = 42
#        self.parent_transform.parent_img_width = 40
#        # tile_pixel_height = 5
#        # tile_pixel_width = 7
#        self.assertEqual(self.parent_transform.calc_padding_height(tile_pixel_height), (0,0))

if __name__ == "__main__":
    unittest.main()