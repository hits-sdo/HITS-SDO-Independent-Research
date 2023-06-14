import unittest
import output_tile_info
import pyprojroot


class TestTileItems(unittest.TestCase):
    """
    Class that tests all the methods in the ParentTransformationsFromTile class
    """
    def setUp(self) -> None:
        """This class contains the unit tests that the tile object is valid."""
        self.tile_item = output_tile_info.ParentTransformationsFromTile(
            tile_name="Name",
            tile_image_type="jpeg",
            tile_is_valid=False,
            tile_time_stamp="blah",
            tile_f_out_path=pyprojroot.here(),
            tile_pixel_width=3,
            tile_pixel_height=2,
            is_padded=False,
            parent_img_width=27,
            parent_img_height=4,
            parent_file_is_valid=True,
            parent_file_file_type="blah",
            parent_file_source="blah")

    def test_tile_division_width(self):
        """This test will check if the tiles width evenly divides into the parent image width"""
        self.assertTrue(self.tile_item.parent_img_width % self.tile_item.tile_pixel_width == 0)

    def test_tile_division_height(self):
        """This test will check if the tiles height evenly divides into the parent image height"""
        self.assertTrue(self.tile_item.parent_img_height % self.tile_item.tile_pixel_height == 0)

    def test_tile_smaller_than_parent_width(self):
        """check to see if tile dimensions are less than parents dimensions"""
        self.assertTrue(self.tile_item.tile_pixel_width < self.tile_item.parent_img_width)

    def test_tile_smaller_than_parent_height(self):
        """check to see if tile dimensions are less than parents dimensions"""
        self.assertTrue(self.tile_item.tile_pixel_height < self.tile_item.parent_img_height)

    def test_correct_padding_width(self):
        """function to test member function of TileItem which calculates padding width
        to add null pixels to both columns ensuring the sun is centered."""
        print("\n\n\n", self.test_tile_smaller_than_parent_width, "\n\n\n")
        (test_left_padding, test_right_padding) = self.tile_item.calc_padding_width()
        temp = (self.tile_item.parent_img_width + test_left_padding + test_right_padding) % \
            self.tile_item.tile_pixel_width
        self.assertEqual(0, temp)

    def test_correct_padding_height(self):
        """function to test member function of TileItem which calculates padding height
        to add null pixels to both columns ensuring the sun is centered."""
        print("\n\n\n", self.test_tile_smaller_than_parent_height, "\n\n\n")
        (test_top_padding, test_bottom_padding) = self.tile_item.calc_padding_height()
        temp = (self.tile_item.parent_img_height + test_top_padding + test_bottom_padding) % \
            self.tile_item.tile_pixel_height
        self.assertEqual(0, temp)
        self.tile_item.tile_pixel_height = 5
        self.tile_item.parent_img_height = 45
        self.assertEqual(self.tile_item.calc_padding_height(), (0,0))

    def test_correct_padding(self):
        """function to test the correct padding from height and width"""
        self.tile_item.parent_img_height = 42
        self.tile_item.parent_img_width = 40
        self.tile_item.tile_pixel_height = 5
        self.tile_item.tile_pixel_width = 7
        self.assertEqual(self.tile_item.calc_padding_height(), (0,0))  

if __name__ == "__main__":
    unittest.main()
