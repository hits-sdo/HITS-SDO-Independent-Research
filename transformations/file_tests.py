import unittest
import input_file_info
import os
from pyprojroot import here
import sys

root = here()
# root = here(".here")
# sys.path.append(str(root))
# path = sys.path[-1][:-5] # variable that gets the last element in the sys.path list and makes
# it a general path by cutting of the last 5 characters
JPG_FILE = "blahg.jpg"
TXT_FILE = "dummy.txt"


class TestFileMethods(unittest.TestCase):
    """Class to unit test file methods.

    Attributes:
        InputFileItem: user defined type
    """

    def setUp(self) -> None:
        """This function sets up all the test variables for our Test_File_Methods.
        Args: None
        Returns: None

        """
        self.input_file = input_file_info.InputFileItem(
            name=JPG_FILE,
            image_type="jpg",
            is_valid=True,
            time_stamp="blah",
            f_path=str(root),
            is_remote=True,
        )

        self.input_file2 = input_file_info.InputFileItem(
            name=TXT_FILE,
            image_type="txt",
            is_valid=True,
            time_stamp="blah",
            f_path=str(root),
            is_remote=True,
        )

    def test_is_jpg(self):
        """Tests if a file name ends in .jpgen
        Args: None
        Returns: None
        """
        tempfile = self.input_file.name.lower()
        self.assertRegex(tempfile, "\.*jpg|\.*jpeg")

    def test_file_exists(self):
        """Tests is the File exists
        Args: None
        Returns: None
        """
        with open(self.input_file2.name):
            self.assertIsNotNone(self.input_file2.name)

    def test_file_is_named(self):
        """Tests if the file has a valid name
        Args: None
        Returns: None
        """
        self.assertIsInstance(self.input_file.name, str)

    def test_path_exists(self):
        """Tests if the project (transformations) folder exists
        Args: None
        Returns: None
        """

        file_path = os.path.join(self.input_file.f_path, "transformations")
        print("File path is : ", file_path)
        self.assertTrue(os.path.exists(file_path))

    def test_same_image(self):
        """."""
        # self.assertEqual()
        pass

    def tearDown(self):
        """_summary_"""
        self.dummy = "dummy"

    # https://docs.python.org/3/library/unittest.html

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)


if __name__ == "__main__":
    unittest.main()
