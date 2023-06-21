import unittest
import deepbooru
import PIL

class TestDeepdanbooru(unittest.TestCase):
    def test_tagger(self):
        dbu = deepbooru.DeepDanbooru()
        dbu.load()
        dbu.start()

        testImage = PIL.Image.open("../test_assest/ddb.jpg")
        result = dbu.tag_multi(testImage)
        print(result)
        dbu.stop()




if __name__ == '__main__':
    unittest.main()
