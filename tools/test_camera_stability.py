"""Pixel stability gates tolerate sparse raster noise, never scene changes."""
import json
from pathlib import Path
import unittest
from PIL import Image, ImageChops
from check_camera import validate_idle_capture


class CameraStabilityTests(unittest.TestCase):
    def setUp(self):
        self.camera = {'pose': {'position': [1, 2, 3]}, 'vertical_fov_degrees': 74}
        self.image = Image.new('RGB', (640, 360), (70, 110, 160))

    def validate(self, other, camera=None):
        return validate_idle_capture(self.image, other, self.camera,
                                     self.camera if camera is None else camera)

    def test_identical_and_sparse_color_noise(self):
        self.assertEqual(self.validate(self.image)['max_channel_change'], 0)
        other = self.image.copy()
        for x in range(600):
            other.putpixel((x, 20), (71, 108, 160))
        self.assertEqual(self.validate(other)['max_channel_change'], 2)

    def test_blank_motion_and_excess_noise_rejected(self):
        shifted = self.image.copy()
        for y in range(360):
            shifted.putpixel((20, y), (255, 255, 255))
        for image in (Image.new('RGB', self.image.size), shifted,
                      ImageChops.add(self.image, Image.new('RGB', self.image.size, (1, 0, 0)))):
            with self.subTest(image=image):
                with self.assertRaisesRegex(AssertionError, 'unstable'):
                    self.validate(image)

    def test_camera_change_rejected_even_with_identical_pixels(self):
        with self.assertRaisesRegex(AssertionError, 'pose'):
            self.validate(self.image, {**self.camera, 'vertical_fov_degrees': 75})

    def test_archived_cloud_pair(self):
        root = Path(__file__).resolve().parents[1]
        archive = root / 'artifacts/modal-alpamayo-validation-20260924T021911Z/alpamayo-validation-20260924T021911Z/renderer-check'
        if not archive.exists():
            self.skipTest('Optional actual cloud preflight archive is not present')
        stats = validate_idle_capture(Image.open(archive / 'tick0.png'),
                                      Image.open(archive / 'tick0_repeat.png'),
                                      json.loads((archive / 'tick0.json').read_text())['camera'],
                                      json.loads((archive / 'tick0_repeat.json').read_text())['camera'])
        self.assertEqual(stats['max_channel_change'], 3)
        self.assertAlmostEqual(stats['changed_pixel_fraction'], 798 / 230400)


if __name__ == '__main__':
    unittest.main()
