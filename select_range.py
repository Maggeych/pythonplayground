#!/usr/bin/python
import cv2
import numpy as np  


class CropArea:
    def __init__(self):
        # (B, G, R) color definitions.
        self._crsh = (0, 255, 255)  # Crosshair color.
        self._box = (255, 0, 0)  # Selection box color.


    def ask_from_user(self, window_name, image):
        self._cursor = [0, 0]  # Current cursor position.
        self._horz = [0, 0]  # Selected horizontal borders.
        self._vert = [0, 0]  # Selected vertical borders.

        # Announce the window and hook up the callback.
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        while(True):
            # Copy the original image in order to preserve it.
            cpy = image.copy()

            # Clip the crop boundaries.
            self._vert = np.clip(self._vert, 0, image.shape[0])
            self._horz = np.clip(self._horz, 0, image.shape[0])

            # Construct the crop's dimensionality and slice object.
            selection = (slice(*sorted(self._vert)), slice(*sorted(self._horz)))

            # Create the selection marker.
            shape = np.abs(np.diff(self._vert))[0], np.abs(np.diff(self._horz))[0], 3
            box = np.full(shape, self._box, dtype=np.uint8)

            # Blend the selection marker into the image copy.
            cpy[selection] = cv2.addWeighted(cpy[selection], 0.5, box, 0.3, 0)

            # Draw mouse crosshair.
            cv2.line(cpy, (self._cursor[0], 0), (self._cursor[0], cpy.shape[1]), self._crsh)
            cv2.line(cpy, (0, self._cursor[1]), (cpy.shape[0], self._cursor[1]), self._crsh)

            # Show the image copy.
            cv2.imshow(window_name, cpy)

            # Wait 20ms and check for keypresses.
            key = cv2.waitKey(20) & 0xFF
            if key == 114:  # R key.
                image = random_image()
            if key == 27:  # ESC key.
                break

        # Free and close the window.
        cv2.destroyAllWindows()
        

    def _mouse_callback(self, event, x, y, flags, param):
        self._cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._horz[0], self._vert[0] = self._cursor
        if flags & cv2.EVENT_LBUTTONDOWN:
            self._horz[1], self._vert[1] = self._cursor


def random_image():
    IMG_DIMENSIONS = (512, 512, 3)
    MIN_BRUSH_THICKNESS = 2
    MAX_BRUSH_THICKNESS = .18 * np.max(IMG_DIMENSIONS)
    COLOR_VARIATION = 50
    BLUR_KERNEL_SIZE = 1
    ITERATIONS = np.random.randint(2, 6)
    DISTANCE_SCALE = 20

    painting = np.zeros(IMG_DIMENSIONS, np.uint8)

    base_color = [
                    np.random.randint(50, 206), 
                    np.random.randint(50, 206),
                    np.random.randint(50, 206)
                ]

    repeat = int(np.ceil(4000.0 / ITERATIONS))
    for r in range(repeat):
        path = []
        position = np.array([np.random.randint(0, painting.shape[0]), 
            np.random.randint(0, painting.shape[1])], dtype=np.int32)
        thickness = int(MAX_BRUSH_THICKNESS - (1 - (repeat - r - 1) /
                        float(repeat - 1)) *
                (MAX_BRUSH_THICKNESS - MIN_BRUSH_THICKNESS))
        for i in range(ITERATIONS):
            rnd_direction = np.random.randint(-1, 2, size=2)
            rnd_distance = DISTANCE_SCALE * np.random.randint(.5 * thickness, 1 * thickness + 1)

            position = np.clip(position + rnd_direction * rnd_distance, [0, 0], painting.shape[:2])
            path.append(position)
            ITERATIONS = np.random.randint(2, 6)

        cv2.polylines(painting, [np.array(path)], True, (
                    base_color[0] + np.random.randint(-COLOR_VARIATION, COLOR_VARIATION + 1), 
                    base_color[1] + np.random.randint(-COLOR_VARIATION, COLOR_VARIATION + 1),
                    base_color[2] + np.random.randint(-COLOR_VARIATION, COLOR_VARIATION + 1)
                    ), thickness, cv2.LINE_AA)

    return cv2.GaussianBlur(painting, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), BLUR_KERNEL_SIZE)


image = random_image()
bound = CropArea()
bound.ask_from_user("Select the conveyor belt", image)

#print("Selected bounds:", bound.values, bound.lower)

