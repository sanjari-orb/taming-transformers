import time
from PIL import Image
import io
import numpy as np
def serialize_image(image):
    start_time = time.time()
    with io.BytesIO() as stream:
        image.save(stream, format='PNG')
        serialized_data = stream.getvalue()
    end_time = time.time()
    return serialized_data, end_time - start_time

def deserialize_image(serialized_data):
    start_time = time.time()
    with io.BytesIO(serialized_data) as stream:
        image = Image.open(stream)
        image = np.array(image)
    end_time = time.time()
    return image, end_time - start_time

def main():
    # Define image sizes to test
    image_sizes = [(100, 100), (500, 500), (1000, 1000), (4000,4000)]  # Add more sizes as needed

    for width, height in image_sizes:
        # Create a random image
        image = Image.new('RGB', (width, height))

        # Serialize and measure time
        serialized_data, serialization_time = serialize_image(image)
        print(f"Serialization time for {width}x{height} image: {serialization_time:.6f} seconds")

        # Deserialize and measure time
        deserialized_image, deserialization_time = deserialize_image(serialized_data)
        print(f"Deserialization time for {width}x{height} image: {deserialization_time:.6f} seconds")

if __name__ == "__main__":
    main()

