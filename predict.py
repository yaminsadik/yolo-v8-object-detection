from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


# Load the model
model = YOLO('best.pt')

# Perform prediction
results = model.predict(source='/home/sadik/objectDetection/test_pic.jpg')

# Print the results
print(results)

# Extract the first result
result = results[0]

# Convert BGR to RGB for plotting
result_img_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

# Plot the result
plt.imshow(result_img_rgb)
plt.axis('off')  # Turn off axis
plt.show()
