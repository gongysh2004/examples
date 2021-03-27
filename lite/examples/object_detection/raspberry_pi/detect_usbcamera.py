import time
import re
import numpy as np

from tflite_runtime.interpreter import Interpreter
import cv2

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels


def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  
  input_tensor[:, :] = cv2.resize(image,(300,300))
  


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results


def main():
  labels = load_labels('/tmp/coco_labels.txt')
  interpreter = Interpreter('/tmp/detect.tflite')
  interpreter.allocate_tensors()
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
  cam = cv2.VideoCapture(0)
  try:
      ret = True
      while ret:
          for i in range(10):
              ret, frame = cam.read()
          image = frame
          
          start_time = time.monotonic()
          results = detect_objects(interpreter, image, 0.5)
          elapsed_ms = (time.monotonic() - start_time) * 1000
          for obj in results:
              ymin, xmin, ymax, xmax = obj['bounding_box']
              xmin = int(xmin * CAMERA_WIDTH)
              xmax = int(xmax * CAMERA_WIDTH)
              ymin = int(ymin * CAMERA_HEIGHT)
              ymax = int(ymax * CAMERA_HEIGHT)
              classid = labels[obj['class_id']]
              score = obj['score']
              y = ymin + 10
              if ymin - 10 > 10:
                  y = ymin - 10
              cv2.rectangle(image, (xmin, ymin),(xmax, ymax), (255, 255, 255), 1)
              cv2.putText(image,classid+" "+ str(round(elapsed_ms)),(xmin, y),
                          cv2.FONT_HERSHEY_PLAIN,
                          1.5, (0, 255, 0),  1,
                          cv2.LINE_AA,False)
        
          cv2.imshow('r',image)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
          #print(results)
  finally:
      cam.release()
      cv2.destroyAllWindows()

    

if __name__ == '__main__':
  main()

