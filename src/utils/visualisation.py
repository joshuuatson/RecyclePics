#will hopefully label and box items using OpenCV
import cv2

def draw_detections(image_path, detections, output_path = None):
    img = cv2.imread(image_path)

    for label, (x,y,w,h), bin_color, conf in detections:
        # color = (255, 0, 0)  # default: blue
        # if bin_color == "green":
        #     color = (0, 255, 0)
        # elif bin_color == "black":
        #     color = (0, 0, 0)
        # elif bin_color == "brown":
        #     color = (42, 42, 165)

        # cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        # text = f"{label} -> ({bin_color})"
        # cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        color_map = {
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "brown": (42, 42, 165),
            "black": (0, 0, 0),
            "grey": (128, 128, 128),
            "unknown": (0, 0, 255),
        }
        color = color_map.get(bin_color, (200, 200, 200))

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label_text = f"{label} -> {bin_color} ({conf:.2f})"
        cv2.putText(img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    if output_path:
        cv2.imwrite(output_path, img)
    else:
        cv2.imshow("Detections", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()