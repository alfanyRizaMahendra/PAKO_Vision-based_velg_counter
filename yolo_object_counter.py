import torch
import cv2
import yaml
from ultralytics import YOLO
from torchvision import transforms
import time
import numpy as np
import mysql.connector as db

# -------------------------- Setup Database Connection -------------------------------------
mydb = db.connect(
         host = "localhost",
         user = "root",
         password = "root",
         database = "auto_counter_painting"
         #connect_timeout=60
)

# post-data function
def post_data(request):
    mycursor = mydb.cursor()
    query = request
    mycursor.execute(query)
    mydb.commit()

# Define Line Cycle Time to prevent double input
time_prev = [time.time(), time.time()]
ct = 10 # in seconds
read_condition = [True, True]
    
# -------------------------- Setup videoo input and output ---------------------------------
video_source = 'velg_compilation.mp4'

# video output name
save_path = "detection_result/"
model = "YOLO"
epoch = str(75)
name = save_path + model + epoch + "_output_detection.avi"

# model path
model_path = 'YOLO/'
model_checkpoint = "best2.pt"
model_name = model_path + model_checkpoint

# set up detection thresshold 
thress = 0.85

# set up font scale 
font_scale = 1/1.5

# average fps variable
fps_container = np.array([])
# -----------------------------------------------------------------------------------

# used to record Video FPS
prev_frame_time = 0
new_frame_time = 0

model = YOLO(model_name)

# Load video file
cap = cv2.VideoCapture(video_source)

# Define classes
# Load the .yml file
with open('data.yaml', 'r') as f:
    class_info = yaml.safe_load(f)
    classes = class_info['names']

# Define colors for bounding boxes
colors = [(0,0,0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

# Initialize object counts
object_counts = {name: 0 for name in classes}

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(name, fourcc, 10.0, (416, 416))

# Define line in the middle of video
tolerance = 3
line_x = int(416/2)
line_y = int(416/2)
line_color = [(0, 0, 255), (0,255,0)]  
line_points = [[(line_x-tolerance, 0), (line_x+tolerance, line_y)], [(line_x-tolerance, line_y), (line_x+tolerance, line_y*2)]]

# Initialize object counter
total_count = 0
count = 0
obj = 0
loop = False
y_tambah = 0

# Loop over frames in the video
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Starting all timer
    time_now = [time.time(), time.time()]
    new_frame_time = time.time()

    # resize frame to 416x416
    frame = cv2.resize(frame, (416,416))

    # Convert frame to tensor
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    
    # Assuming `tensor` is your input tensor
    tensor_to_pil = transforms.ToPILImage()
    pil = tensor_to_pil(tensor[0])
        
    # Perform inference
    with torch.no_grad():
        output = model(pil)
        output = sorted(output, key = lambda x: x.boxes.conf, reverse=True)

    # # Parse output
    # boxes_list = output[0].boxes.xyxy
    # scores_list = output[0].boxes.conf
    # labels_list = output[0].boxes.cls
    
    # # Draw bounding boxes
    # for box, score, label in zip(boxes_list, scores_list, labels_list):
    #     if score > thress:
    #         x1, y1, x2, y2 = box.int().tolist()
    #         conf = format(float(score.item()), '.2f')
    #         color = colors[int(label)]
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    #         cv2.putText(frame, classes[int(label)] + ' ' + str(conf), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    
    cv2.rectangle(frame, line_points[0][0], line_points[0][1], line_color[0], thickness=-1)
    cv2.rectangle(frame, line_points[1][0], line_points[1][1], line_color[1], thickness=-1)

    #print(f'bbox < {line_points[0][0]} and bbox > {line_points[1][0]}')
    for detection in output:
            if detection is not None:
                boxes = output[0].boxes.xyxy
                scores = output[0].boxes.conf
                class_indices = output[0].boxes.cls.long()
                labels = [classes[i] for i in class_indices]
                
                #print(object_counts)
                cv2.rectangle(frame, line_points[0][0], line_points[0][1], line_color[0], thickness=-1)
                cv2.rectangle(frame, line_points[1][0], line_points[1][1], line_color[1], thickness=-1)
                
                # Draw bounding boxes and labels
                for box, score, label, color, class_detect in zip(boxes, scores, labels, colors, class_indices):
                    if score > thress:
                        x1, y1, x2, y2 = box.int().tolist()

                        # setting color different on each class 
                        color = colors[int(class_detect)]

                        # Get box center and check if it is above the line
                        box_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        cv2.circle(frame, box_center, 2, (0, 0, 255), -1)
                        
                        # Upper detector
                        if  (line_points[0][0][0] < box_center[0] < line_points[0][1][0]) and (box_center[1] <= line_points[0][1][1]) and read_condition[0]:

                            #read indicator
                            cv2.rectangle(frame, line_points[0][0], line_points[0][1], (240,240,240), thickness=-1)

                            #count object per class 
                            object_counts[classes[int(class_indices[obj])]] += 1
                            count += 1

                            #count total detection
                            total_count += 1
                            print()
                            print("-"*40)
                            print("Object passed line! Total count:", total_count)
                            print("-"*40)
                            print()

                            #prevent double in
                            read_condition[0] = False
                            time_prev[0] = time.time()
                        
                        # Lower Detector
                        if  (line_points[1][0][0] < box_center[0] < line_points[1][1][0]) and (box_center[1] <= line_points[1][1][1]) and read_condition[1]:
                            
                            #read indicator
                            cv2.rectangle(frame, line_points[1][0], line_points[1][1], (240,240,240), thickness=-1)

                            #count object per class 
                            object_counts[classes[int(class_indices[obj])]] += 1
                            count += 1

                            #count total detection
                            total_count += 1
                            print()
                            print("-"*40)
                            print("Object passed line! Total count:", total_count)
                            print("-"*40)
                            print()

                            #prevent double in
                            read_condition[1] = False
                            time_prev[1] = time.time()
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
                        cv2.putText(frame, f'{label}: {score:.2f} {obj}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)    
                        
                    # post-data to sql
                    if(count > 0):
                        querry = """INSERT INTO transaction (waktu_upload, id_product, product_name, qty, sum)
                            SELECT '%s', master_data.id, '%s', '%s', '%s'
                            FROM master_data
                            WHERE master_data.product_name = '%s'
                        """%(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), str(classes[int(class_indices[obj])]), str(count), str(object_counts[classes[int(class_indices[obj])]]), str(classes[int(class_indices[obj])]))
                        post_data(querry)
                        count = 0

                    obj += 1
                
                # Reset read condition
                if(time_now[0] - time_prev[0] >= ct):
                    read_condition[0] = True

                if(time_now[1] - time_prev[1] >= ct):
                    read_condition[1] = True

                # Show all calculated classes
                for key, value in object_counts.items():
                    if (key == "0"):
                        continue
                    key = str(key).replace("_"," ")
                    value = str(value)
                    show = key + '= ' + value 
                    cv2.putText(frame, show, (10, line_y + 120 + y_tambah), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240,240,240), 2)
                    y_tambah += 21

    y_tambah = 0
    obj = 0

    # Display fps
    fps = 1/(new_frame_time-prev_frame_time)
    fps_container = np.append(fps_container, fps)
    fps = format(fps, '.2f')
    fps = str(fps)
    text = "FPS = " + fps

    # update time counter
    prev_frame_time = new_frame_time

    # putting the FPS count on the frame
    cv2.putText(frame, text, (7, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    # Write the frame to the output video file
    out.write(frame)

    loop = True
    if cv2.waitKey(1) == ord('q'):
        break

average = np.average(fps_container)
average = format(average, '.2f')
print()
print("Average FPS = ", end = '')
print(average)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Note:
# Using GPIO input from conveyor can be the best way to disable counter when hanger is stop
