
import numpy as np
import cv2
import time
import pathlib


"""
a blueprint for a bounded box with its corresponding name,confidence score and 
"""
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
COCO_FILE = ROOT_DIR / "data" / "coco.names"
if not COCO_FILE.exists():
    COCO_FILE = ROOT_DIR / "datas" / "coco.names"
if COCO_FILE.exists():
    with open(str(COCO_FILE), 'rt') as f:
        COCO_CLASSES = f.read().rstrip('\n').split('\n')
else:
    COCO_CLASSES = []

class BoundedBox:
    def __init__(self, xmin, ymin, xmax, ymax, ids, confidence):
        self.classes = COCO_CLASSES
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax       
        self.ymax = ymax
        self.name = self.classes[ids] if ids < len(self.classes) else "unknown"
        self.confidence = confidence   


"""
a blueprint that has lanes as lists and give queue like functionality 
to reorder lanes based on their turn for green and red light state
"""

class Lanes:
    def __init__(self,lanes):
        self.lanes=lanes
    
    def getLanes(self):
        return self.lanes
    
    def lanesTurn(self):
       return self.lanes.pop(0)

    def enque(self,lane):
       return self.lanes.append(lane)

    def lastLane(self):
       return self.lanes[len(self.lanes)-1]

"""
a blueprint that has lanes as lists and give queue like functionality 
to reorder lanes based on their turn for green and red light state
"""
class Lane:
    def __init__(self,count,frame,lane_number):
        self.count = count
        self.frame = frame
        self.lane_number = lane_number
    
"""
given lanes object return a duration based on comparison of each lane vehicle count
"""
def schedule(lanes):
    standard=10 #standard duration
    reward =0  #reward to be added or subtracted on the standard duration
    turn = lanes.lanesTurn()
    
    for i,lane in enumerate(lanes.getLanes()):
        if(i==(len(lanes.getLanes())-1)):
            reward = reward + (turn.count-lane.count)*0.2
        else:
            reward = reward + (turn.count-lane.count)*0.5
    scheduled_time = round((standard+reward),0)
    # minimum duration guard
    if scheduled_time < 3:
        scheduled_time = 3
    lanes.enque(turn)
    return scheduled_time
       
"""
given duration and lanes, returns a grid image containing frames of each lane with
their corresponding waiting duration
"""   

def display_result(wait_time,lanes):
    green = (0,255,0)
    red  = (0,0,255)
    yellow= (0,255,255)
    lane_frames = {}
    lane_list = lanes.getLanes()
    total_lanes = len(lane_list)
     
    for i, lane in enumerate(lane_list):
        lane.frame = cv2.resize(lane.frame,(1280, 720)) 
        
        if(wait_time<=0 and (i==(total_lanes-1) or i==0)):
           color=yellow
           text="yellow:2 sec"
        elif(wait_time>=0 and i==(total_lanes-1)):
            color = green 
            text=f"green:{int(wait_time)} sec"
        else:
            color=red
            text=f"red:{int(wait_time)} sec"

        lane.frame = cv2.putText(lane.frame,text,(60,105),cv2.FONT_HERSHEY_SIMPLEX,3,color,5)
        lane.frame = cv2.putText(lane.frame,f"vehicle count:{lane.count}",(60,195),cv2.FONT_HERSHEY_SIMPLEX,2,color,4)
        lane_frames[lane.lane_number] = lane.frame

    f1 = lane_frames.get(1, lane_list[0].frame)
    f2 = lane_frames.get(2, lane_list[1 % total_lanes].frame)
    f3 = lane_frames.get(3, lane_list[2 % total_lanes].frame)
    f4 = lane_frames.get(4, lane_list[3 % total_lanes].frame)

    hori_image = np.concatenate((f1, f2), axis=1)
    hori2_image = np.concatenate((f3, f4), axis=1)
    all_lanes_image = np.concatenate((hori_image, hori2_image), axis=0)
    
    return all_lanes_image


# given detecteed boxes, return number of vehicles on each box
def vehicle_count(Boxes):
    vehicle=0
    for box in Boxes:
        if box.name in ("car", "truck", "bus", "motorbike"):
            vehicle=vehicle+1  
    return vehicle

# given the grid dimension, returns a 2d grid
def _make_grid(nx=20, ny=20):
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

def drawPred(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), thickness=4)
    label = f"{COCO_CLASSES[classId] if classId < len(COCO_CLASSES) else ''}: {conf:.2f}"
    cv2.putText(frame, label, (left, max(top - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return frame

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    ratioh, ratiow = frameHeight / 320, frameWidth / 320
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = int(np.argmax(scores))
            confidence = float(scores[classId])
            if confidence > 0.4 and float(detection[4]) > 0.4:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(confidence)
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
    correct_boxes = []
    if len(indices) > 0:
        for i in indices:
            idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else int(i)
            box = boxes[idx]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            bbox = BoundedBox(box[0], box[1], box[2], box[3], classIds[idx], confidences[idx])
            correct_boxes.append(bbox)
            frame = drawPred(frame, classIds[idx], confidences[idx], left, top, left + width, top + height)
    return correct_boxes, frame


"""
interpret the ouptut boxes into the appropriate bounding boxes based on the yolo paper 
logspace transform
"""
def modify(outs,confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
    num_classes = len(COCO_CLASSES)
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    nl = len(anchors)
    na = len(anchors[0]) // 2
    no = num_classes + 5
    grid = [np.zeros(1)] * nl
    stride = np.array([8., 16., 32.])
    anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, 1, -1, 1, 1, 2)

    z = []  # inference output
    for i in range(nl):
        bs, _, ny, nx,c = outs[i].shape  
        if grid[i].shape[2:4] != outs[i].shape[2:4]:
            grid[i] = _make_grid(nx, ny)

        y = 1 / (1 + np.exp(-outs[i])) 
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * int(stride[i])
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        z.append(y.reshape(bs, -1,no))
    z = np.concatenate(z, axis=1)
    return z


"""
given each lanes image, it inferences using trt engine on the image, return lanes object
containg processed image and waiting duration for each image
"""
def final_output_tensorrt(processor,lanes):
    for lane in lanes.getLanes():
        lane.frame=cv2.resize(lane.frame,(1280,720))
        output = processor.detect(lane.frame)
        dets = modify(output)
        boxes,frame = postprocess(lane.frame,dets)
        count = vehicle_count(boxes)
        lane.count= count
        lane.frame=frame
    return lanes

"""
given each lanes image, it inferences onnx model on the image, return lanes object
containg processed image and waiting duration for each image
"""
def final_output(net,output_layer,lanes):
    for lane in lanes.getLanes():
        lane.frame=cv2.resize(lane.frame,(1280,720))
        blob = cv2.dnn.blobFromImage(lane.frame, 1 / 255.0, (320, 320),
            swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(output_layer)
        dets = modify(layerOutputs)
        boxes,frame = postprocess(lane.frame,dets)
        count = vehicle_count(boxes)
        lane.count= count
        lane.frame=frame
    return lanes
