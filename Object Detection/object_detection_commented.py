# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Defining a function that will do the detections
def detect(frame, net, transform): 
# We define a detect function that will take as inputs, a frame, 
#a ssd neural network, and a transformation to be applied on the images, 
#and that will return the frame with the detector rectangle.
    
    height, width = frame.shape[:2] # We get the height and the width of the frame that is
    frame_t = transform(frame)[0] # We apply the transformation to our frame.	
    x = torch.from_numpy(frame_t).permute(2, 0, 1) # We convert the frame into a torch tensor.
    x = Variable(x.unsqueeze(0)) # We add a fake dimension corresponding to the batch.


    y = net(x) # We feed the neural network ssd with the image and we get the output y.
    detections = y.data # We create the detections tensor contained in the output y.

    #make a fake tensor that will be used to normalise the results of y.
    scale = torch.Tensor([width, height, width, height]) 
    # We create a tensor object of dimensions [width, height, width, height].


    #detections contains that in frame:  4 entries
    # [batch, number_of_classes, no_of_occurences of class, (score,x0,y0,x1,y1)]
    # if score > 0.6 ; then object considered to be present in frame.

    for i in range(detections.size(1)): # For every class:
        j = 0 
        # We initialize the loop variable j that will correspond to the occurrences of the class.
       
       # detections[0, i, j, 0] is the score of occurance j of a class i
       # detections[0, i, j, 1:] are the coordinates of that occurance
       # detections[0, i, j, 1:] * scale does normalization of the coordinates
       # We take into account all the occurrences j of the class i that have a matching score larger than 0.6.
        while detections[0, i, j, 0] >= 0.6: 
            pt = (detections[0, i, j, 1:] * scale).numpy() 
            # We get the coordinates of the points at the upper left and the lower right
            # of the detector rectangle.

            # we draw a rectangle using the corner cordinates:upper left and lower right.
            cv2.rectangle(frame, ( int(pt[0]), int(pt[1]) ), ( int(pt[2]), int(pt[3]) ), \
            (255, 0, 0), 2) # We draw a rectangle around the detected object.

            #labelmap is a dict that maps classes to the object names:
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) 
			# We put the label of the class right above the rectangle.

            j += 1 # We increment j to get to the next occurrence.
    
    return frame # We return the original frame with the detector rectangle and the label around the detected object.





# Creating the SSD neural network
# importing weights and not training so - mode is 'test'
net = build_ssd('test') # We create an object that is our neural network ssd.
net.load_state_dict(torch.load('trainedSSDmodel.pth', map_location = lambda storage, loc: storage))
 # We get the weights of the neural network from another one that is pretrained.

# Creating the transformation
 # We create an object of the BaseTransform class, a class that will do the required 
 #transformations so that the image can be the input of the neural network.
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))



# Doing some Object Detection on a video
reader = imageio.get_reader('funny_dog.mp4') # We open the video.
fps = reader.get_meta_data()['fps'] # We get the fps frequence (frames per second).
writer = imageio.get_writer('output.mp4', fps = fps)
 # We create an output video with this same fps frequence.

for i, frame in enumerate(reader): # We iterate on the frames of the output video:

	# We call our detect function (defined above) to detect the object on the frame.
    frame = detect(frame, net.eval(), transform) 

    writer.append_data(frame) # We add the next frame in the output video.
    print(i) # We print the frame number being processed frame.
writer.close() # We close the process that handles the creation of the output video.