# Tailgating Detection using PyTorch & OpenCV
### Final Presentation Video:
* [Youtube](https://youtu.be/Xb545qwHpEk) 
* [Drive](https://drive.google.com/file/d/1JDm7MpOG-HvCXqUzP3sGkZLh2_D3vHjG/view?usp=sharing)

Tailgating Detection using Instance Segmentation and bounding box tracking via a fine-tuned Mask RCNN model.

## Abstract:
The issue of security is paramount to any organisation, especially organisations like large corporations. These corporations not only house employees whose safety is an issue but also intellectual property whose theft could bankrupt a company. Most companies nowadays have multiple entries which cannot all be manually monitored leading to automated entries and surveillance via CCTVs. To enter a building an employee must swipe his ID card at the entrance after which the door unlocks to let him in. This method has been successful in most instances, however it has one glaring flaw. A motivated intruder could lurk near the entrance and sneak inside behind an unsuspecting employee unnoticed. This security flaw has been exploited numerous times and the only available solution currently is to place a security guard at the entrance. This solution is both inefficient and expensive as it requires us to hire a guard and is still susceptible to human error. Our proposed solution is to use the installed CCTVs at the entrance to detect instances of tailgating by DeepLearning models. This information is passed on to security via an alert whenever tailgating is detected. We also propose to capture the information about the intruder by capturing his image from within the bounding box. This information would be critical in identifying the individual by security personnel so that he can be apprehended immediately. 

## Solution:
It works in three steps:
* The first step uses the model to segment human subjects from the background and other objects. The segmentation is done by classifying every single pixel in the input frame to generate a mask of the human. We also output a bounding box around the subjects detected. The model used by us would be a Mask RCNN model with a ResNet50 backbone pre-trained on the famous COCO dataset. We then modify the architecture and fine-tune the model on the Penn Fudan Pedestrian Dataset to specialize in detecting human subjects. This gives us better accuracy as well as computational performance.
* The second step uses the bounding box output from the model to calculate the centroid of the object. The centroid is then tracked by an object tracking framework. During the installation of the camera the user is asked to draw the region of entrance via a GUI to help the model identify where the gate is. Whenever the card is swiped the model immediately starts tracking the centroids in the frame. It counts the number of points entering the gate region. If the number of people entering is more than the number of swipes then we report tailgating.
* When tailgating is detected the model captures the mask of the tailgating individual and sends a mail to the security personnel as well as the employee whose id was swiped with a photo of the tailgating individual. The photo of the individual is captured from the mask.

## Industry Application:

The problem we are trying to solve is a security flaw in the current paradigm of automatic entrance security systems. Tailgating is when an intruder gets access to the building by entering immediately after a verified employee without flagging the system. Our proposed idea involves no manual human intervention and can be easily implemented in almost all entrances with need for no additional equipment. The scalability of this idea is important as it should be able to integrate well with existing security systems with minimum installation.

This problem is very important to the industry. Tailgating gives the individual unrestricted access to the premise. This could be a safety concern for the employees of the organisation. It could also give the individual access to crucial information of the company like the details of their new product leading to huge losses to the company. The current solution to the problem involves hiring a security to guard to monitor the entrances. This has two downsides. Firstly, large companies have multiple entrances requiring multiple personnel to survey them. This is expensive to the company. Secondly, even manual surveillance is error-prone and slow. With a large number of entrances there are more chances to miss. A human is also slower at reacting to this intrusion than an automated system and may not be able to identify the intruder. Our idea solves all these problems by introducing an automatic robust system.

## License:

This project is licensed under the [MIT License](LICENSE).
