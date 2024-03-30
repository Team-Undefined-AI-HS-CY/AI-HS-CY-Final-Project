V1
EnglishFnt was a dataset with many different fonts, but it wasn't images from license plates, that meant that the model would train on some images that there is no way it would ever encounter in the wild
E.g (insert wheel image here)
Which led into wrong conclusions and as the characters were too far off(Different fonts of the same character, were vastly different) it would output very false results.
Even though we tried to approach it with Dropout and regularization the model would still overfit too much on the training data and would not bring any results of value when tested on actual license plates

V2
CNN_Letters was a much better dataset trying to solve all the problems that were encountered previously as it was directly extracted and labelled from actual car license plates(which makes it perfect for our situation)
But unfortuanetly we felt that it was still no where near encompassing enough to cover all different fonts that could be seen in the wild. In addition we found that some common variations of characters were not included
E.g (insert 3)
This was because the dataset was only from car license plates in Belgium. Thus we still somehow had to increase the variation to get better results.

V2.5
To resolve the problems of V2 we took it upon our own to find common variations that were missing and make our own extraction and labelling.
Thus a new dataset("Manual") was born
(image of manual here)
That dataset was then included when training number V2.5
In addition just to make sure that the model was able to distinguish between charactes(it had enough nodes to really pull out the differences), we added one more Dense layer of 64 nodes just before the output

V3
Having all the expirience of the 3 previous models, we now had a new course of action.
Firstly we wanted to include as much data as possible to allow it to generalise.
Thus we opted for using the EnglishFnt dataset but only include selected fonts that were actually useful towards our goal
Secondly we wanted to make sure that the model was able to distinguish between characters, thus we added BatchNormalization after the 3rd Conv2D layer, while also increasing the Dropout to ensure it doesn't overfit
In summary V3 uses all the datasets with cherry-picked data from EnglishFnt and some incresed regularization to ensure it doesn't overfit


| Name | Datasets                                           |
| ---- | -------------------------------------------------- |
| V1   | EnglishFnt                                         |
| V2   | CNN_Letters                                        |
| V2.5 | CNN_Letters, Manual                                |
| V3   | EnglishFnt(Selected fonts only),CNN_Letters,Manual |

V1
|Architecture|
|------------|
|Input(100x50)|
|Conv2D(32, (3x3))|
|Conv2D(64, (3x3))|
|Conv2D(128, (3x3))|
|MPL(2x2, strides=(2x2), padding='same')|
|Dropout(0.25)|
|Flatten()|
|Dense(128, activation='relu')|
|Dropout(0.5)|
|Dense(len(COMBINED_CHARS), activation='softmax')|

V2
|Architecture|
|------------|
|Input(100x50)|
|Conv2D(32, (3x3))|
|Conv2D(64, (3x3))|
|Conv2D(128, (3x3))|
|MPL(2x2, strides=(2x2), padding='same')|
|Dropout(0.25)|
|Flatten()|
|Dense(128, activation='relu')|
|Dropout(0.3)|
|Dense(len(COMBINED_CHARS), activation='softmax')|

V2.5
|Architecture|
|------------|
|Input(100x50)|
|Conv2D(32, (3x3))|
|Conv2D(64, (3x3))|
|Conv2D(128, (3x3))|
|MPL(2x2, strides=(2x2), padding='same')|
|Dropout(0.25)|
|Flatten()|
|Dense(128, activation='relu')|
|Dense(64, activation='relu')|
|Dropout(0.3)|
|Dense(len(COMBINED_CHARS), activation='softmax')|

V3
|Architecture|
|------------|
|Input(100x50)|
|Conv2D(32, (3x3))|
|Conv2D(64, (3x3))|
|BatchNormalization()|
|Conv2D(128, (3x3))|
|MPL(2x2, strides=(2x2), padding='same')|
|Dropout(0.25)|
|Flatten()|
|Dense(128, activation='relu')|
|Dense(64, activation='relu')|
|Dropout(0.4)|
|Dense(len(COMBINED_CHARS), activation='softmax')|
