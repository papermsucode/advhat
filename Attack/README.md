## Preparing an attack

1. First, make a full-face photo of attacked person, a full-face photo in a hat, and a 
full-face photo in a hat with an example sticker on the hat. To be sure that you use a
sticker with the correct size follow instructions:
    1. Download an example.png.
    2. Open downloaded image with standard Windows print util.
    3. Choose the regime with 4 photos per page (9 x 13 sm).
    4. Uncheck the box "Fit picture to frame".
    5. Print page with an example sticker.
    6. Cut out the sticker and put in on the hat.
    
2. Use the next command to prepare photos:

`python3 face_preparation.py PATH_TO_THE_IMAGE`

3. You need to find parameters for the sticker position initialization. Use the next 
command to find these parameters:

`python3 face_preparation.py PATH_TO_THE_IMAGE_WITH_HAT_ONLY --mask`

It will show sticker placement with default parameters. Change parameters until the 
image looks like a prepared image with the sticker. You can see the parameters using `--help`
flag.

4. Download TensorFlow ArcFace model 
[here](https://drive.google.com/file/d/1fb70KgMRSmaEUF5cJ67BCD_DmTPCR5uJ/view?usp=sharing).

5. Launch an attack preparation:

`python3 attack.py PATH_TO_THE_PREPARED_IMAGED_WITH_HAT PATH_TO_THE_TF_MODEL --anchor_face 
PATH_TO_THE_PREPARED_IMAGE_WITHOUT_HAT (sticker position parameters in the same format from
the third step)`

6. Print the obtained sticker, put it on the hat as before, and make a new photo with the sticker.

7. Use "face_preparation.py" again to prepare a new photo and "cos_tf.py" to calculate a new similarity.

`python3 cos_tf.py PATH_TO_THE_PREPARED_IMAGE_WITHOUT_HAT PATH_TO_THE_PREPARED_IMAGE_WITH_HAT_ONLY` - baseline similarity


`python3 cos_tf.py PATH_TO_THE_PREPARED_IMAGE_WITHOUT_HAT PATH_TO_THE_PREPARED_IMAGE_WITH_THE_NEW_STICKER` - final similarity

### Notes

Note that our printer has good color rendering, that is why NPS-loss does not make influence in our experiments.
You may need to add NPS-loss for your printer.
