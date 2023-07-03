import numpy as np
import pickle
import sys
sys.path.insert(0, '..')
from sdo_augmentation.augmentation_list import AugmentationList
from sdo_augmentation.augmentation import Augmentations
import streamlit as st
from PIL import Image
from augmentation_test import read_image
import tempfile
import json
from streamlit_cropper import st_cropper


def generate_augmentation():
    '''
    creates a random augmentation dictionary and
    assigns it to a session state variable
    '''

    augment_list = AugmentationList(st.session_state['instrument'])
    st.session_state['random_init_dict'] = augment_list.randomize()


def apply_augmentation(img, col2, user_dict, cord_tup):
    '''
    Applys the current augmentation settings to the selected image
    checked if a user selected a region of interest -> cord_tup
    And displays the augmented image to the user
    '''

    st.session_state['random_init_dict'] = user_dict

    # if user defined crop, perform cropped augmentation
    if cord_tup is not None and st.session_state['reg_fill'] == 'Yes':
        # get the center of the image when it's cropped
        # cord_tup: (ystart as 1, xstart as 0,height as 3 and width as 2)
        center_pos = (cord_tup[1] + (cord_tup[3]//2),
                      cord_tup[0] + (cord_tup[2]//2))

        # get distance from center to each side
        height, width = img.shape[:2]
        left_length = center_pos[1]
        right_length = width - center_pos[1]
        top_length = center_pos[0]
        down_length = height - center_pos[0]
        diff_h = right_length - left_length
        diff_v = top_length - down_length

        # This is the tuple describng horizontile paddeng >:) XD
        if diff_h > 0:
            h_padding = (diff_h, 0)
        else:
            h_padding = (0, -diff_h)

        # This is vertical padding parameters
        if diff_v > 0:
            v_padding = (0, diff_v)
        else:
            v_padding = (-diff_v, 0)

        # st.write((v_padding, h_padding))
        # np.pad(A, ((top,bottom), (left,right)), 'constant')
        if len(img.shape) == 3:
            pad_tuple = (v_padding, h_padding, (0, 0))
        else:
            pad_tuple = (v_padding, h_padding)

        padded_img = np.pad(img, pad_tuple,
                            mode='edge')

        augments = Augmentations(padded_img,
                                 st.session_state['random_init_dict'])

        augmented_img, title = augments.perform_augmentations(
            fill_void='Nearest')

        # define bounds of crop to match input image
        center_h, center_w = padded_img.shape[0]//2, padded_img.shape[1]//2
        crop_half_height, crop_half_width = cord_tup[3]//2, cord_tup[2]//2

        # update crop coordinates
        augmented_img = augmented_img[center_h - crop_half_height:
                                      center_h + crop_half_height,
                                      center_w - crop_half_width:
                                      center_w + crop_half_width]  
    elif cord_tup is not None:
        augments = Augmentations(img[cord_tup[1]:
                                     cord_tup[1] + cord_tup[3],
                                     cord_tup[0]:
                                     cord_tup[0] + cord_tup[2]],
                                 st.session_state['random_init_dict'])

        augmented_img, title = augments.perform_augmentations()

    elif st.session_state['reg_fill'] == 'No':
        augments = Augmentations(img,
                                 st.session_state['random_init_dict'])

        augmented_img, title = augments.perform_augmentations()

    else:
        augments = Augmentations(img,
                                 st.session_state['random_init_dict'])

        augmented_img, title = augments.perform_augmentations(fill_void='Nearest')

    col2.header(title)
    col2.image(augmented_img, use_column_width='always', clamp=True)
    js = json.dumps(st.session_state['random_init_dict'])

    # Convert transformation dictionary to JSON string
    augmented_img *= 255  # un-normalize image
    a_img = pickle.dumps(augmented_img)  # serialize into bytestring

    col2.download_button(label="Download Augmented Image",
                         data=a_img, file_name="augmented_img.p",
                         mime="image/p")
    col2.download_button(label="Download Augmentation Dictionary", data=js,
                         file_name="augmented_dict.json",
                         mime="application/json")


def main():
    ''' Start of the GUI appplication '''

    st.header("Image Augmentation Tool")
    # Allow user to upload an image to augment on
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["p", "jpg", "png"], key='img'
        )

    # Allow user to upload a dictionary of augmentations
    with st.form('my-form', clear_on_submit=True):
        uploaded_dict = st.file_uploader("Choose a dictionary", type=["json"])
        submitted = st.form_submit_button("UPLOAD!")

    col1, col2 = st.columns([1, 1])

    # Setup Sidebar
    st.sidebar.title("Settings")
    st.sidebar.radio("Fill voids after augmentatuion?", ('Yes', 'No'),
                     key='reg_fill')

    # Get user-selected instument
    imageInst = st.sidebar.selectbox("Select Instrument", ('euv', 'mag'),
                                     key='instrument',
                                     on_change=generate_augmentation)
    augment_list = AugmentationList(instrument=imageInst)

    # Setup the random_init_dict session state variable
    if 'random_init_dict' not in st.session_state:
        st.session_state['random_init_dict'] = augment_list.randomize()
    if uploaded_dict is not None and submitted is True:
        st.session_state["random_init_dict"] = json.load(uploaded_dict)

    # Allow user to select which augmentations to be applied
    user_dict = {}
    options = st.sidebar.multiselect("Choose augmentations: ",
                                     augment_list.keys,
                                     default=list
                                     (st.session_state
                                      ['random_init_dict'].keys()))

    # Loop through all the user selected augmentation options
    # and apply them to our dictionary
    for key in options:
        if 'rotate' == key:
            if key in st.session_state['random_init_dict']:
                state = float(st.session_state['random_init_dict'][key])
            else:
                state = 0.0

            rotation = st.sidebar.slider("Rotation:", -180.0, 180.0, state)
            user_dict['rotate'] = rotation

        if 'v_flip' == key:
            user_dict['v_flip'] = True

        if 'h_flip' == key:
            user_dict['h_flip'] = True

        if 'blur' == key:
            user_dict['blur'] = (2, 2)

        if 'brighten' == key:
            if key in st.session_state['random_init_dict']:
                state = float(st.session_state['random_init_dict'][key])
            else:
                state = 1.0

            brighten = st.sidebar.slider("Brighten:", 0.5, 1.5, state)
            user_dict['brighten'] = brighten

        if 'zoom' == key:
            if key in st.session_state['random_init_dict']:
                state = float(st.session_state['random_init_dict'][key])
            else:
                state = 1.0

            zoom = st.sidebar.slider("Zoom:", 0.5, 5.0, state)
            user_dict['zoom'] = zoom

        if 'translate' == key:
            if key in st.session_state['random_init_dict']:
                state = st.session_state['random_init_dict'][key]
            else:
                state = (0, 0)

            x_translate = st.sidebar.slider(
                "X-Axis Translation:", -10, 10, state[0])
            y_translate = st.sidebar.slider(
                "Y-Axis Translation:", -10, 10, state[1])

            user_dict['translate'] = (x_translate, y_translate)

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # write to temp file
            temp_file.write(uploaded_file.read())
            # get path
            file_path = temp_file.name
            temp_file.close()
            img = read_image(file_path, uploaded_file.name.split('.')[-1])
            col1.header("Original Image")
            cordTuple = None

        # Prompt User to crop if they clicked the "Crop Image" button
        crop_button_clicked = st.button("Crop Image")
        check_crop = (
            "crop_button" in st.session_state and
            st.session_state["crop_button"] is True
            )

        if crop_button_clicked or check_crop:
            st.session_state["crop_button"] = True

            with col1:
                pilImg = Image.fromarray(np.uint8(255*img))

                # TODO: Cannot figure out how to get column width
                # So used constant 350
                aspect_ratio = img.shape[1]/img.shape[0]
                size = 350
                pilImg = pilImg.resize((int(size*aspect_ratio), size))

                # Prompt to select bounds
                # using the streamlit cropper library
                cropped_coords = st_cropper(pilImg, realtime_update=True,
                                            box_color='#0000FF',
                                            aspect_ratio=None,
                                            should_resize_image=True,
                                            return_type='box')

                scaling_factor = img.shape[0] / size
                cordTuple = tuple(map(int, cropped_coords.values()))

                # scale all values
                cordTuple = tuple(
                    [int(x * scaling_factor) for x in cordTuple]
                    )

                cropped_image = img[
                    cordTuple[1]:cordTuple[1] + cordTuple[3],
                    cordTuple[0]:cordTuple[0] + cordTuple[2]
                ]

                # Show Preview
                st.subheader("Cropped Image")
                st.image(cropped_image, use_column_width='always', clamp=True)

        else:
            col1.image(img, use_column_width='always', clamp=True)

        st.sidebar.button(
            "Random augmentation", on_click=generate_augmentation
            )

        st.sidebar.button(
            "Apply augmentation", on_click=apply_augmentation,
            args=([img, col2, user_dict, cordTuple])
            )


if __name__ == '__main__':
    main()
